from __future__ import annotations

import math

import torch
import torch.nn as nn
from dataclasses import dataclass
from synthefy_nori.model.layer import EncoderBaseLayer, LayerStack, RMSNorm
from typing import Any, Callable, Literal


@dataclass
class ContextCache:
    """A reusable encoding of a fixed context (train) table for cached regression.

    Produced once by :meth:`FeaturesTransformer.build_context_cache` and consumed
    by :meth:`FeaturesTransformer.apply_context_cache` for any number of query
    batches, so the O(N_train) context forward (the per-layer row-attention K/V
    build) is paid ONCE and amortized across queries instead of recomputed per
    call. Everything here is train-derived and query-independent:

    - ``caches``: per-layer sequence-attention K/V from ``build_train_cache``
      (already int8/host-offload-scaled per the ``cache_dtype``/``offload`` used).
    - ``feature_pos_emb``: the (randomly drawn) feature positional embedding, kept
      so train and every query batch share the SAME one (required for correctness
      since ``subortho``/``learned`` embeddings are redrawn per forward).
    - ``norm_stats``: the ``NormalizationEncoder`` train stats (lower/upper/mean/
      std) to re-apply to query rows via the frozen path.
    - ``nan_mean``: the ``NanEncoder`` per-column context mean it imputes NaN/Inf
      with, likewise re-applied to query rows via its frozen path.
    - ``valid_feature_num``: the context-only count used by
      ``ValidFeatureEncoder`` to keep feature-group scaling independent of the
      query rows presented in a particular call.
    - ``query_y_embedding``: the context-derived embedding of one NaN query
      target. Streaming expands this one row per query chunk instead of encoding
      ``N_context + N_query`` targets during decode.
    - ``y_train``: context targets, to reconstruct the query y-placeholder embedding
      exactly as the transductive path does (query rows are NaN-masked anyway).
    - ``eval_pos``: number of context rows (``n_train``).

    ``norm_stats``, ``nan_mean`` and ``valid_feature_num`` cover every
    ``eval_pos``-dependent stage of ``x_preprocess`` (``NanEncoder`` ->
    ``NormalizationEncoder`` -> ``ValidFeatureEncoder``). All three must be carried,
    or query rows get stats derived from THEMSELVES instead of the context: a silent,
    data-dependent divergence rather than an error. With them, the cached
    path stays within float-reassociation distance of the transductive forward (~3e-05
    on nori-6m) for finite, NaN- and Inf-bearing tables alike. The legacy cached pair
    is bit-identical to ``forward_cached_regression``, which is literally that
    build/apply pair. Streamed BF16 preserves the same rows and statistics, but its
    chunked GEMMs and FP32 online-softmax reduction are only numerically close, not
    bit-exact.
    """

    caches: list
    feature_pos_emb: torch.Tensor | None
    norm_stats: dict | None
    y_train: torch.Tensor
    eval_pos: int
    nan_mean: torch.Tensor | None = None
    valid_feature_num: torch.Tensor | None = None
    query_y_embedding: torch.Tensor | None = None


from synthefy_nori.model.encoders import get_x_encoder, get_reg_y_encoder, preprocesss_4_x
from torch.amp import autocast


class FeaturesTransformer(nn.Module):
    def __init__(
        self,
        *,
        preprocess_config_x: dict[str, Any],
        encoder_config_x: dict[str, Any],
        encoder_config_y: dict[str, Any],
        decoder_config: dict[str, Any],
        nlayers: int,
        nhead: int,
        embed_dim: int,
        hid_dim: int,
        feature_positional_embedding_type: Literal["none", "subortho", "learned"] = "subortho",
        feature_positional_embedding_num_slots: int = 1000,
        mask_prediction: bool = False,
        features_per_group: int = 2,
        dropout: float = 0,
        pre_norm: bool = False,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        recompute_attn: bool = False,
        mlp_use_residual: bool = False,
        layer_arch: str = "fmfmsm",
        norm_type: str = "layernorm",
        deepnorm_alpha: float | None = None,
        use_target_aware_embedding: bool = False,
        use_column_specific_y_aware: bool = False,
        use_logn_attention: bool = False,
        use_learnable_attn_temperature: bool = False,
        attn_n_ref: float = 1024.0,
        omit_feature_decoder: bool = False,
        **layer_kwargs: Any,
    ):
        super().__init__()
        if mlp_use_residual:
            raise ValueError(
                "mlp_use_residual=True is unsupported: every MLP sublayer already "
                "uses the transformer's outer residual connection. This legacy "
                "flag was a no-op; leave it false or remove it from the config."
            )

        self.preprocess_config_x = preprocess_config_x
        self.encoder_config_x = encoder_config_x
        self.encoder_config_y = encoder_config_y
        self.decoder_config = decoder_config
        self.feature_positional_embedding_type = feature_positional_embedding_type
        self.nlayers = nlayers
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.mask_prediction = mask_prediction
        self.features_per_group = int(features_per_group)
        if self.features_per_group < 1:
            raise ValueError("features_per_group must be at least 1")
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.device = device
        self.dtype = dtype
        self.recompute_attn = recompute_attn
        self.mlp_use_residual = False
        self.layer_arch = layer_arch
        self.norm_type = norm_type
        self.deepnorm_alpha = deepnorm_alpha
        self.omit_feature_decoder = bool(omit_feature_decoder)
        self.num_reg_quantiles = int(decoder_config.get("num_reg_quantiles", 1))
        # Bar-distribution metadata (persisted via decoder_config at training).
        # When regression_loss == 'bar_distribution', the K reg_y_decoder outputs
        # are interpreted as K bin logits over [bar_borders_low, bar_borders_high]
        # at inference time.
        configured_regression_loss = decoder_config.get("regression_loss")
        if configured_regression_loss is None:
            # Legacy configs persisted only the decoder width. Historically a
            # scalar head was the MSE default and wider heads were pinball.
            # A one-level pinball head is inherently indistinguishable here;
            # new checkpoints persist regression_loss and avoid that ambiguity.
            configured_regression_loss = "mse" if self.num_reg_quantiles == 1 else "pinball"
        self.regression_loss = str(configured_regression_loss)
        configured_quantiles = decoder_config.get("regression_quantiles")
        if self.regression_loss == "pinball":
            if configured_quantiles is None:
                # Legacy checkpoints recorded only the decoder width. Those
                # runs used the historical evenly-spaced grid, so reconstruct
                # that grid exactly as the old inference path did.
                self.regression_quantiles = tuple(
                    (index + 1.0) / (self.num_reg_quantiles + 1.0) for index in range(self.num_reg_quantiles)
                )
            else:
                self.regression_quantiles = tuple(float(q) for q in configured_quantiles)
                if len(self.regression_quantiles) != self.num_reg_quantiles:
                    raise ValueError(
                        "decoder_config.regression_quantiles length "
                        f"{len(self.regression_quantiles)} does not match "
                        f"num_reg_quantiles={self.num_reg_quantiles}"
                    )
                if any(not math.isfinite(q) or q <= 0.0 or q >= 1.0 for q in self.regression_quantiles) or any(
                    left >= right
                    for left, right in zip(
                        self.regression_quantiles,
                        self.regression_quantiles[1:],
                    )
                ):
                    raise ValueError("decoder_config.regression_quantiles must be strictly increasing values in (0, 1)")
        else:
            self.regression_quantiles = ()
        self.num_bars = int(decoder_config.get("num_bars", self.num_reg_quantiles))
        self.bar_borders_low = float(decoder_config.get("bar_borders_low", -10.0))
        self.bar_borders_high = float(decoder_config.get("bar_borders_high", 10.0))
        # Bar-borders mode: 'uniform' (linspace over [low, high]) or
        # 'normal_quantile' (N(0,1) quantile-spaced for equal mass per bin
        # under standard normal). Normal-quantile concentrates ~3-4× more
        # bins near y=0 where context-normalized targets actually live;
        # eliminates wasted tail bins. Borders are persisted as a buffer so
        # inference uses the exact same edges.
        self.bar_borders_mode = str(decoder_config.get("bar_borders_mode", "uniform"))
        if self.regression_loss == "bar_distribution":
            if self.bar_borders_mode == "normal_quantile":
                # N(0,1) quantile spacing — bins are equal-mass under standard
                # normal. Replace ±inf at the edges with finite extreme values
                # (±8 std covers all real ctx-normalized data).
                try:
                    from scipy.stats import norm as _norm

                    qs = torch.linspace(0.0, 1.0, self.num_bars + 1, dtype=torch.float32)
                    edges_np = _norm.ppf(qs.numpy())
                    edges_np[0] = -8.0
                    edges_np[-1] = 8.0
                    bar_borders = torch.from_numpy(edges_np).float()
                except ImportError:
                    # Fallback: uniform if scipy isn't available.
                    bar_borders = torch.linspace(
                        self.bar_borders_low,
                        self.bar_borders_high,
                        self.num_bars + 1,
                        dtype=torch.float32,
                    )
            else:
                bar_borders = torch.linspace(
                    self.bar_borders_low,
                    self.bar_borders_high,
                    self.num_bars + 1,
                    dtype=torch.float32,
                )
            # register_buffer makes it part of state_dict and follows .to()
            self.register_buffer("bar_borders_buffer", bar_borders, persistent=True)
        else:
            self.bar_borders_buffer = None
        self.use_target_aware_embedding = use_target_aware_embedding
        # Column-specific y-aware modulation: extends row-level target_aware
        # embedding so the y-derived bias is *gated per column* by the inner
        # product between y_emb and the column's embedding. Columns whose
        # representations align with y direction get more y-aware bias;
        # orthogonal columns get less. Targets QSAR-style "find the relevant
        # bits among 1024 noisy fingerprints" failure mode.
        # Mixed with the original row-broadcast bias via a learned alpha
        # (sigmoid). alpha is initialized very negative (-5.0) so sigmoid(α)
        # ≈ 0.007 — at step 0 of FT, the new path contributes ~0.7%, so V8old
        # behavior is preserved exactly. Training can grow alpha if the
        # column-specific gate is useful.
        self.use_column_specific_y_aware = use_column_specific_y_aware
        if self.use_column_specific_y_aware:
            self.column_y_aware_alpha = nn.Parameter(
                torch.tensor(-5.0, device=device, dtype=dtype if dtype else torch.float32)
            )
        else:
            self.column_y_aware_alpha = None
        self.target_aware_scale = 1.0

        # logN attention scaling + learnable per-layer temperature.
        # Plumbed through EncoderBaseLayer to all per-layer MultiheadAttentions.
        self.use_logn_attention = use_logn_attention
        self.use_learnable_attn_temperature = use_learnable_attn_temperature
        self.attn_n_ref = float(attn_n_ref)

        layer_creator = lambda: EncoderBaseLayer(
            embed_dim=self.embed_dim,
            hid_dim=self.hid_dim,
            nhead=self.nhead,
            dropout=self.dropout,
            pre_norm=self.pre_norm,
            activation=self.activation,  # type: ignore
            layer_norm_eps=self.layer_norm_eps,
            device=self.device,
            dtype=self.dtype,
            recompute_attn=self.recompute_attn,
            mlp_use_residual=self.mlp_use_residual,
            layer_arch=self.layer_arch,  # type: ignore
            norm_type=self.norm_type,
            deepnorm_alpha=self.deepnorm_alpha,
            use_logn_attention=self.use_logn_attention,
            use_learnable_attn_temperature=self.use_learnable_attn_temperature,
            attn_n_ref=self.attn_n_ref,
            **layer_kwargs,
        )

        self.encoder_x = get_x_encoder(**encoder_config_x)
        # Explicit keys, not **encoder_config_y: pre-Tier-6 checkpoints still
        # carry a `max_num_classes` entry from the deleted classification
        # encoder, and it must not reach this constructor.
        self.reg_y_encoder = get_reg_y_encoder(
            num_inputs=encoder_config_y["num_inputs"],
            embedding_size=encoder_config_y["embedding_size"],
            nan_handling_y_encoder=encoder_config_y["nan_handling_y_encoder"],
        )
        if self.use_target_aware_embedding:
            self.reg_target_aware_embedding = nn.Linear(
                1,
                self.embed_dim,
                device=self.device,
                dtype=self.dtype,
            )
            nn.init.normal_(self.reg_target_aware_embedding.weight, std=0.02)
            nn.init.zeros_(self.reg_target_aware_embedding.bias)
        else:
            self.reg_target_aware_embedding = None

        self.transformer_encoder = LayerStack([layer_creator() for _ in range(self.nlayers)])
        if pre_norm:
            if norm_type == "rmsnorm":
                self.encoder_out_norm = RMSNorm(self.embed_dim, eps=1e-5, elementwise_affine=False)
            else:
                self.encoder_out_norm = nn.LayerNorm(self.embed_dim, eps=1e-5, elementwise_affine=False)
        else:
            self.encoder_out_norm = nn.Identity()

        self.reg_y_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.hid_dim),
            nn.LayerNorm(self.hid_dim),
            nn.GELU(),
            nn.Linear(self.hid_dim, self.num_reg_quantiles),
        )
        self.feature_decoder = (
            None
            if self.omit_feature_decoder
            else nn.Sequential(
                nn.Linear(self.embed_dim, self.hid_dim),
                nn.LayerNorm(self.hid_dim),
                nn.GELU(),
                nn.Linear(self.hid_dim, self.features_per_group),
            )
        )

        if feature_positional_embedding_type == "learned":
            self.feature_positional_embedding = nn.Embedding(feature_positional_embedding_num_slots, self.embed_dim)
            nn.init.normal_(self.feature_positional_embedding.weight, std=0.02)
            self.feature_positional_embedding_num_slots = feature_positional_embedding_num_slots
        elif feature_positional_embedding_type == "subortho":
            self.feature_positional_embedding = nn.Linear(self.embed_dim // 4, self.embed_dim)

        self.x_preprocess = preprocesss_4_x(**preprocess_config_x)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        eval_pos: int,
        y_type: torch.Tensor = None,
        calculate_sample_attention: bool = False,
        calculate_feature_attention: bool = False,
        return_embeddings: bool = False,
        **kwargs,
    ) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        x: The input x, which includes both train x and test x, Shape: [batch, sequence, feature]
        y: The input y, which includes both train y and test y, Shape: [batch, label]
        eval_pos: Train x and train y split point
        return_embeddings: when True, skip the decoder head and return the
            per-row target-token representation from the final encoder layer
            (post encoder_out_norm), shape [batch, seq, embed_dim]. Callers
            slice context (``:eval_pos``) vs query (``eval_pos:``) themselves.
        """
        assert x is not None and y is not None, "x and y must not be none"
        assert eval_pos > 0, "eval_pos must be a positive number"
        assert len(x.shape) == 3, "x must be [Batch, seq, Feature]"
        assert len(y.shape) == 2, "y must be [Batch, label]"
        assert eval_pos < x.shape[1] and eval_pos <= y.shape[1], (
            "The split point between train x and test x must be less than the feature dimension of x, and less than or equal to the label dimension of y"
        )

        _, seq_len, _ = x.shape
        x, feature_to_add = self._build_x_preprocess_inputs(x, eval_pos)
        y = {"data": y}
        preprocessed_x = self.x_preprocess(x)
        preprocessed_x = self.process_4_x(preprocessed_x)
        x_encoder_result = self.encoder_x(preprocessed_x)
        x_emb_result = x_encoder_result["data"]

        for k in y:
            # Extend the label dimension of y when it is insufficient
            y[k] = y[k].unsqueeze(-1)
            if y[k].shape[1] < seq_len:
                y[k] = torch.cat(
                    (
                        y[k],
                        torch.nan
                        * torch.zeros(
                            y[k].shape[0],
                            seq_len - y[k].shape[1],
                            y[k].shape[2],
                            device=y[k].device,
                            dtype=y[k].dtype,
                        ),
                    ),
                    dim=1,
                )
        target_aware_y = y["data"].squeeze(-1).clone()
        # Mask the test y — functional (no in-place mutation of the input dict)
        seq_positions = torch.arange(y["data"].shape[1], device=y["data"].device)
        y_data_masked = torch.where(
            seq_positions.view(1, -1, 1) >= eval_pos,
            torch.tensor(float("nan"), device=y["data"].device, dtype=y["data"].dtype),
            y["data"],
        )

        # Embed y — direct encoder call (avoids mixed_y_embedding graph break).
        # Encoder output may be 4-D [B, S, 1, E] because y has a trailing
        # dim of 1; squeeze it to [B, S, E] to match the old contract.
        y_enc_input = {"data": y_data_masked, "eval_pos": eval_pos}
        embedded_y = self.reg_y_encoder(y_enc_input)["data"].squeeze(2)

        embedded_x = self.add_embeddings(x_emb_result)
        embedded_x = self.apply_target_aware_embedding(
            embedded_x,
            target_aware_y,
            eval_pos=eval_pos,
        )
        embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)
        if calculate_sample_attention or calculate_feature_attention:
            return self.transformer_encoder(
                embedded_all,
                feature_atten_mask=None,
                eval_pos=eval_pos,
                calculate_sample_attention=calculate_sample_attention,
                calculate_feature_attention=calculate_feature_attention,
                **kwargs,
            )
        else:
            pass
        encoder_out = self.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos, **kwargs)[0]
        encoder_out = self.encoder_out_norm(encoder_out)

        if return_embeddings:
            # Per-row target-token representation at the final encoder layer.
            # The target token is the last feature-group slot (index -1); this
            # is the same representation the regression decoder consumes.
            # [batch, seq, embed_dim] — context rows are [:, :eval_pos]
            # and query rows are [:, eval_pos:].
            return encoder_out[:, :, -1]

        test_encoder_out = encoder_out[:, eval_pos:, -1]
        encoder_out_4_feature = encoder_out[:, :, :-1, :]
        if self.mask_prediction:
            # Direct decoder call (avoids y_decoder graph break)
            reg_output = self.reg_y_decoder(test_encoder_out)
            feature_pred = (
                None
                if (self.feature_decoder is None or getattr(self, "_skip_feature_decoder", False))
                else self.feature_decoder(encoder_out_4_feature)
            )
            output_decoded = {
                "reg_output": reg_output,
                "feature_pred": feature_pred,
                # Query target-token representation, exposed (with gradient) only
                # when the embedding-geometry regularizer is enabled, so the
                # trainer can apply an orthogonality / variance-floor penalty.
                **(
                    {"_aux_query_embedding": test_encoder_out}
                    if getattr(self, "_capture_query_embedding", False)
                    else {}
                ),
                "process_config": {
                    "n_x_padding": feature_to_add,
                    "features_per_group": self.features_per_group,
                    "num_used_features": preprocessed_x.get("_valid_feature_num"),
                    "mean_for_normalization": preprocessed_x.get("_norm_mean"),
                    "std_for_normalization": preprocessed_x.get("_norm_std"),
                },
            }
        else:
            output_decoded = self.reg_y_decoder(test_encoder_out)

        return output_decoded

    def _build_x_preprocess_inputs(
        self,
        x: torch.Tensor,
        eval_pos: int,
    ) -> tuple[dict[str, torch.Tensor | int], int]:
        batch_size, seq_len, num_feature = x.shape
        x_dict: dict[str, torch.Tensor | int] = {
            "data": x,
            # The numeric channel treats NaN and both infinities as missing.
            # NanEncoder may retain their kind in a separate indicator channel,
            # but process_4_x must never let an infinity reach the numeric encoder.
            "mask": (~torch.isfinite(x)).to(torch.int32).to(x.device),
        }
        feature_to_add = (-num_feature) % self.features_per_group
        if feature_to_add > 0:
            for k in ("data", "mask"):
                value = x_dict[k]
                assert isinstance(value, torch.Tensor)
                x_dict[k] = torch.cat(
                    (
                        value,
                        torch.zeros(
                            batch_size,
                            seq_len,
                            feature_to_add,
                            device=value.device,
                            dtype=value.dtype,
                        ),
                    ),
                    dim=-1,
                )
        for k in ("data", "mask"):
            value = x_dict[k]
            assert isinstance(value, torch.Tensor)
            x_dict[k] = value.reshape(
                batch_size,
                seq_len,
                value.shape[2] // self.features_per_group,
                self.features_per_group,
            )
        x_dict["eval_pos"] = eval_pos
        return x_dict, feature_to_add

    def _slice_preprocessed_x(
        self,
        preprocessed_x: dict[str, torch.Tensor | int],
        row_slice: slice,
        total_rows: int,
    ) -> dict[str, torch.Tensor | int]:
        sliced: dict[str, torch.Tensor | int] = {}
        for k, v in preprocessed_x.items():
            if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == total_rows:
                sliced[k] = v[:, row_slice].contiguous()
            else:
                sliced[k] = v
        return sliced

    def _choose_feature_slots(self, n_groups: int, device: torch.device) -> torch.Tensor:
        """Pick ``n_groups`` learned-positional slot indices.

        Distinct slots (a random permutation) in the intended regime where the
        table is large enough (``n_groups <= num_slots``), so every feature group
        gets a unique positional code. When there are MORE feature groups than
        slots, fall back to sampling WITH replacement so wide tables degrade
        gracefully — some columns share a code, like subortho past its dimension
        ceiling — instead of crashing on a ``randperm[:n_groups]`` shape mismatch.
        """
        n_slots = self.feature_positional_embedding_num_slots
        if n_groups <= n_slots:
            return torch.randperm(n_slots, device=device)[:n_groups]
        return torch.randint(n_slots, (n_groups,), device=device)

    def make_feature_positional_embeddings(
        self,
        n_groups: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.feature_positional_embedding_type == "subortho":
            with autocast(device_type=device.type, enabled=False):
                embs = torch.randn(
                    (n_groups, self.embed_dim // 4),
                    device=device,
                    dtype=torch.float32,
                )
                torch.nn.init.orthogonal_(embs)
            return self.feature_positional_embedding(embs.to(dtype))
        if self.feature_positional_embedding_type == "learned":
            slot_indices = self._choose_feature_slots(n_groups, device)
            return self.feature_positional_embedding(slot_indices).to(dtype)
        if self.feature_positional_embedding_type is None or self.feature_positional_embedding_type == "none":
            return None
        raise ValueError(f"Unknown feature_positional_embedding_type={self.feature_positional_embedding_type}")

    def apply_feature_positional_embeddings(
        self,
        x: torch.Tensor,
        embs: torch.Tensor | None,
    ) -> torch.Tensor:
        if embs is None:
            return x
        return x + embs[None, None, :, :]

    def _encode_x_rows(
        self,
        preprocessed_x: dict[str, torch.Tensor | int],
        row_slice: slice,
        *,
        total_rows: int,
        feature_pos_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        x_part = self._slice_preprocessed_x(preprocessed_x, row_slice, total_rows)
        encoded = self.encoder_x(x_part)["data"]
        return self.apply_feature_positional_embeddings(encoded, feature_pos_emb)

    def _encode_y_full(
        self,
        y: torch.Tensor,
        *,
        total_rows: int,
        eval_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_dict: dict[str, torch.Tensor] = {"data": y}
        for k in y_dict:
            y_dict[k] = y_dict[k].unsqueeze(-1)
            if y_dict[k].shape[1] < total_rows:
                y_dict[k] = torch.cat(
                    (
                        y_dict[k],
                        torch.full(
                            (
                                y_dict[k].shape[0],
                                total_rows - y_dict[k].shape[1],
                                y_dict[k].shape[2],
                            ),
                            float("nan"),
                            device=y_dict[k].device,
                            dtype=y_dict[k].dtype,
                        ),
                    ),
                    dim=1,
                )
        target_aware_y = y_dict["data"].squeeze(-1).clone()
        seq_positions = torch.arange(total_rows, device=y_dict["data"].device)
        y_data_masked = torch.where(
            seq_positions.view(1, -1, 1) >= eval_pos,
            torch.tensor(float("nan"), device=y_dict["data"].device, dtype=y_dict["data"].dtype),
            y_dict["data"],
        )
        y_enc_input = {"data": y_data_masked, "eval_pos": eval_pos}
        embedded_y = self.reg_y_encoder(y_enc_input)["data"].squeeze(2)
        return target_aware_y, embedded_y

    def forward_cached_regression(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        eval_pos: int,
        *,
        row_chunk_size: int | None = None,
        cache_dtype: str = "bf16",
        offload_kv_cache: bool = False,
        fit_row_chunk: int | None = None,
        adaptive_query_chunk: bool = True,
        quantize_kv_cache: bool | None = None,
        adaptive_oom_chunk: bool | None = None,
    ) -> torch.Tensor:
        """Regression-only cached prediction path for chunked inference.

        The train-side transformer states and projected sequence-attention K/Vs
        are computed once. Test rows are then streamed through the same layers
        using those train caches. Preprocessing still sees the full transductive
        X table once, preserving the existing inference semantics.

        Args:
            x: inputs, ``[batch, seq, feature]``.
            y: targets, ``[batch, label]``.
            eval_pos: index separating context rows from query rows.
            row_chunk_size: query rows per decode forward.
            cache_dtype: precision the stored K/V cache is kept at — ``"bf16"``
                (bit-exact, the default) or ``"int8"``. This method takes a
                CONCRETE precision: it does not know the device budget, so it
                cannot decide when quantizing is worth it. That decision belongs
                to :class:`~synthefy_nori.inference.memory_policy.MemoryPolicy`,
                which ``NoriPredictor.predict`` resolves before calling here.
            offload_kv_cache: keep the cache in host RAM, streaming slices back.
            fit_row_chunk: bound the fit-time build working set to this many
                context rows (deterministic, with small floating-point reassociation differences). ``None`` = off.
            adaptive_query_chunk: on a decode OOM, halve the query chunk and
                retry rather than raising.
            quantize_kv_cache: alias for ``cache_dtype``, accepted so the WS1
                benchmark harnesses run unmodified
                against both this branch and #257 — which is what makes the
                before/after memory comparison apples-to-apples. Prefer
                ``cache_dtype`` in new code.
            adaptive_oom_chunk: alias for ``adaptive_query_chunk``, same reason.

        Returns:
            Predictions for the query rows.
        """
        # Aliases: honour them, but let the new names win when both are given, so a
        # caller migrating one call site cannot end up with the old value silently
        # applied.
        if quantize_kv_cache is not None and cache_dtype == "bf16":
            cache_dtype = "int8" if quantize_kv_cache else "bf16"
        if adaptive_oom_chunk is not None:
            adaptive_query_chunk = bool(adaptive_oom_chunk)
        if cache_dtype not in ("bf16", "int8"):
            raise ValueError(
                "forward_cached_regression takes a concrete cache_dtype of "
                f"'bf16' or 'int8', got {cache_dtype!r}. Deciding which one is "
                "worth it needs the device budget, so it belongs to "
                "MemoryPolicy.resolve() (synthefy_nori.inference.memory_policy) — "
                "go through NoriPredictor(memory_policy=...) rather than guessing here."
            )
        if self.mask_prediction:
            raise NotImplementedError("forward_cached_regression requires mask_prediction=False")
        if x is None or y is None:
            raise AssertionError("x and y must not be none")
        if eval_pos <= 0:
            raise AssertionError("eval_pos must be a positive number")
        if len(x.shape) != 3:
            raise AssertionError("x must be [Batch, seq, Feature]")
        if len(y.shape) != 2:
            raise AssertionError("y must be [Batch, label]")
        if eval_pos >= x.shape[1] or eval_pos > y.shape[1]:
            raise AssertionError("Invalid eval_pos for cached regression")

        # Thin wrapper over the reusable context-cache split: encode the context
        # (train) rows once, then apply that bundle to the query rows. Behaviour is
        # bit-identical to the previous inline implementation; the split exists so a
        # caller serving many query batches against a FIXED context can build the
        # bundle once (build_context_cache) and reuse it across calls
        # (apply_context_cache) instead of rebuilding the O(N_train) cache each time.
        bundle = self.build_context_cache(
            x[:, :eval_pos],
            y[:, :eval_pos],
            cache_dtype=cache_dtype,
            offload_kv_cache=offload_kv_cache,
            fit_row_chunk=fit_row_chunk,
        )
        return self.apply_context_cache(
            x[:, eval_pos:],
            bundle,
            row_chunk_size=row_chunk_size,
            adaptive_query_chunk=adaptive_query_chunk,
        )

    def build_context_cache(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        cache_dtype: str = "bf16",
        offload_kv_cache: bool = False,
        fit_row_chunk: int | None = None,
        quantize_kv_cache: bool | None = None,
        stream_context: bool = False,
        _hybrid_resident_int8_prefill: bool = False,
    ) -> ContextCache:
        """Encode a fixed context (train) table once into a reusable ``ContextCache``.

        The expensive O(N_train) work -- preprocessing, feature/target encoding, and
        the per-layer row-attention K/V build -- happens HERE, once. The returned
        bundle carries everything :meth:`apply_context_cache` needs to score query
        rows without touching the context again, so serving many query batches
        against the same context pays this cost a single time.

        ``cache_dtype`` / ``offload_kv_cache`` / ``fit_row_chunk`` behave exactly as
        in :meth:`forward_cached_regression` (which is now a wrapper over this pair).
        ``stream_context=True`` additionally requires CPU inputs, host offload, and a
        concrete row cap; it never materializes a full-N activation or K/V layer on
        the compute device.
        """
        if quantize_kv_cache is not None and cache_dtype == "bf16":
            cache_dtype = "int8" if quantize_kv_cache else "bf16"
        if cache_dtype not in ("bf16", "int8"):
            raise ValueError(
                f"build_context_cache takes a concrete cache_dtype of 'bf16' or 'int8', got {cache_dtype!r}."
            )
        if self.mask_prediction:
            raise NotImplementedError("build_context_cache requires mask_prediction=False")
        if not bool(self.preprocess_config_x.get("normalize_on_train_only", True)) and (
            bool(self.preprocess_config_x.get("normalize_x", False))
            or bool(self.preprocess_config_x.get("remove_outliers", False))
        ):
            raise NotImplementedError(
                "build_context_cache cannot preserve normalize_on_train_only=False: "
                "that transductive configuration derives preprocessing statistics "
                "from context and query rows together; use the uncached forward path"
            )
        if x_train is None or y_train is None:
            raise AssertionError("x_train and y_train must not be None")
        if len(x_train.shape) != 3:
            raise AssertionError("x_train must be [Batch, seq, Feature]")
        if len(y_train.shape) != 2:
            raise AssertionError("y_train must be [Batch, label]")
        eval_pos = x_train.shape[1]
        if eval_pos <= 0:
            raise AssertionError("x_train must have at least one context row")
        if y_train.shape[1] < eval_pos:
            raise AssertionError("y_train must cover all context rows")
        if _hybrid_resident_int8_prefill and (
            stream_context or offload_kv_cache or cache_dtype != "int8" or fit_row_chunk is None
        ):
            raise ValueError(
                "hybrid resident INT8 prefill requires cache_dtype='int8', "
                "offload_kv_cache=False, stream_context=False, and a concrete "
                "context_row_chunk"
            )
        if stream_context or _hybrid_resident_int8_prefill:
            if stream_context and not offload_kv_cache:
                raise ValueError("stream_context requires offload_kv_cache=True")
            if fit_row_chunk is None:
                requested = "stream_context" if stream_context else "hybrid resident INT8 prefill"
                raise ValueError(f"{requested} requires a concrete context_row_chunk")
            return self._build_context_cache_streaming(
                x_train,
                y_train[:, :eval_pos],
                cache_dtype=cache_dtype,
                fit_row_chunk=fit_row_chunk,
                _hybrid_resident_int8_prefill=_hybrid_resident_int8_prefill,
            )

        x_dict, _feature_to_add = self._build_x_preprocess_inputs(x_train, eval_pos)
        preprocessed_x = self.x_preprocess(x_dict)
        # Capture the train-derived normalization stats BEFORE process_4_x so they
        # can be re-applied (frozen) to query rows in apply_context_cache. Empty ->
        # None (no normalization is active, so query rows need no train stats).
        captured = preprocessed_x.get("_norm_stats") or {}
        norm_stats = {k: v.detach() for k, v in captured.items()} if captured else None
        # Same for NanEncoder's imputation fill (the train column mean). The key is
        # absent when nan_handling_enabled=False.
        captured_nan_mean = preprocessed_x.get("_nan_mean")
        nan_mean = captured_nan_mean.detach() if isinstance(captured_nan_mean, torch.Tensor) else None
        captured_valid_feature_num = preprocessed_x.get("_valid_feature_num")
        valid_feature_num = (
            captured_valid_feature_num.detach() if isinstance(captured_valid_feature_num, torch.Tensor) else None
        )
        preprocessed_x = self.process_4_x(preprocessed_x)
        data_tensor = preprocessed_x["data"]
        assert isinstance(data_tensor, torch.Tensor)
        feature_pos_emb = self.make_feature_positional_embeddings(
            data_tensor.shape[2],
            device=data_tensor.device,
            dtype=data_tensor.dtype,
        )
        target_aware_y, embedded_y = self._encode_y_full(
            y_train[:, :eval_pos],
            total_rows=eval_pos,
            eval_pos=eval_pos,
        )
        x_train_enc = self._encode_x_rows(
            preprocessed_x,
            slice(0, eval_pos),
            total_rows=eval_pos,
            feature_pos_emb=feature_pos_emb,
        )
        x_train_enc = self.apply_target_aware_embedding(
            x_train_enc,
            target_aware_y[:, :eval_pos],
            eval_pos=eval_pos,
        )
        train_tokens = torch.cat((x_train_enc, embedded_y[:, :eval_pos].unsqueeze(2)), dim=2)
        # WS1 Stage 2 memory lever: int8-quantize and/or host-offload the
        # O(N_train) seq K/V cache. Threaded INTO build_train_cache so each layer's
        # cache is scaled the moment it's produced -- otherwise all L layers'
        # full-precision caches accumulate on the GPU and the prefill peak is
        # O(L*N) regardless of any post-hoc offload.
        #
        # Arguments only: this path reads no environment variables. Callers configure
        # it through NoriPredictor(memory_policy=MemoryPolicy(...)), which resolves the rung
        # and passes the concrete decision down.
        quantize_kv_cache = cache_dtype == "int8"
        _, caches = self.transformer_encoder.build_train_cache(
            train_tokens,
            feature_atten_mask=None,
            quantize_kv_cache=quantize_kv_cache,
            offload_kv_cache=offload_kv_cache,
            fit_row_chunk=fit_row_chunk,
            device=data_tensor.device,
        )
        return ContextCache(
            caches=caches,
            feature_pos_emb=feature_pos_emb,
            norm_stats=norm_stats,
            y_train=y_train[:, :eval_pos].detach(),
            eval_pos=eval_pos,
            nan_mean=nan_mean,
            valid_feature_num=valid_feature_num,
        )

    def apply_context_cache(
        self,
        x_test: torch.Tensor,
        context: ContextCache,
        *,
        row_chunk_size: int | None = None,
        adaptive_query_chunk: bool = True,
        query_chunk_attempt_callback: (Callable[[int, Literal["success", "oom"]], None] | None) = None,
    ) -> torch.Tensor:
        """Score query rows against a prebuilt :class:`ContextCache`.

        Bit-identical to :meth:`forward_cached_regression` on the same (context,
        query) pair, but the context forward is skipped: only the query rows stream
        through the cached per-layer K/V. Safe to call repeatedly with different
        query batches for the same context -- it mutates nothing on ``context``.

        ``query_chunk_attempt_callback`` is an internal observability hook. It is
        called for every decode OOM with the chunk limit that failed, then once
        with the effective chunk limit after the full query batch succeeds. If
        retries are exhausted, the final event is the OOM at chunk size one.
        """
        if self.mask_prediction:
            raise NotImplementedError("apply_context_cache requires mask_prediction=False")
        if len(x_test.shape) != 3:
            raise AssertionError("x_test must be [Batch, seq, Feature]")
        eval_pos = context.eval_pos
        n_test = x_test.shape[1]
        if n_test <= 0:
            raise AssertionError("x_test must have at least one query row")

        # Preprocess query rows with the FROZEN train stats -- bit-identical to the
        # transductive path, which normalizes test rows with train (context) stats.
        # eval_pos here only feeds the (overridden) NormalizationEncoder split point.
        x_dict, _feature_to_add = self._build_x_preprocess_inputs(x_test, n_test)
        if context.norm_stats is not None:
            x_dict["_frozen_norm_stats"] = {key: value.to(x_test.device) for key, value in context.norm_stats.items()}
        if context.nan_mean is not None:
            x_dict["_frozen_nan_mean"] = context.nan_mean.to(x_test.device)
        if context.valid_feature_num is not None:
            x_dict["_frozen_valid_feature_num"] = context.valid_feature_num.to(x_test.device)
        preprocessed_x = self.x_preprocess(x_dict)
        preprocessed_x = self.process_4_x(preprocessed_x)

        streaming_context = context.query_y_embedding is not None
        if not streaming_context:
            # Legacy bundles retain context y. Reconstruct the same query embedding
            # by encoding [context y ; NaN query] and slicing the tail.
            _, embedded_y_full = self._encode_y_full(
                context.y_train,
                total_rows=eval_pos + n_test,
                eval_pos=eval_pos,
            )
            embedded_y_query = embedded_y_full[:, eval_pos:]

        if row_chunk_size is None or row_chunk_size <= 0:
            row_chunk_size = n_test

        def _run_chunk(start: int, end: int) -> torch.Tensor:
            if streaming_context:
                # Keep the raw query matrix and its parameter-free preprocessing
                # on the host. Only this query slice and its one-row y placeholder
                # are resident on the compute device.
                width = end - start
                host_chunk = self._slice_preprocessed_x(preprocessed_x, slice(start, end), n_test)
                compute_device = next(self.parameters()).device
                staged = {
                    key: value.to(compute_device) if isinstance(value, torch.Tensor) else value
                    for key, value in host_chunk.items()
                }
                x_q = self._encode_x_rows(
                    staged,
                    slice(0, width),
                    total_rows=width,
                    feature_pos_emb=context.feature_pos_emb,
                )
                y_q = context.query_y_embedding.to(compute_device).expand(-1, width, -1)
            else:
                x_q = self._encode_x_rows(
                    preprocessed_x,
                    slice(start, end),
                    total_rows=n_test,
                    feature_pos_emb=context.feature_pos_emb,
                )
                y_q = embedded_y_query[:, start:end]
            test_tokens = torch.cat((x_q, y_q.unsqueeze(2)), dim=2)
            test_out = self.transformer_encoder.forward_test_with_cache(
                test_tokens,
                context.caches,
                feature_atten_mask=None,
            )
            test_out = self.encoder_out_norm(test_out)
            return self.reg_y_decoder(test_out[:, :, -1])

        outputs = []
        start = 0
        chunk = row_chunk_size
        while start < n_test:
            end = min(start + chunk, n_test)
            try:
                outputs.append(_run_chunk(start, end))
                start = end
            except torch.cuda.OutOfMemoryError:
                if query_chunk_attempt_callback is not None:
                    query_chunk_attempt_callback(chunk, "oom")
                # Adaptive halving: degrade the query chunk instead of dying. The
                # reduced chunk is deliberately NOT restored for later chunks --
                # whatever made this one not fit is a property of the request, so
                # restoring would re-OOM on the next chunk and thrash.
                if not adaptive_query_chunk or chunk <= 1:
                    raise
                torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
        if query_chunk_attempt_callback is not None:
            query_chunk_attempt_callback(chunk, "success")
        return torch.cat(outputs, dim=1)

    def apply_target_aware_embedding(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        eval_pos: int,
    ) -> torch.Tensor:
        scale = float(getattr(self, "target_aware_scale", 1.0))
        if not self.use_target_aware_embedding or eval_pos <= 0 or abs(scale) < 1e-8:
            return x

        # Compute bias for context rows, pad query rows with zeros (functional)
        assert self.reg_target_aware_embedding is not None
        y_ctx = y[:, :eval_pos].unsqueeze(-1).to(x.dtype)
        ctx_bias = self.reg_target_aware_embedding(y_ctx).to(x.dtype)

        n_query = x.shape[1] - eval_pos
        query_bias = ctx_bias.new_zeros(x.shape[0], n_query, self.embed_dim)
        target_bias = torch.cat([ctx_bias, query_bias], dim=1)
        if scale != 1.0:
            target_bias = target_bias * scale

        # Original row-broadcast bias (same y signal across all feature groups
        # per row). target_bias.unsqueeze(2): [B, seq, 1, embed_dim] →
        # broadcasts over the n_feature_groups dim of x.
        row_broadcast_bias = target_bias.unsqueeze(2)

        if self.use_column_specific_y_aware and self.column_y_aware_alpha is not None:
            # Column-specific gating: each feature group gets a y-bias scaled
            # by its alignment with y_emb. Columns "in the direction of" the
            # y signal get strong y-aware bias; orthogonal columns get less.
            # This is the missing piece vs TabICL's per-column y-injection.
            #
            # col_score[b, t, c] = <x[b, t, c, :], target_bias[b, t, :]>
            # Higher score → column emb is closer to y signal → higher gate.
            inv_sqrt_d = 1.0 / (self.embed_dim**0.5)
            col_score = (x * row_broadcast_bias).sum(dim=-1) * inv_sqrt_d  # [B, seq, n_cols]
            col_gate = torch.sigmoid(col_score).unsqueeze(-1)  # [B, seq, n_cols, 1]
            col_specific_bias = row_broadcast_bias * col_gate  # [B, seq, n_cols, embed_dim]
            # Mix old row-broadcast with new column-specific via learned α.
            # alpha=sigmoid(α_param) starts at sigmoid(-5)≈0.007 → ~99.3%
            # original V8old behavior preserved at FT step 0.
            alpha = torch.sigmoid(self.column_y_aware_alpha).to(x.dtype)
            return x + (1.0 - alpha) * row_broadcast_bias + alpha * col_specific_bias

        return x + row_broadcast_bias

    def process_4_x(self, data: dict):
        x_input = data["data"]
        mask = data["mask"].to(torch.bool)
        x_input = torch.where(mask, float("nan"), x_input)
        data["data"] = x_input
        return data

    def add_embeddings(self, x: torch.Tensor):
        if self.feature_positional_embedding_type == "subortho":
            with autocast(device_type=x.device.type, enabled=False):
                embs = torch.randn(
                    (x.shape[2], x.shape[3] // 4),
                    device=x.device,
                    dtype=torch.float32,
                )
                torch.nn.init.orthogonal_(embs)
            embs = self.feature_positional_embedding(embs.to(x.dtype))
            x += embs[None, None]
        elif self.feature_positional_embedding_type == "learned":
            # Learned positional embeddings (TabPFN-2.6 style).
            # n_slots slots in the Embedding table; each forward uses a random
            # permutation of slot indices to assign positions to feature groups.
            # This preserves permutation invariance: the model can't rely on
            # a fixed feature ↔ slot mapping across episodes. When there are more
            # feature groups than slots, _choose_feature_slots samples with
            # replacement so wide tables degrade gracefully instead of crashing.
            n_groups = x.shape[2]
            slot_indices = self._choose_feature_slots(n_groups, x.device)
            embs = self.feature_positional_embedding(slot_indices).to(x.dtype)
            x = x + embs[None, None, :, :]
        elif self.feature_positional_embedding_type is None or self.feature_positional_embedding_type == "none":
            embs = None
        else:
            raise ValueError(f"Unknown feature_positional_embedding_type={self.feature_positional_embedding_type}")
        return x

    def _build_context_cache_streaming(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        cache_dtype: str,
        fit_row_chunk: int,
        _hybrid_resident_int8_prefill: bool = False,
    ) -> ContextCache:
        """Build a context cache without a full-N tensor on the compute device."""
        if x_train.device.type != "cpu" or y_train.device.type != "cpu":
            mode = "hybrid resident INT8 prefill" if _hybrid_resident_int8_prefill else "stream_context"
            raise ValueError(
                f"{mode} requires CPU x_train/y_train; upload only row chunks, not the full training matrix"
            )
        compute_device = next(self.parameters()).device
        eval_pos = x_train.shape[1]

        # Preprocessing reductions are context-wide but parameter-free, so keep
        # their small raw-feature representation and derived stats on CPU.
        x_dict, _feature_to_add = self._build_x_preprocess_inputs(x_train, eval_pos)
        preprocessed_x = self.x_preprocess(x_dict)
        captured = preprocessed_x.get("_norm_stats") or {}
        norm_stats = {k: v.detach() for k, v in captured.items()} if captured else None
        captured_nan_mean = preprocessed_x.get("_nan_mean")
        nan_mean = captured_nan_mean.detach() if isinstance(captured_nan_mean, torch.Tensor) else None
        captured_valid_feature_num = preprocessed_x.get("_valid_feature_num")
        valid_feature_num = (
            captured_valid_feature_num.detach() if isinstance(captured_valid_feature_num, torch.Tensor) else None
        )
        n_groups = (x_train.shape[2] + self.features_per_group - 1) // self.features_per_group
        feature_pos_emb = self.make_feature_positional_embeddings(
            int(n_groups),
            device=compute_device,
            dtype=x_train.dtype,
        )
        # The full pass above exists only to derive context-wide frozen stats.
        # Re-run the parameter-free preprocessing per row chunk below so it does
        # not overlap a full preprocessed table with the O(N) token buffer.
        del x_dict, preprocessed_x, captured, captured_nan_mean
        del captured_valid_feature_num

        # Query y is always NaN-masked. Cache its one-row embedding, derived from
        # the full context mean, rather than recreating an N_context+N_query y
        # activation at decode time.
        y_values = y_train.unsqueeze(-1)
        y_finite = torch.isfinite(y_values)
        y_count = y_finite.sum(dim=1).clamp_min(1)
        y_nan_mean = torch.where(y_finite, y_values, torch.zeros_like(y_values)).sum(dim=1) / y_count
        y_nan_mean_device = y_nan_mean.to(compute_device)
        query_y_input = {
            "data": torch.full(
                (y_train.shape[0], 1, 1),
                float("nan"),
                device=compute_device,
                dtype=y_train.dtype,
            ),
            "eval_pos": 1,
            "_frozen_nan_mean": y_nan_mean_device,
        }
        query_y_embedding = self.reg_y_encoder(query_y_input)["data"].squeeze(2).detach()

        train_tokens = None
        for row_start in range(0, eval_pos, fit_row_chunk):
            row_end = min(row_start + fit_row_chunk, eval_pos)
            width = row_end - row_start
            chunk, _ = self._build_x_preprocess_inputs(x_train[:, row_start:row_end], width)
            if norm_stats is not None:
                chunk["_frozen_norm_stats"] = norm_stats
            if nan_mean is not None:
                chunk["_frozen_nan_mean"] = nan_mean
            if valid_feature_num is not None:
                chunk["_frozen_valid_feature_num"] = valid_feature_num
            chunk = self.x_preprocess(chunk)
            chunk = self.process_4_x(chunk)
            staged = {
                key: value.to(compute_device) if isinstance(value, torch.Tensor) else value
                for key, value in chunk.items()
            }
            x_encoded = self._encode_x_rows(
                staged,
                slice(0, width),
                total_rows=width,
                feature_pos_emb=feature_pos_emb,
            )
            y_rows = y_train[:, row_start:row_end].to(compute_device)
            x_encoded = self.apply_target_aware_embedding(x_encoded, y_rows, eval_pos=width)
            y_input = {
                "data": y_rows.unsqueeze(-1),
                "eval_pos": width,
                "_frozen_nan_mean": y_nan_mean_device,
            }
            y_encoded = self.reg_y_encoder(y_input)["data"].squeeze(2)
            tokens = torch.cat((x_encoded, y_encoded.unsqueeze(2)), dim=2)
            host_tokens = tokens.detach().to("cpu")
            if train_tokens is None:
                train_tokens = torch.empty(
                    (x_train.shape[0], eval_pos, *host_tokens.shape[2:]),
                    dtype=host_tokens.dtype,
                    device="cpu",
                )
            train_tokens[:, row_start:row_end].copy_(host_tokens)
            del chunk, staged, x_encoded, y_rows, y_encoded, tokens, host_tokens
        if train_tokens is None:
            raise ValueError("stream_context received an empty context")

        empty_y_train = y_train[:, :0].detach()
        del x_train, y_train, y_values, y_finite, y_count, y_nan_mean
        del y_nan_mean_device, query_y_input

        final_tokens, caches = self.transformer_encoder.build_train_cache(
            train_tokens,
            feature_atten_mask=None,
            quantize_kv_cache=cache_dtype == "int8",
            offload_kv_cache=not _hybrid_resident_int8_prefill,
            fit_row_chunk=fit_row_chunk,
            device=compute_device,
            stream_context=not _hybrid_resident_int8_prefill,
            _hybrid_resident_int8_prefill=_hybrid_resident_int8_prefill,
        )
        del train_tokens, final_tokens
        return ContextCache(
            caches=caches,
            feature_pos_emb=feature_pos_emb,
            norm_stats=norm_stats,
            y_train=empty_y_train,
            eval_pos=eval_pos,
            nan_mean=nan_mean,
            valid_feature_num=valid_feature_num,
            query_y_embedding=query_y_embedding,
        )
