from __future__ import annotations

import torch
import torch.nn as nn
from synthefy_nori.model.layer import EncoderBaseLayer, MLP, LayerStack, RMSNorm
from typing import Any, Literal
from synthefy_nori.model.encoders import get_x_encoder, get_cls_y_encoder, get_reg_y_encoder, preprocesss_4_x
from torch.amp import autocast


class FeaturesTransformer(nn.Module):
    def __init__(
                self,
                *,
                preprocess_config_x:dict[str, Any],
                encoder_config_x:dict[str, Any],
                encoder_config_y:dict[str, Any],
                decoder_config:dict[str, Any],
                nlayers:int,
                nhead: int, 
                embed_dim: int, 
                hid_dim:int,
                feature_positional_embedding_type:Literal['none','subortho','learned'] = 'subortho',
                feature_positional_embedding_num_slots:int = 1000,
                mask_prediction: bool = False,
                features_per_group:int = 2,
                dropout: float=0,
                pre_norm: bool=False,
                activation: str='gelu',
                layer_norm_eps: float=1e-5,
                device: torch.device|None=None,
                dtype: torch.dtype|None=None,
                recompute_attn: bool=False,
                mlp_use_residual:bool=False,
                layer_arch: str = 'fmfmsm',
                norm_type: str = 'layernorm',
                deepnorm_alpha: float|None = None,
                use_target_aware_embedding: bool = False,
                use_column_specific_y_aware: bool = False,
                use_logn_attention: bool = False,
                use_learnable_attn_temperature: bool = False,
                attn_n_ref: float = 1024.0,
                **layer_kwargs:Any
                ):
        super().__init__()
        
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
        self.features_per_group = features_per_group
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.device = device
        self.dtype = dtype
        self.recompute_attn = recompute_attn
        self.mlp_use_residual = mlp_use_residual
        self.layer_arch = layer_arch
        self.norm_type = norm_type
        self.deepnorm_alpha = deepnorm_alpha
        self.num_reg_quantiles = int(decoder_config.get('num_reg_quantiles', 1))
        # Bar-distribution metadata (persisted via decoder_config at training).
        # When regression_loss == 'bar_distribution', the K reg_y_decoder outputs
        # are interpreted as K bin logits over [bar_borders_low, bar_borders_high]
        # at inference time.
        self.regression_loss = str(decoder_config.get('regression_loss', 'pinball'))
        self.num_bars = int(decoder_config.get('num_bars', self.num_reg_quantiles))
        self.bar_borders_low = float(decoder_config.get('bar_borders_low', -10.0))
        self.bar_borders_high = float(decoder_config.get('bar_borders_high', 10.0))
        # Bar-borders mode: 'uniform' (linspace over [low, high]) or
        # 'normal_quantile' (N(0,1) quantile-spaced for equal mass per bin
        # under standard normal). Normal-quantile concentrates ~3-4× more
        # bins near y=0 where context-normalized targets actually live;
        # eliminates wasted tail bins. Borders are persisted as a buffer so
        # inference uses the exact same edges.
        self.bar_borders_mode = str(decoder_config.get('bar_borders_mode', 'uniform'))
        if self.regression_loss == 'bar_distribution':
            if self.bar_borders_mode == 'normal_quantile':
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
                        self.bar_borders_low, self.bar_borders_high,
                        self.num_bars + 1, dtype=torch.float32,
                    )
            else:
                bar_borders = torch.linspace(
                    self.bar_borders_low, self.bar_borders_high,
                    self.num_bars + 1, dtype=torch.float32,
                )
            # register_buffer makes it part of state_dict and follows .to()
            self.register_buffer('bar_borders_buffer', bar_borders, persistent=True)
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
        self.max_num_classes = int(
            encoder_config_y.get('max_num_classes', decoder_config.get('num_classes', 10))
        )

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
            activation=self.activation, # type: ignore
            layer_norm_eps=self.layer_norm_eps,
            device=self.device,
            dtype=self.dtype,
            recompute_attn=self.recompute_attn,
            mlp_use_residual=self.mlp_use_residual,
            layer_arch=self.layer_arch, # type: ignore
            norm_type=self.norm_type,
            deepnorm_alpha=self.deepnorm_alpha,
            use_logn_attention=self.use_logn_attention,
            use_learnable_attn_temperature=self.use_learnable_attn_temperature,
            attn_n_ref=self.attn_n_ref,
            **layer_kwargs
        )

        self.encoder_x = get_x_encoder( **encoder_config_x)
        self.cls_y_encoder = get_cls_y_encoder(**encoder_config_y)
        self.reg_y_encoder = get_reg_y_encoder(**encoder_config_y)
        if self.use_target_aware_embedding:
            self.cls_target_aware_embedding = nn.Embedding(
                self.max_num_classes,
                self.embed_dim,
                device=self.device,
                dtype=self.dtype,
            )
            self.reg_target_aware_embedding = nn.Linear(
                1,
                self.embed_dim,
                device=self.device,
                dtype=self.dtype,
            )
            nn.init.normal_(self.cls_target_aware_embedding.weight, std=0.02)
            nn.init.normal_(self.reg_target_aware_embedding.weight, std=0.02)
            nn.init.zeros_(self.reg_target_aware_embedding.bias)
        else:
            self.cls_target_aware_embedding = None
            self.reg_target_aware_embedding = None

        self.transformer_encoder = LayerStack([layer_creator() for _ in range(self.nlayers)])
        if pre_norm:
            if norm_type == 'rmsnorm':
                self.encoder_out_norm = RMSNorm(self.embed_dim, eps=1e-5, elementwise_affine=False)
            else:
                self.encoder_out_norm = nn.LayerNorm(self.embed_dim, eps=1e-5, elementwise_affine=False)
        else:
            self.encoder_out_norm = nn.Identity()

        self.cls_y_decoder = nn.Sequential(
                                            nn.Linear(self.embed_dim, self.hid_dim),
                                            nn.GELU(),
                                            nn.Linear(self.hid_dim, decoder_config['num_classes']),
                                            )
        
        self.reg_y_decoder = nn.Sequential(
                                        nn.Linear(self.embed_dim, self.hid_dim),
                                        nn.LayerNorm(self.hid_dim),
                                        nn.GELU(),
                                        nn.Linear(self.hid_dim, self.num_reg_quantiles),
                                        )
        self.feature_decoder = nn.Sequential(
                                        nn.Linear(self.embed_dim, self.hid_dim),
                                        nn.LayerNorm(self.hid_dim),
                                        nn.GELU(),
                                        nn.Linear(self.hid_dim, self.features_per_group),
                                        )
        
        if feature_positional_embedding_type == "learned":
            self.feature_positional_embedding = nn.Embedding(
                feature_positional_embedding_num_slots, self.embed_dim
            )
            nn.init.normal_(self.feature_positional_embedding.weight, std=0.02)
            self.feature_positional_embedding_num_slots = feature_positional_embedding_num_slots
        elif feature_positional_embedding_type == "subortho":
            self.feature_positional_embedding = nn.Linear(self.embed_dim // 4, self.embed_dim)
        
        self.x_preprocess = preprocesss_4_x(**preprocess_config_x)


    def forward(self, x: torch.Tensor, 
                y: torch.Tensor, 
                eval_pos: int, 
                y_type: torch.Tensor = None,
                task_type: Literal['reg', 'cls'] = 'cls',
                calculate_sample_attention: bool = False,
                calculate_feature_attention: bool = False,
                **kwargs
                ) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        '''
            x: The input x, which includes both train x and test x, Shape: [batch, sequence, feature]
            y: The input y, which includes both train y and test y, Shape: [batch, label]
            eval_pos: Train x and train y split point
            task_type: Type of task, options: cls(classification), reg(regression)
        '''
        assert x is not None and y is not None, "x and y must not be none"
        assert eval_pos > 0, "eval_pos must be a positive number"
        assert len(x.shape)==3, "x must be [Batch, seq, Feature]"
        assert len(y.shape)==2, "y must be [Batch, label]"
        assert eval_pos < x.shape[1] and eval_pos <= y.shape[1], "The split point between train x and test x must be less than the feature dimension of x, and less than or equal to the label dimension of y"
        
        batch_size, seq_len, num_feature = x.shape
        x = {'data':x, 'mask':torch.isnan(x).to(torch.int32).to(x.device)}
        y = {'data':y}
        
        feature_to_add = num_feature%self.features_per_group
        if feature_to_add > 0:
            # Extend the feature dimension of x when it is insufficient
            for k in x:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            batch_size,
                            seq_len,
                            feature_to_add,
                            device=x[k].device,
                            dtype=x[k].dtype
                        )
                    ),
                    dim=-1
                )
        for k in x:
            x[k] = x[k].reshape(batch_size, seq_len, x[k].shape[2]//self.features_per_group, self.features_per_group)
        x['eval_pos'] = eval_pos
        preprocessed_x = self.x_preprocess(x)
        preprocessed_x = self.process_4_x(preprocessed_x)
        x_encoder_result = self.encoder_x(preprocessed_x)
        x_emb_result = x_encoder_result['data']
        
        for k in y:
            # Extend the label dimension of y when it is insufficient
            y[k] = y[k].unsqueeze(-1)
            if y[k].shape[1] < x['data'].shape[1]:
                y[k] = torch.cat(
                    (
                        y[k],
                        torch.nan
                        * torch.zeros(
                            y[k].shape[0],
                            x["data"].shape[1] - y[k].shape[1],
                            y[k].shape[2],
                            device=y[k].device,
                            dtype=y[k].dtype,
                        ),
                    ),
                    dim=1
                )
        target_aware_y = y["data"].squeeze(-1).clone()
        # Mask the test y — functional (no in-place mutation of the input dict)
        seq_positions = torch.arange(y["data"].shape[1], device=y["data"].device)
        y_data_masked = torch.where(
            seq_positions.view(1, -1, 1) >= eval_pos,
            torch.tensor(float('nan'), device=y["data"].device, dtype=y["data"].dtype),
            y["data"],
        )

        # Embed y — direct encoder call (avoids mixed_y_embedding graph break).
        # Encoder output may be 4-D [B, S, 1, E] because y has a trailing
        # dim of 1; squeeze it to [B, S, E] to match the old contract.
        y_enc_input = {'data': y_data_masked, 'eval_pos': eval_pos}
        if task_type == 'cls':
            embedded_y = self.cls_y_encoder(y_enc_input)['data'].squeeze(2)
        else:
            embedded_y = self.reg_y_encoder(y_enc_input)['data'].squeeze(2)

        embedded_x = self.add_embeddings(x_emb_result)
        embedded_x = self.apply_target_aware_embedding(
            embedded_x,
            target_aware_y,
            task_type=task_type,
            eval_pos=eval_pos,
        )
        embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)
        if calculate_sample_attention or calculate_feature_attention:
            return self.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos,
                                            calculate_sample_attention=calculate_sample_attention,
                                            calculate_feature_attention=calculate_feature_attention, **kwargs)
        else:
            pass
        encoder_out = self.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos, **kwargs)[0]
        encoder_out = self.encoder_out_norm(encoder_out)

        test_encoder_out = encoder_out[:, eval_pos:, -1]
        encoder_out_4_feature = encoder_out[:, :, :-1, :]
        if self.mask_prediction:
            # Direct decoder call (avoids y_decoder graph break)
            if task_type == 'cls':
                cls_output = self.cls_y_decoder(test_encoder_out)
                reg_output = test_encoder_out.new_zeros(
                    test_encoder_out.shape[0], test_encoder_out.shape[1],
                    self.num_reg_quantiles)
            else:
                cls_output = test_encoder_out.new_zeros(
                    test_encoder_out.shape[0], test_encoder_out.shape[1],
                    self.cls_y_decoder[-1].out_features)
                reg_output = self.reg_y_decoder(test_encoder_out)
            feature_pred = self.feature_decoder(encoder_out_4_feature)
            output_decoded = {
                "cls_output": cls_output,
                "reg_output": reg_output,
                "feature_pred": feature_pred,
                "process_config": {
                    "n_x_padding": feature_to_add,
                    "features_per_group": self.x_preprocess[3].num_features,
                    "num_used_features": preprocessed_x.get(
                        '_valid_feature_num', self.x_preprocess[3].valid_feature_num),
                    "mean_for_normalization": preprocessed_x.get(
                        '_norm_mean', self.x_preprocess[2].mean),
                    "std_for_normalization": preprocessed_x.get(
                        '_norm_std', self.x_preprocess[2].std),
                }
            }
        else:
            if task_type == 'cls':
                output_decoded = self.cls_y_decoder(test_encoder_out)
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
            'data': x,
            'mask': torch.isnan(x).to(torch.int32).to(x.device),
        }
        feature_to_add = (-num_feature) % self.features_per_group
        if feature_to_add > 0:
            for k in ('data', 'mask'):
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
        for k in ('data', 'mask'):
            value = x_dict[k]
            assert isinstance(value, torch.Tensor)
            x_dict[k] = value.reshape(
                batch_size,
                seq_len,
                value.shape[2] // self.features_per_group,
                self.features_per_group,
            )
        x_dict['eval_pos'] = eval_pos
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
            n_slots = self.feature_positional_embedding_num_slots
            slot_indices = torch.randperm(n_slots, device=device)[:n_groups]
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
        encoded = self.encoder_x(x_part)['data']
        return self.apply_feature_positional_embeddings(encoded, feature_pos_emb)

    def _encode_y_full(
            self,
            y: torch.Tensor,
            *,
            total_rows: int,
            eval_pos: int,
            task_type: Literal['reg', 'cls'],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_dict: dict[str, torch.Tensor] = {'data': y}
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
            torch.tensor(float('nan'), device=y_dict["data"].device, dtype=y_dict["data"].dtype),
            y_dict["data"],
        )
        y_enc_input = {'data': y_data_masked, 'eval_pos': eval_pos}
        if task_type == 'cls':
            embedded_y = self.cls_y_encoder(y_enc_input)['data'].squeeze(2)
        else:
            embedded_y = self.reg_y_encoder(y_enc_input)['data'].squeeze(2)
        return target_aware_y, embedded_y

    def forward_cached_regression(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            eval_pos: int,
            *,
            row_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Regression-only cached prediction path for chunked inference.

        The train-side transformer states and projected sequence-attention K/Vs
        are computed once. Test rows are then streamed through the same layers
        using those train caches. Preprocessing still sees the full transductive
        X table once, preserving the existing inference semantics.
        """
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

        total_rows = x.shape[1]
        x_dict, _feature_to_add = self._build_x_preprocess_inputs(x, eval_pos)
        preprocessed_x = self.x_preprocess(x_dict)
        preprocessed_x = self.process_4_x(preprocessed_x)
        data_tensor = preprocessed_x['data']
        assert isinstance(data_tensor, torch.Tensor)
        feature_pos_emb = self.make_feature_positional_embeddings(
            data_tensor.shape[2],
            device=data_tensor.device,
            dtype=data_tensor.dtype,
        )
        target_aware_y, embedded_y = self._encode_y_full(
            y,
            total_rows=total_rows,
            eval_pos=eval_pos,
            task_type='reg',
        )

        x_train = self._encode_x_rows(
            preprocessed_x,
            slice(0, eval_pos),
            total_rows=total_rows,
            feature_pos_emb=feature_pos_emb,
        )
        x_train = self.apply_target_aware_embedding(
            x_train,
            target_aware_y[:, :eval_pos],
            task_type='reg',
            eval_pos=eval_pos,
        )
        train_tokens = torch.cat((x_train, embedded_y[:, :eval_pos].unsqueeze(2)), dim=2)
        _, caches = self.transformer_encoder.build_train_cache(
            train_tokens,
            feature_atten_mask=None,
        )

        n_test = total_rows - eval_pos
        if row_chunk_size is None or row_chunk_size <= 0:
            row_chunk_size = n_test
        outputs = []
        for start in range(eval_pos, total_rows, row_chunk_size):
            end = min(start + row_chunk_size, total_rows)
            x_test = self._encode_x_rows(
                preprocessed_x,
                slice(start, end),
                total_rows=total_rows,
                feature_pos_emb=feature_pos_emb,
            )
            test_tokens = torch.cat((x_test, embedded_y[:, start:end].unsqueeze(2)), dim=2)
            test_out = self.transformer_encoder.forward_test_with_cache(
                test_tokens,
                caches,
                feature_atten_mask=None,
            )
            test_out = self.encoder_out_norm(test_out)
            outputs.append(self.reg_y_decoder(test_out[:, :, -1]))
        return torch.cat(outputs, dim=1)

    def apply_target_aware_embedding(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        task_type: Literal['reg', 'cls'],
        eval_pos: int,
    ) -> torch.Tensor:
        scale = float(getattr(self, 'target_aware_scale', 1.0))
        if (not self.use_target_aware_embedding
                or eval_pos <= 0
                or abs(scale) < 1e-8):
            return x

        # Compute bias for context rows, pad query rows with zeros (functional)
        if task_type == 'cls':
            assert self.cls_target_aware_embedding is not None
            y_ctx = y[:, :eval_pos].to(torch.long)
            y_ctx = torch.clamp(y_ctx, min=0, max=self.max_num_classes - 1)
            ctx_bias = self.cls_target_aware_embedding(y_ctx).to(x.dtype)
        else:
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

        if (self.use_column_specific_y_aware
                and self.column_y_aware_alpha is not None):
            # Column-specific gating: each feature group gets a y-bias scaled
            # by its alignment with y_emb. Columns "in the direction of" the
            # y signal get strong y-aware bias; orthogonal columns get less.
            # This is the missing piece vs TabICL's per-column y-injection.
            #
            # col_score[b, t, c] = <x[b, t, c, :], target_bias[b, t, :]>
            # Higher score → column emb is closer to y signal → higher gate.
            inv_sqrt_d = 1.0 / (self.embed_dim ** 0.5)
            col_score = (x * row_broadcast_bias).sum(dim=-1) * inv_sqrt_d  # [B, seq, n_cols]
            col_gate = torch.sigmoid(col_score).unsqueeze(-1)               # [B, seq, n_cols, 1]
            col_specific_bias = row_broadcast_bias * col_gate               # [B, seq, n_cols, embed_dim]
            # Mix old row-broadcast with new column-specific via learned α.
            # alpha=sigmoid(α_param) starts at sigmoid(-5)≈0.007 → ~99.3%
            # original V8old behavior preserved at FT step 0.
            alpha = torch.sigmoid(self.column_y_aware_alpha).to(x.dtype)
            return x + (1.0 - alpha) * row_broadcast_bias + alpha * col_specific_bias

        return x + row_broadcast_bias
    
    def process_4_x(self, data:dict):
        x_input = data['data']
        mask = data['mask'].to(torch.bool)
        x_input = torch.where(mask, float('nan'), x_input)
        data['data'] = x_input
        return data
    
    def add_embeddings(self, x:torch.Tensor):
        if self.feature_positional_embedding_type == "subortho":
            with autocast(device_type=x.device.type, enabled=False):
                embs = torch.randn(
                    (x.shape[2], x.shape[3] // 4),
                    device=x.device,
                    dtype=torch.float32,
                )
                torch.nn.init.orthogonal_(embs)
            embs =self.feature_positional_embedding(embs.to(x.dtype))
            x += embs[None, None]
        elif self.feature_positional_embedding_type == "learned":
            # Learned positional embeddings (TabPFN-2.6 style).
            # n_slots slots in the Embedding table; each forward uses a random
            # permutation of slot indices to assign positions to feature groups.
            # This preserves permutation invariance: the model can't rely on
            # a fixed feature ↔ slot mapping across episodes.
            n_groups = x.shape[2]
            n_slots = self.feature_positional_embedding_num_slots
            slot_indices = torch.randperm(n_slots, device=x.device)[:n_groups]
            embs = self.feature_positional_embedding(slot_indices).to(x.dtype)
            x = x + embs[None, None, :, :]
        elif self.feature_positional_embedding_type is None or self.feature_positional_embedding_type == "none":
            embs = None
        else:
            raise ValueError(f"Unknown feature_positional_embedding_type={self.feature_positional_embedding_type}")
        return x
    
