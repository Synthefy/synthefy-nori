import torch
import torch.nn as nn
from synthefy_tabular.model.layer import EncoderBaseLayer, MLP, LayerStack, RMSNorm
from typing import Any, Literal
from synthefy_tabular.model.encoders import get_x_encoder, get_cls_y_encoder, get_reg_y_encoder, preprocesss_4_x
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
        elif feature_positional_embedding_type == "subspace":
            self.feature_positional_embedding = nn.Linear(self.embed_dim // 4, self.embed_dim)
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

    
    @torch.compiler.disable
    def mixed_y_embedding(self, y:dict, y_type:torch.Tensor, eval_pos:int):
        y = y['data']
        batch_size, seq_len, y_num = y.shape
        y_flat = y.reshape(-1)
        y_type_flat = y_type.reshape(-1)
        
        idx = torch.arange(len(y_flat), device=y.device)
        idx_cls = idx[y_type_flat == 0]
        idx_reg = idx[y_type_flat == 1]
        y_cls = y_flat[idx_cls]
        y_reg = y_flat[idx_reg]

        y_cls = y_cls.reshape(-1, seq_len, y_num)
        y_reg = y_reg.reshape(-1, seq_len, y_num)
        y_cls = {'data': y_cls, 'eval_pos':eval_pos}
        y_reg = {'data': y_reg, 'eval_pos':eval_pos}

        cls_y_emb = self.cls_y_encoder(y_cls) if len(idx_cls) > 0 else None
        reg_y_emb = self.reg_y_encoder(y_reg) if len(idx_reg) > 0 else None
        cls_y_emb = cls_y_emb['data'] if cls_y_emb is not None else None
        reg_y_emb = reg_y_emb['data'] if reg_y_emb is not None else None
        
        emb_size = self.embed_dim
        # Determine dtype from encoder outputs (bf16 under autocast) to avoid
        # index_put dtype mismatch between float32 y_flat and bf16 embeddings.
        if cls_y_emb is not None:
            out_dtype = cls_y_emb.dtype
        elif reg_y_emb is not None:
            out_dtype = reg_y_emb.dtype
        else:
            out_dtype = y_flat.dtype
        out = torch.empty(len(y_flat), emb_size, dtype=out_dtype, device=y_flat.device)
        if cls_y_emb is not None:
            cls_y_emb_flat = cls_y_emb.reshape(-1, emb_size)
            out.index_put_((idx_cls,), cls_y_emb_flat)

        if reg_y_emb is not None:
            reg_y_emb_flat = reg_y_emb.reshape(-1, emb_size)
            out.index_put_((idx_reg,), reg_y_emb_flat)

        output = out.reshape(batch_size, seq_len, emb_size)
        return output

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
    
    @torch.compiler.disable
    def y_decoder(self, test_encoder_out, test_y_type):
        _, seq_len, emb_size = test_encoder_out.shape
        flat_test_encoder_out = test_encoder_out.reshape(-1, emb_size)
        flat_test_y_type = test_y_type.reshape(-1)
        
        idx = torch.arange(len(flat_test_encoder_out), device=flat_test_encoder_out.device)
        idx_cls = idx[flat_test_y_type == 0]
        idx_reg = idx[flat_test_y_type == 1]

        cls_y_encoder_out = flat_test_encoder_out[idx_cls]
        reg_y_encoder_out = flat_test_encoder_out[idx_reg]
        cls_y_encoder_out = cls_y_encoder_out.reshape(-1, seq_len, emb_size)
        reg_y_encoder_out = reg_y_encoder_out.reshape(-1, seq_len, emb_size)

        cls_y = self.cls_y_decoder(cls_y_encoder_out)
        reg_y = self.reg_y_decoder(reg_y_encoder_out)

        return cls_y, reg_y
    