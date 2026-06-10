from __future__ import annotations

import torch
import random
import numpy as np
from synthefy_tabular.model.transformer import FeaturesTransformer

def build_model(config:dict):
    if config.get('architecture') == 'v14':
        from synthefy_tabular.model.transformer_v14 import FeaturesTransformerV14

        decoder_config = config.get('decoder_config', {})
        return FeaturesTransformerV14(
            embed_dim=config.get('embed_dim', 128),
            feature_group_size=config.get('v14_feature_group_size', 3),
            num_cls_tokens=config.get('v14_num_cls_tokens', 4),
            dist_embed_num_blocks=config.get('v14_dist_embed_num_blocks', 3),
            dist_embed_num_inducing_points=config.get('v14_dist_embed_num_inducing_points', 128),
            dist_embed_num_heads=config.get('v14_dist_embed_num_heads', 8),
            feat_agg_num_blocks=config.get('v14_feat_agg_num_blocks', 3),
            feat_agg_num_heads=config.get('v14_feat_agg_num_heads', 8),
            icl_num_layers=config.get('v14_icl_num_layers', 16),
            icl_num_heads=config.get('v14_icl_num_heads', 8),
            icl_num_kv_heads_test=config.get('v14_icl_num_kv_heads_test', 1),
            shared_test_kv_projection=config.get('v14_shared_test_kv_projection', False),
            num_reg_quantiles=decoder_config.get(
                'num_reg_quantiles',
                config.get('num_reg_quantiles', 999),
            ),
            ff_factor=config.get('v14_ff_factor', 2),
            rope_base=config.get('v14_rope_base', 100_000.0),
            native_missing_indicators=config.get('v14_native_missing_indicators', True),
            context_standardize=config.get('v14_context_standardize', False),
            input_clip_value=config.get('v14_input_clip_value', 100.0),
            inference_row_chunk_size=config.get('v14_inference_row_chunk_size', 2048),
            device=config.get('device', None),
            dtype=config.get('dtype', None),
        )

    model = FeaturesTransformer(
        preprocess_config_x=config['preprocess_config_x'],
        encoder_config_x=config['encoder_config_x'],
        encoder_config_y=config['encoder_config_y'],
        decoder_config=config['decoder_config'],
        feature_positional_embedding_type=config.get('feature_positional_embedding_type', "subortho"),
        feature_positional_embedding_num_slots=config.get('feature_positional_embedding_num_slots', 1000),
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        embed_dim=config['embed_dim'],
        hid_dim=config['hid_dim'],
        mask_prediction=config.get('mask_prediction', False),
        features_per_group=config['features_per_group'],
        dropout=config['dropout'],
        pre_norm=config.get('pre_norm', True),
        activation=config.get('activation', 'gelu'),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        device=config.get('device', None),
        dtype=config.get('dtype', None),
        recompute_attn=config['recompute_attn'],
        layer_arch=config.get('layer_arch', 'fmfmsm'),
        norm_type=config.get('norm_type', 'layernorm'),
        deepnorm_alpha=config.get('deepnorm_alpha', None),
        self_share_all_kv_heads=config.get('self_share_all_kv_heads', False),
        cross_share_all_kv_heads=config.get('cross_share_all_kv_heads', True),
        seq_attn_isolated=config.get('seq_attn_isolated', False),
        seq_attn_serial=config.get('seq_attn_serial', False),
        use_qassmax=config.get('use_qassmax', False),
        use_target_aware_embedding=config.get('use_target_aware_embedding', False),
        use_column_specific_y_aware=config.get('use_column_specific_y_aware', False),
        use_logn_attention=config.get('use_logn_attention', False),
        use_learnable_attn_temperature=config.get('use_learnable_attn_temperature', False),
        attn_n_ref=float(config.get('attn_n_ref', 1024.0)),
    )
    return model

def load_model(model_path, mask_prediction:bool=False, base_config_path:str=None):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    # Support both pretrained (.ckpt) and training checkpoint (.pt) formats
    if 'model_config' in state_dict:
        # Training checkpoint format (new): has architecture config embedded
        config = state_dict['model_config']
        weights = state_dict.get('ema_state_dict') or state_dict['model_state_dict']
    elif 'model_state_dict' in state_dict:
        # Training checkpoint format (legacy): no model_config saved
        # Fall back to extracting config from base pretrained checkpoint
        if base_config_path is None:
            raise KeyError(
                f"Training checkpoint at {model_path} does not contain 'model_config'. "
                "Provide base_config_path pointing to the pretrained .ckpt file."
            )
        print(f"Loading model architecture config from {base_config_path}")
        base_state = torch.load(base_config_path, map_location="cpu", weights_only=False)
        config = base_state['config']
        weights = state_dict.get('ema_state_dict') or state_dict['model_state_dict']
    else:
        # Pretrained .ckpt format: {'state_dict': ..., 'config': ...}
        config = state_dict['config']
        weights = state_dict['state_dict']

    config['mask_prediction'] = mask_prediction

    # Strip torch.compile "_orig_mod." prefix if present
    if any(k.startswith('_orig_mod.') for k in weights):
        weights = {k.removeprefix('_orig_mod.'): v for k, v in weights.items()}

    model = build_model(config)
    model.load_state_dict(weights)

    model.eval()
    return model
