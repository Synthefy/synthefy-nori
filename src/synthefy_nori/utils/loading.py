from __future__ import annotations

import torch
import random
import numpy as np
from synthefy_nori.model.transformer import FeaturesTransformer

def build_model(config:dict):
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
