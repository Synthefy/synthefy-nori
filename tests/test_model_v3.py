"""Model v3 (Nori 1.5) inference support: the two-phase encoder and its tokenizer hooks.

This tier carries Model v3 for inference only. The checks here cover what the port adds and
what it must leave alone: the two-phase forward, the prompt-quantile PLE tokenizer, the
options the loader refuses, and that dense checkpoints still build the original modules.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch

from synthefy_nori.configs import DEFAULT_MODEL_CONFIG, config_path
from synthefy_nori.model.encoders import MaskEmbEncoder
from synthefy_nori.model.layer import MultiheadAttention
from synthefy_nori.utils.loading import build_model, finalize_arch_config

SMALL_TWO_PHASE_CONFIG = {"target_tokens": 2, "row_layers": 3, "row_heads": 4, "row_hidden_dim": 96}


def _get_small_two_phase_model_config(mask_prediction: bool = False, *, features_per_group: int = 1) -> dict:
    """Return a small Model v3 regression config built on the bundled base model config."""
    with open(config_path(DEFAULT_MODEL_CONFIG)) as stream:
        config = copy.deepcopy(json.load(stream))
    config.update(
        nlayers=2,
        nhead=2,
        embed_dim=32,
        hid_dim=64,
        features_per_group=features_per_group,
        feature_group_circular=False,
        activation="swiglu",
        norm_type="rmsnorm",
        pre_norm=True,
        deepnorm_alpha=5**-0.5,
        two_phase_config=dict(SMALL_TWO_PHASE_CONFIG),
        mask_prediction=mask_prediction,
        use_prompt_quantile_ple=True,
        prompt_quantile_ple_bins=16,
        use_target_aware_embedding=True,
        use_column_specific_y_aware=True,
        use_qassmax=True,
        qass_mode="log_only",
        omit_feature_decoder=True,
    )
    config["decoder_config"] = {"num_reg_quantiles": 9, "regression_loss": "pinball"}
    for key in ("encoder_config_x", "encoder_config_y"):
        for field in ("embedding_size", "mask_embedding_size"):
            if field in config[key]:
                config[key][field] = 32
    config["preprocess_config_x"]["num_features"] = features_per_group
    config["encoder_config_x"]["num_features"] = features_per_group
    return finalize_arch_config(config)


def test_two_phase_model_is_regression_only():
    model = build_model(_get_small_two_phase_model_config())
    keys = model.state_dict()
    assert "transformer_encoder.target_encoder.mlp.0.weight" in keys
    assert not any(key.startswith("cls_") for key in keys)
    assert model.supports_context_cache is False
    with pytest.raises(ValueError, match="regression only"):
        model(x=torch.randn(1, 12, 3), y=torch.randn(1, 8), eval_pos=8, task_type="cls")


@pytest.mark.parametrize("features_per_group", [1, 2])
def test_query_rows_never_see_each_other(features_per_group):
    torch.manual_seed(0)
    model = build_model(_get_small_two_phase_model_config(features_per_group=features_per_group)).eval()
    x = torch.randn(1, 16, 5)
    y = torch.randn(1, 16)
    with torch.no_grad():
        full = model(x=x, y=y, eval_pos=10, task_type="reg")
        x_alt = x.clone()
        x_alt[0, 15] = torch.randn(5)  # perturb the last query row only
        alternate_labels = y.clone()
        alternate_labels[:, 10:] = 1000.0
        alt = model(x=x_alt, y=alternate_labels, eval_pos=10, task_type="reg")
    assert full.shape == (1, 6, 9)
    assert torch.allclose(full[:, :-1], alt[:, :-1], atol=1e-5)
    assert not torch.allclose(full[:, -1], alt[:, -1], atol=1e-5)


def test_two_phase_refuses_the_context_cache():
    model = build_model(_get_small_two_phase_model_config()).eval()
    x = torch.randn(1, 12, 3)
    y = torch.randn(1, 8)
    with pytest.raises(NotImplementedError, match="two-phase"):
        model.forward_cached_regression(x, y, eval_pos=8)
    with pytest.raises(NotImplementedError, match="two-phase"):
        model.build_context_cache(x[:, :8], y)


def test_every_two_phase_attention_normalises_queries_and_keys():
    model = build_model(_get_small_two_phase_model_config())
    attentions = [m for m in model.transformer_encoder.modules() if isinstance(m, MultiheadAttention)]
    assert attentions and all(m.qk_norm for m in attentions)
    # Scores no longer depend on the scale of the query and key projections.
    torch.manual_seed(0)
    attention = MultiheadAttention(64, 4, qkv_combined=False, use_qassmax=True, qass_mode="log_only", qk_norm=True)
    x = torch.randn(2, 3, 20, 64)
    with torch.no_grad():
        before = attention(x=x, x_kv=x[:, :, :12])[0]
        attention.qkv_proj_weight[:2].mul_(10.0)
        after = attention(x=x, x_kv=x[:, :, :12])[0]
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-4)


def test_dense_attention_keeps_its_projection_path():
    attention = MultiheadAttention(64, 4, qkv_combined=False)
    assert attention.qk_norm is False and attention.q_norm is None
    assert attention._is_linear_projection_used(torch.device("cpu")) is False
    assert MultiheadAttention(64, 4, qk_norm=True)._is_linear_projection_used(torch.device("mps")) is False


def test_ple_bins_split_the_cell_token_and_default_keeps_full_width():
    rbf_config = {
        "token_embed_dim": 16,
        "n_kernels": 64,
        "sigma": 0.5,
        "use_learn_sigma": False,
        "use_learn_embeddings": False,
        "use_original_features": False,
    }
    dense = MaskEmbEncoder(1, 32, 32, numeric_embed_type="RBF", RBF_config=rbf_config)
    assert dense.numeric_mlp.out_layer.out_features == 32
    with_ple = MaskEmbEncoder(1, 32, 32, numeric_embed_type="RBF", RBF_config=rbf_config, prompt_quantile_ple_bins=8)
    assert with_ple.numeric_mlp.out_layer.out_features == 24
    with pytest.raises(TypeError, match="reference is required"):
        with_ple({"data": torch.randn(1, 4, 2, 1)})


@pytest.mark.parametrize(
    "option,value",
    [("input_transform", "asinh"), ("compute_dtype", "float32"), ("use_raw_input_residual", True)],
)
def test_rbf_options_this_release_lacks_are_refused(option, value):
    config = _get_small_two_phase_model_config()
    config["encoder_config_x"]["RBF_config"][option] = value
    with pytest.raises(ValueError, match="Unsupported RBF_config"):
        build_model(config)


@pytest.mark.parametrize(
    "option,value",
    [
        ("feature_group_circular", True),
        ("use_column_deepset", True),
        ("use_conditional_landmarks", True),
        ("task_type", "cls"),
    ],
)
def test_research_only_options_are_refused(option, value):
    config = _get_small_two_phase_model_config()
    config[option] = value
    with pytest.raises(ValueError, match="does not support"):
        build_model(config)


def test_released_model_v3_flags_set_off_are_accepted():
    config = _get_small_two_phase_model_config()
    config.update(
        use_column_deepset=False,
        column_deepset_hidden_dim=None,
        use_conditional_landmarks=False,
        conditional_landmark_embedding_dim=32,
    )
    config["encoder_config_x"]["RBF_config"].update(
        input_transform="scientific", compute_dtype="float64", use_raw_input_residual=False
    )
    build_model(config)
