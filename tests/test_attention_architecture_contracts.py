"""Regression tests for attention flags and backend-independent semantics."""

import json
import math

import pytest
import torch

from synthefy_nori.model.layer import EncoderBaseLayer, MultiheadAttention, QASSMaxScaling
from synthefy_nori.training.config import package_config_path
from synthefy_nori.utils.loading import build_model


def _attention(*, combined=True, **kwargs) -> MultiheadAttention:
    return MultiheadAttention(
        embed_dim=8,
        num_heads=2,
        qkv_combined=combined,
        device=torch.device("cpu"),
        **kwargs,
    )


@pytest.mark.parametrize("training,expected", [(True, 0.6), (False, 0.0)])
def test_torch_sdpa_dropout_tracks_module_mode(monkeypatch, training, expected):
    attention = _attention(dropout=0.6)
    attention.train(training)
    seen = []

    def fake_sdpa(q, _k, _v, **kwargs):
        seen.append(kwargs["dropout_p"])
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    qkv = torch.randn(2, 4, 3, 2, 4)
    attention.compute_attention_by_torch(qkv, None, None, None)
    assert seen == [expected]


def test_eval_attention_with_configured_dropout_is_deterministic():
    torch.manual_seed(0)
    attention = _attention(dropout=0.9).eval()
    x = torch.randn(1, 2, 5, 8)
    with torch.no_grad():
        first = attention(x)[0]
        second = attention(x)[0]
    assert torch.equal(first, second)


def test_feature_mask_uses_sdpa_true_means_allowed_contract():
    layer = EncoderBaseLayer(embed_dim=8, hid_dim=16, nhead=2, layer_arch="smf")
    valid = torch.tensor([[[True, False, True], [True, True, False]]])
    mask = layer.create_attn_mask(valid, valid)

    expected = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    expected = expected.reshape(-1, 3, 3).unsqueeze(1).expand(-1, 2, -1, -1)
    assert torch.equal(mask, expected)
    assert mask[0, 0, 0, 0]
    assert not mask[0, 0, 0, 1]


@pytest.mark.parametrize(
    "mode,missing_components",
    [("log_only", ("base_mlp", "gate_mlp")), ("base_only", ("gate_mlp",))],
)
def test_qass_omits_unused_components_but_loads_legacy_weights_strictly(mode, missing_components):
    legacy = QASSMaxScaling(num_heads=2, head_dim=4, qass_mode="full")
    current = QASSMaxScaling(num_heads=2, head_dim=4, qass_mode=mode)

    for component in missing_components:
        assert getattr(current, component) is None
        assert not any(key.startswith(f"{component}.") for key in current.state_dict())

    incompatible = current.load_state_dict(legacy.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


@pytest.mark.parametrize("mode", ["log_only", "base_only"])
def test_qass_pruning_preserves_legacy_live_initialization_and_rng(mode):
    kwargs = dict(
        embed_dim=8,
        hid_dim=16,
        nhead=2,
        layer_arch="smf",
        seq_attn_isolated=True,
        use_qassmax=True,
    )
    torch.manual_seed(42)
    legacy = EncoderBaseLayer(**kwargs, qass_mode="full")
    legacy_rng_state = torch.get_rng_state()

    torch.manual_seed(42)
    slim = EncoderBaseLayer(**kwargs, qass_mode=mode)
    slim_rng_state = torch.get_rng_state()

    legacy_state = legacy.state_dict()
    slim_state = slim.state_dict()
    if mode == "log_only":
        assert not any("qassmax.base_mlp." in key for key in slim_state)
    assert not any("qassmax.gate_mlp." in key for key in slim_state)
    assert all(torch.equal(value, legacy_state[key]) for key, value in slim_state.items())
    assert torch.equal(slim_rng_state, legacy_rng_state)


def test_log_only_qass_has_no_parameters_and_keeps_exact_formula():
    qass = QASSMaxScaling(num_heads=2, head_dim=4, qass_mode="log_only")
    assert list(qass.parameters()) == []
    q = torch.randn(1, 3, 2, 4)
    assert torch.allclose(qass(q, 17), q * math.log(17.0))


def test_mlp_use_residual_true_fails_fast_in_direct_layer():
    with pytest.raises(ValueError, match="legacy flag was a no-op"):
        EncoderBaseLayer(
            embed_dim=8,
            hid_dim=16,
            nhead=2,
            layer_arch="smf",
            mlp_use_residual=True,
        )


def test_mlp_use_residual_true_fails_fast_when_loading_config():
    with open(package_config_path("model_base.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(nlayers=1, mlp_use_residual=True)
    with pytest.raises(ValueError, match="legacy flag was a no-op"):
        build_model(config)
