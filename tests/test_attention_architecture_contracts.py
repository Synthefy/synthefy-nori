"""Regression tests for attention flags and backend-independent semantics."""

import json
import math

import pytest
import torch

import synthefy_nori.model.layer as layer_module
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


@pytest.mark.parametrize("training,expected", [(True, 0.6), (False, 0.0)])
def test_flash2_packed_dropout_tracks_module_mode(monkeypatch, training, expected):
    attention = _attention(dropout=0.6)
    attention.train(training)
    seen = []

    def fake_flash(qkv, *_args, **kwargs):
        seen.append(kwargs["dropout_p"])
        return torch.zeros_like(qkv[:, 0])

    monkeypatch.setattr(layer_module, "HAVE_FLASH_ATTN", True)
    monkeypatch.setattr(layer_module, "flash_attn_varlen_qkvpacked_func", fake_flash, raising=False)
    attention.compute_attention_by_flashattn(torch.randn(2, 4, 3, 2, 4), None, None)
    assert seen == [expected]


@pytest.mark.parametrize("training,expected", [(True, 0.6), (False, 0.0)])
def test_flash2_kvpacked_dropout_tracks_module_mode(monkeypatch, training, expected):
    attention = _attention(combined=False, dropout=0.6)
    attention.train(training)
    seen = []

    def fake_flash(q, _kv, *_args, **kwargs):
        seen.append(kwargs["dropout_p"])
        return torch.zeros_like(q)

    monkeypatch.setattr(layer_module, "HAVE_FLASH_ATTN", True)
    monkeypatch.setattr(layer_module, "flash_attn_varlen_kvpacked_func", fake_flash, raising=False)
    attention.compute_attention_by_flashattn(
        None,
        torch.randn(2, 3, 2, 4),
        torch.randn(2, 5, 2, 2, 4),
    )
    assert seen == [expected]


def test_eval_attention_with_configured_dropout_is_deterministic():
    torch.manual_seed(0)
    attention = _attention(dropout=0.9).eval()
    x = torch.randn(1, 2, 5, 8)
    with torch.no_grad():
        first = attention(x)[0]
        second = attention(x)[0]
    assert torch.equal(first, second)


def test_qass_fold_keeps_explicit_logn_and_temperature_active(monkeypatch):
    monkeypatch.setenv("SYNTHEFY_QASS_SDPA_SCALE", "1")
    torch.manual_seed(0)
    control = _attention(use_qassmax=True, qass_mode="log_only")
    scaled = _attention(
        use_qassmax=True,
        qass_mode="log_only",
        use_logn_attention=True,
        use_learnable_attn_temperature=True,
        attn_n_ref=2.0,
    )
    with torch.no_grad():
        scaled.qkv_proj_weight.copy_(control.qkv_proj_weight)
        scaled.out_proj_weight.copy_(control.out_proj_weight)
    x = torch.randn(1, 2, 5, 8)

    control_out = control(x)[0]
    scaled_out = scaled(x)[0]
    assert not torch.allclose(scaled_out, control_out)

    scaled_out.square().sum().backward()
    assert scaled.attn_temperature is not None
    assert scaled.attn_temperature.grad is not None
    assert torch.isfinite(scaled.attn_temperature.grad).all()
    assert scaled.attn_temperature.grad.abs().sum() > 0


def test_qass_fold_scale_matches_materialized_qass_with_extra_scale(monkeypatch):
    torch.manual_seed(1)
    monkeypatch.setenv("SYNTHEFY_QASS_SDPA_SCALE", "1")
    folded = _attention(
        use_qassmax=True,
        qass_mode="log_only",
        use_logn_attention=True,
        use_learnable_attn_temperature=True,
        attn_n_ref=3.0,
    ).eval()
    monkeypatch.setenv("SYNTHEFY_QASS_SDPA_SCALE", "0")
    materialized = _attention(
        use_qassmax=True,
        qass_mode="log_only",
        use_logn_attention=True,
        use_learnable_attn_temperature=True,
        attn_n_ref=3.0,
    ).eval()
    materialized.load_state_dict(folded.state_dict())
    x = torch.randn(1, 2, 5, 8)
    with torch.no_grad():
        folded_out = folded(x)[0]
        materialized_out = materialized(x)[0]
    assert torch.allclose(folded_out, materialized_out, atol=1e-6, rtol=1e-5)


def test_cached_attention_keeps_qass_fold_and_extra_scale(monkeypatch):
    monkeypatch.setenv("SYNTHEFY_QASS_SDPA_SCALE", "1")
    torch.manual_seed(2)
    attention = _attention(
        combined=False,
        use_qassmax=True,
        qass_mode="log_only",
        use_logn_attention=True,
        use_learnable_attn_temperature=True,
        attn_n_ref=2.0,
    ).eval()
    x_query = torch.randn(1, 2, 3, 8)
    x_context = torch.randn(1, 2, 5, 8)
    cache = attention.project_kv_cache(x_context)
    with torch.no_grad():
        direct = attention(x_query, x_kv=x_context)[0]
        cached = attention.forward_with_kv_cache(x_query, cache)[0]
    assert torch.allclose(direct, cached, atol=1e-6, rtol=1e-5)


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
    assert torch.equal(qass(q, 1), q)


@pytest.mark.parametrize(
    "attention_kwargs",
    [
        {"use_qassmax": True, "qass_mode": "log_only"},
        {"use_logn_attention": True, "attn_n_ref": 16.0},
    ],
    ids=["qass-sdpa-fold", "explicit-logn"],
)
def test_compiled_log_length_scaling_is_shape_generic(monkeypatch, attention_kwargs):
    monkeypatch.setenv("SYNTHEFY_QASS_SDPA_SCALE", "1")
    torch.manual_seed(20260824)
    attention = _attention(**attention_kwargs).eval()
    compiled_graphs = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compiled_graphs
        compiled_graphs += 1
        return graph_module.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        attention,
        backend=counting_backend,
        dynamic=True,
        fullgraph=False,
    )
    try:
        with torch.no_grad():
            for key_len in (4, 8, 16, 32, 64, 128, 192):
                x = torch.randn(1, 2, key_len, 8)
                expected = attention(x)[0]
                actual = compiled(x)[0]
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert compiled_graphs == 1
    finally:
        torch._dynamo.reset()


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
