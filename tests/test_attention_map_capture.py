"""Attention-map diagnostics must follow the actual final architecture layer."""

import itertools

import pytest
import torch

from synthefy_nori.model.layer import EncoderBaseLayer, LayerStack


CPU = torch.device("cpu")


def _build_stack(n_layers: int, arch: str, pre_norm: bool) -> LayerStack:
    return LayerStack(
        [
            EncoderBaseLayer(
                embed_dim=8,
                hid_dim=16,
                nhead=2,
                pre_norm=pre_norm,
                layer_arch=arch,
                norm_type="rmsnorm",
                deepnorm_alpha=0.25,
                seq_attn_isolated=True,
                device=CPU,
            )
            for _ in range(n_layers)
        ]
    ).eval()


@pytest.mark.parametrize(
    "arch,pre_norm,n_layers",
    list(itertools.product(["smf", "fmfmsm"], [True, False], [3, 16])),
)
def test_capture_works_for_both_arches_norm_orders_and_depths(arch, pre_norm, n_layers):
    torch.manual_seed(0)
    stack = _build_stack(n_layers, arch, pre_norm)
    x = torch.randn(1, 6, 3, 8)

    with torch.no_grad():
        _, feature_attention, sample_attention = stack(
            x,
            feature_atten_mask=None,
            eval_pos=4,
            calculate_feature_attention=True,
            calculate_sample_attention=True,
        )

    case = f"arch={arch}, pre_norm={pre_norm}, layers={n_layers}"
    assert feature_attention is not None, case
    assert sample_attention is not None, case


def test_capture_points_are_derived_from_architecture():
    smf = _build_stack(1, "smf", True).layers[0]
    assert (smf._sequence_capture_idx, smf._feature_capture_idx) == (0, 2)

    fmfmsm = _build_stack(1, "fmfmsm", True).layers[0]
    assert (fmfmsm._sequence_capture_idx, fmfmsm._feature_capture_idx) == (4, 2)


def test_stack_returns_final_layer_maps_not_an_earlier_capture(monkeypatch):
    stack = _build_stack(3, "smf", True)

    for layer_idx, layer in enumerate(stack.layers):
        for attention in [*layer.feature_attentions, *layer.sequence_attentions]:
            marker = torch.tensor(float(layer_idx))

            def score(_q, _k, *, softmax_scale=None, marker=marker):
                return marker

            monkeypatch.setattr(attention, "caculate_attention_score", score)

    with torch.no_grad():
        _, feature_attention, sample_attention = stack(
            torch.randn(1, 6, 3, 8),
            feature_atten_mask=None,
            eval_pos=4,
            calculate_feature_attention=True,
            calculate_sample_attention=True,
        )

    assert feature_attention.item() == 2.0
    assert sample_attention.item() == 2.0


def test_no_diagnostic_flags_returns_no_maps():
    stack = _build_stack(3, "smf", True)
    with torch.no_grad():
        _, feature_attention, sample_attention = stack(
            torch.randn(1, 6, 3, 8), feature_atten_mask=None, eval_pos=4
        )
    assert feature_attention is None
    assert sample_attention is None
