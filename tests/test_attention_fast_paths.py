from __future__ import annotations

import pytest
import torch

import synthefy_nori.model.layer as layer_module
from synthefy_nori.model.layer import MultiheadAttention


@pytest.mark.parametrize("copy_first_head_kv", [False, True])
def test_cached_kv_projection_matches_einsum(copy_first_head_kv):
    torch.manual_seed(0)
    attention = MultiheadAttention(
        embed_dim=8,
        num_heads=2,
        qkv_combined=False,
    )
    x_kv = torch.randn(2, 3, 5, 8)
    x_kv_flat = x_kv.reshape(-1, *x_kv.shape[-2:])
    weights = attention.qkv_proj_weight[1:]
    if copy_first_head_kv:
        weights = weights[:, :1]
    expected = torch.einsum("... s, j h d s -> ... j h d", x_kv_flat, weights)

    cache = attention.project_kv_cache(
        x_kv,
        copy_first_head_kv=copy_first_head_kv,
    )

    assert torch.equal(cache["kv"], expected)


def test_cached_q_linear_projection_matches_einsum(monkeypatch):
    torch.manual_seed(1)
    attention = MultiheadAttention(
        embed_dim=8,
        num_heads=2,
        qkv_combined=False,
    )
    x = torch.randn(2, 3, 7, 8)
    x_flat = x.reshape(-1, *x.shape[-2:])
    expected = torch.einsum(
        "... s, h d s -> ... h d",
        x_flat,
        attention.qkv_proj_weight[0],
    )
    captured = {}

    def capture_attention(qkv, q, kv, attn_mask, sdpa_scale=None):
        captured["q"] = q
        return q

    monkeypatch.setattr(attention, "compute_attention_by_torch", capture_attention)
    cache = {
        "kv": torch.zeros(6, 5, 2, 2, 4),
        "batch": 2,
        "groups": 3,
    }

    attention.forward_with_kv_cache(x, cache)

    assert torch.equal(captured["q"], expected)


@pytest.mark.parametrize(
    ("batch", "expected_calls"),
    [
        pytest.param(2, 1, id="within-limit"),
        pytest.param(3, 2, id="above-limit"),
    ],
)
def test_sdpa_chunks_only_above_cuda_batch_head_limit(monkeypatch, batch, expected_calls):
    attention = MultiheadAttention(
        embed_dim=2,
        num_heads=2,
        qkv_combined=False,
    )
    q = torch.zeros(batch, 1, 2, 1)
    kv = torch.zeros(batch, 1, 2, 2, 1)
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append(query.shape)
        return query

    monkeypatch.setattr(layer_module, "SDPA_BATCH_HEAD_LIMIT", 5)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    output = attention.compute_attention_by_torch(None, q, kv, None)

    assert output.shape == q.shape
    assert len(calls) == expected_calls


def test_sdpa_batch_head_limit_uses_full_cuda_grid_capacity():
    assert layer_module.SDPA_BATCH_HEAD_LIMIT == 65_535
