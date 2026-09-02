"""CPU tests for bounded-GPU context/KV streaming.

These tests exercise the math and storage contract without requiring CUDA. Device
transfer is still traversed (CPU -> CPU), so a future implementation that asks the
cache for a full ``["kv"]`` tensor fails the same spy on every backend.
"""

from __future__ import annotations

import pytest
import torch

import synthefy_nori.model.kv_cache_scaling as kv_cache_scaling
import synthefy_nori.model.layer as layer_module

from synthefy_nori.model.kv_cache_scaling import (
    BlockwiseSeqKV,
    ScalableSeqKV,
)
from synthefy_nori.model.layer import MultiheadAttention


def _row_slices(kv: torch.Tensor, rows: tuple[int, ...]):
    start = 0
    for width in rows:
        yield kv[:, start : start + width]
        start += width
    assert start == kv.shape[1]


def test_resident_int8_preallocates_without_cat(monkeypatch):
    torch.manual_seed(17)
    kv = torch.randn(2, 13, 2, 1, 8)

    def forbid_cat(*args, **kwargs):
        raise AssertionError("resident blockwise construction must not call torch.cat")

    monkeypatch.setattr(kv_cache_scaling.torch, "cat", forbid_cat)
    cache = BlockwiseSeqKV.from_row_slices(
        _row_slices(kv, (4, 6, 3)),
        1,
        2,
        quantize=True,
        device=torch.device("cpu"),
        dtype=kv.dtype,
        offload=False,
        n_rows=13,
    )
    assert cache.quantized is True
    assert cache.offloaded is False
    assert cache.n_rows == 13
    assert cache.max_block_rows == 6
    assert cache.host_bytes() == 0
    assert cache.resident_gpu_bytes() > 0
    assert len({payload.untyped_storage().data_ptr() for payload, _ in cache._blocks}) == 1
    assert len({scale.untyped_storage().data_ptr() for _, scale in cache._blocks}) == 1

    monkeypatch.undo()
    expected = ScalableSeqKV(kv, 1, 2, quantize=True, offload=False, device=torch.device("cpu"))["kv"]
    actual = torch.cat(list(cache.iter_kv_blocks()), dim=1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_blockwise_bf16_transport_reassembles_original(dtype):
    torch.manual_seed(0)
    kv = torch.randn(2, 11, 2, 3, 4, dtype=dtype)
    cache = BlockwiseSeqKV.from_row_slices(
        _row_slices(kv, (3, 5, 3)),
        1,
        2,
        quantize=False,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    got = torch.cat(list(cache.iter_kv_blocks()), dim=1)
    assert torch.equal(got, kv)
    assert cache.resident_gpu_bytes() == 0
    assert cache.host_bytes() == kv.numel() * kv.element_size()
    with pytest.raises(RuntimeError, match="cannot materialize a full layer"):
        cache["kv"]


def test_blockwise_int8_matches_full_cache_dequantization_exactly():
    torch.manual_seed(1)
    kv = torch.randn(2, 13, 2, 1, 8)
    full = ScalableSeqKV(kv, 1, 2, quantize=True, offload=True, device=torch.device("cpu"))["kv"]
    cache = BlockwiseSeqKV.from_row_slices(
        _row_slices(kv, (4, 6, 3)),
        1,
        2,
        quantize=True,
        device=torch.device("cpu"),
        dtype=kv.dtype,
    )
    got = torch.cat(list(cache.iter_kv_blocks()), dim=1)
    # Quantization is per row/head vector, so slice-wise and whole-cache paths
    # must produce exactly the same payload/scales and dequantized values.
    assert torch.equal(got, full)


def test_ragged_stored_blocks_are_resliced_before_staging():
    torch.manual_seed(2)
    kv = torch.randn(1, 17, 2, 2, 4)
    cache = BlockwiseSeqKV.from_row_slices(
        _row_slices(kv, (5, 7, 5)),
        1,
        1,
        quantize=False,
        device=torch.device("cpu"),
        dtype=kv.dtype,
    )
    blocks = list(cache.iter_kv_blocks(block_rows=3))
    assert max(block.shape[1] for block in blocks) <= 3
    assert cache.max_staged_rows == 3
    assert torch.equal(torch.cat(blocks, dim=1), kv)


def _attention_pair(
    *, n_heads: int, copy_first_head: bool, dtype: torch.dtype, key_rows: int, stored_rows: tuple[int, ...]
):
    torch.manual_seed(3)
    embed_dim = n_heads * 4
    attn = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=n_heads,
        qkv_combined=False,
        dropout=0.0,
        dtype=dtype,
        use_logn_attention=True,
        use_learnable_attn_temperature=True,
    ).eval()
    x_kv = torch.randn(1, 2, key_rows, embed_dim, dtype=dtype)
    x_q = torch.randn(1, 2, 5, embed_dim, dtype=dtype)
    full = attn.project_kv_cache(x_kv, copy_first_head_kv=copy_first_head)
    projected = full["kv"]
    streamed = BlockwiseSeqKV.from_row_slices(
        _row_slices(projected, stored_rows),
        1,
        2,
        quantize=False,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    with torch.inference_mode():
        expected = attn.forward_with_kv_cache(x_q, full)[0]
        got = attn.forward_with_kv_cache(x_q, streamed)[0]
    return expected.float(), got.float(), streamed


@pytest.mark.parametrize(
    "n_heads,copy_first_head",
    [pytest.param(2, False, id="mha"), pytest.param(4, True, id="mqa")],
)
@pytest.mark.parametrize(
    "key_rows,stored_rows",
    [
        pytest.param(7, (1, 1, 1, 1, 1, 1, 1), id="block-1"),
        pytest.param(11, (3, 5, 3), id="odd-ragged"),
        pytest.param(5, (5,), id="block-ge-n"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_online_softmax_matches_full_attention(n_heads, copy_first_head, key_rows, stored_rows, dtype):
    expected, got, streamed = _attention_pair(
        n_heads=n_heads,
        copy_first_head=copy_first_head,
        dtype=dtype,
        key_rows=key_rows,
        stored_rows=stored_rows,
    )
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(got, expected, rtol=tolerance, atol=tolerance)
    assert streamed.max_staged_rows == max(stored_rows)


def test_int8_online_attention_uses_sliced_dequantization():
    torch.manual_seed(4)
    attn = MultiheadAttention(
        embed_dim=8,
        num_heads=2,
        qkv_combined=False,
        dropout=0.0,
    ).eval()
    x_kv = torch.randn(1, 1, 9, 8)
    x_q = torch.randn(1, 1, 3, 8)
    projected = attn.project_kv_cache(x_kv)["kv"]
    full = ScalableSeqKV(projected, 1, 1, quantize=True, offload=True, device=torch.device("cpu"))
    streamed = BlockwiseSeqKV.from_row_slices(
        _row_slices(projected, (4, 3, 2)),
        1,
        1,
        quantize=True,
        device=torch.device("cpu"),
        dtype=projected.dtype,
    )
    with torch.inference_mode():
        expected = attn.forward_with_kv_cache(x_q, full)[0]
        got = attn.forward_with_kv_cache(x_q, streamed)[0]
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)
    assert streamed.max_staged_rows == 4


def test_score_workspace_reslices_stored_blocks(monkeypatch):
    torch.manual_seed(5)
    attn = MultiheadAttention(
        embed_dim=8,
        num_heads=2,
        qkv_combined=False,
        dropout=0.0,
    ).eval()
    x_kv = torch.randn(1, 2, 11, 8)
    x_q = torch.randn(1, 2, 5, 8)
    full = attn.project_kv_cache(x_kv)
    streamed = BlockwiseSeqKV.from_row_slices(
        _row_slices(full["kv"], (11,)),
        1,
        2,
        quantize=False,
        device=torch.device("cpu"),
        dtype=x_kv.dtype,
    )
    score_bytes_per_key_row = 2 * 2 * 5 * 4
    monkeypatch.setattr(
        layer_module,
        "BLOCKWISE_SCORE_WORKSPACE_BYTES",
        2 * score_bytes_per_key_row,
    )
    with torch.inference_mode():
        expected = attn.forward_with_kv_cache(x_q, full)[0]
        got = attn.forward_with_kv_cache(x_q, streamed)[0]
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)
    assert streamed.max_staged_rows == 2


def test_blockwise_cache_device_mismatch_is_actionable():
    attn = MultiheadAttention(
        embed_dim=8,
        num_heads=2,
        qkv_combined=False,
        dropout=0.0,
    ).eval()
    x_kv = torch.randn(1, 1, 3, 8)
    streamed = BlockwiseSeqKV.from_row_slices(
        _row_slices(attn.project_kv_cache(x_kv)["kv"], (3,)),
        1,
        1,
        quantize=False,
        device=torch.device("cpu"),
        dtype=x_kv.dtype,
    )
    streamed.device = torch.device("meta")
    with pytest.raises(RuntimeError, match="cache device does not match"):
        attn.forward_with_kv_cache(torch.randn(1, 1, 1, 8), streamed)


@pytest.mark.parametrize("kv_heads", [1, 4])
def test_flex_block_lse_merge_matches_one_full_softmax(monkeypatch, kv_heads):
    torch.manual_seed(6)
    n_heads = 4
    head_dim = 8
    q = torch.randn(2, 7, n_heads, head_dim)
    kv = torch.randn(2, 13, 2, kv_heads, head_dim)
    cache = BlockwiseSeqKV.from_row_slices(
        _row_slices(kv, (4, 6, 3)),
        1,
        2,
        quantize=False,
        device=torch.device("cpu"),
        dtype=kv.dtype,
    )
    calls = []

    def fake_flex(q_block, k_block, v_block, *, scale, enable_gqa):
        calls.append((k_block.shape[2], enable_gqa))
        if enable_gqa:
            repeats = q_block.shape[1] // k_block.shape[1]
            k_block = k_block.repeat_interleave(repeats, dim=1)
            v_block = v_block.repeat_interleave(repeats, dim=1)
        logits = torch.einsum("bhqd,bhkd->bhqk", q_block.float(), k_block.float()) * scale
        lse = torch.logsumexp(logits, dim=-1)
        output = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(logits, dim=-1), v_block.float())
        return output, lse

    monkeypatch.setattr(layer_module, "_flex_attention_with_lse", fake_flex)
    attn = MultiheadAttention(
        embed_dim=n_heads * head_dim,
        num_heads=n_heads,
        qkv_combined=False,
        dropout=0.0,
    ).eval()
    with torch.inference_mode():
        got = attn._compute_attention_blockwise_flex(q, cache)

    q_full = q.to(torch.bfloat16).permute(0, 2, 1, 3)
    k_full, v_full = kv.to(torch.bfloat16).unbind(dim=2)
    k_full = k_full.permute(0, 2, 1, 3)
    v_full = v_full.permute(0, 2, 1, 3)
    expected, _ = fake_flex(
        q_full,
        k_full,
        v_full,
        scale=head_dim**-0.5,
        enable_gqa=kv_heads != n_heads,
    )
    expected = expected.permute(0, 2, 1, 3)
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)
    assert calls[:-1] == [
        (4, kv_heads != n_heads),
        (6, kv_heads != n_heads),
        (3, kv_heads != n_heads),
    ]
    assert cache.max_staged_rows == 6


@pytest.mark.parametrize(("head_dim", "flex_head_dim"), [(8, 8), (44, 64), (56, 64)])
def test_flex_blockwise_pads_non_power_of_two_head_widths(monkeypatch, head_dim, flex_head_dim):
    torch.manual_seed(7)
    n_heads = 4
    q = torch.randn(1, 5, n_heads, head_dim)
    kv = torch.randn(1, 9, 2, n_heads, head_dim)
    cache = BlockwiseSeqKV.from_row_slices(
        _row_slices(kv, (4, 5)),
        1,
        2,
        quantize=False,
        device=torch.device("cpu"),
        dtype=kv.dtype,
    )
    calls = []

    def fake_flex(q_block, k_block, v_block, *, scale, enable_gqa):
        calls.append((q_block.shape[-1], k_block.shape[-1], v_block.shape[-1], scale, enable_gqa))
        logits = torch.einsum("bhqd,bhkd->bhqk", q_block.float(), k_block.float()) * scale
        lse = torch.logsumexp(logits, dim=-1)
        output = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(logits, dim=-1), v_block.float())
        return output, lse

    monkeypatch.setattr(layer_module, "_flex_attention_with_lse", fake_flex)
    attn = MultiheadAttention(
        embed_dim=n_heads * head_dim,
        num_heads=n_heads,
        qkv_combined=False,
        dropout=0.0,
    ).eval()
    with torch.inference_mode():
        got = attn._compute_attention_blockwise_flex(q, cache)

    q_full = q.to(torch.bfloat16).permute(0, 2, 1, 3)
    k_full, v_full = kv.to(torch.bfloat16).unbind(dim=2)
    logits = torch.einsum("bhqd,bhkd->bhqk", q_full.float(), k_full.permute(0, 2, 1, 3).float()) * head_dim**-0.5
    expected = torch.einsum(
        "bhqk,bhkd->bhqd",
        torch.softmax(logits, dim=-1),
        v_full.permute(0, 2, 1, 3).float(),
    ).permute(0, 2, 1, 3)

    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)
    assert [(q_dim, k_dim, v_dim, enable_gqa) for q_dim, k_dim, v_dim, _, enable_gqa in calls] == [
        (flex_head_dim, flex_head_dim, flex_head_dim, False),
        (flex_head_dim, flex_head_dim, flex_head_dim, False),
    ]
    assert [scale for _, _, _, scale, _ in calls] == pytest.approx([head_dim**-0.5] * 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kv_heads", [1, 4])
def test_cuda_flex_blockwise_matches_full_sdpa(kv_heads):
    torch.manual_seed(7)
    device = torch.device("cuda", torch.cuda.current_device())
    n_heads = 4
    head_dim = 16
    q = torch.randn(2, 137, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(2, 385, 2, kv_heads, head_dim, dtype=torch.bfloat16)
    cache = BlockwiseSeqKV.from_row_slices(
        _row_slices(kv, (128, 128, 129)),
        1,
        2,
        quantize=False,
        device=device,
        dtype=kv.dtype,
    )
    k_full, v_full = kv.to(device).unbind(dim=2)
    with torch.inference_mode():
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k_full.permute(0, 2, 1, 3),
            v_full.permute(0, 2, 1, 3),
            enable_gqa=kv_heads != n_heads,
        ).permute(0, 2, 1, 3)
        got = (
            MultiheadAttention(
                embed_dim=n_heads * head_dim,
                num_heads=n_heads,
                qkv_combined=False,
                dropout=0.0,
                device=device,
                dtype=torch.bfloat16,
            )
            .eval()
            .compute_attention_blockwise(q, cache)
        )
    torch.testing.assert_close(got.float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert cache.max_staged_rows == 129


@pytest.mark.parametrize("uses_aux_api", [False, True])
def test_flex_lse_adapter_supports_both_torch_apis(monkeypatch, uses_aux_api):
    q = torch.zeros(1, 2, 3, 4)
    k = torch.zeros(1, 2, 5, 4)
    v = torch.zeros_like(k)
    expected_output = torch.ones_like(q)
    expected_lse = torch.full(q.shape[:-1], 2.0)
    request = object() if uses_aux_api else None
    seen_kwargs = {}

    class FakeAux:
        lse = expected_lse

    def fake_compiled(*args, **kwargs):
        seen_kwargs.update(kwargs)
        if uses_aux_api:
            return expected_output, FakeAux()
        return expected_output, expected_lse

    monkeypatch.setattr(layer_module, "_FLEX_LSE_REQUEST", request)
    monkeypatch.setattr(layer_module, "_compiled_flex_attention", lambda: fake_compiled)
    output, lse = layer_module._flex_attention_with_lse(
        q,
        k,
        v,
        scale=0.5,
        enable_gqa=True,
    )
    assert output is expected_output
    assert lse is expected_lse
    assert seen_kwargs["scale"] == 0.5
    assert seen_kwargs["enable_gqa"] is True
    if uses_aux_api:
        assert seen_kwargs["return_aux"] is request
        assert "return_lse" not in seen_kwargs
    else:
        assert seen_kwargs["return_lse"] is True
        assert "return_aux" not in seen_kwargs
