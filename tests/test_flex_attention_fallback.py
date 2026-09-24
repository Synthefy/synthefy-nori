"""Streamed attention falls back to plain PyTorch attention when FlexAttention cannot compile.

The SageMaker image ships without Triton, and Inductor needs Triton to generate the
fused CUDA kernel. These CPU tests pin the fallback contract: the fallback returns what
FlexAttention returns (output and natural-log LSE, grouped-query heads included) without
calling anything in ``torch.nn.attention.flex_attention``, a broken backend is detected
at most once per process, and fallback blocks are bounded by the score workspace.
"""

from __future__ import annotations

import logging
import types

import pytest
import torch
import torch._dynamo.exc as dynamo_exc
import torch._inductor.exc as inductor_exc
import torch.nn.attention.flex_attention as torch_flex_module

import synthefy_nori.model.layer as layer_module
from synthefy_nori.model.kv_cache_scaling import BlockwiseSeqKV
from synthefy_nori.model.layer import MultiheadAttention

_LAYER_LOGGER = "synthefy_nori.model.layer"
_REAL_FLEX_ATTENTION = torch_flex_module.flex_attention


@pytest.fixture(autouse=True)
def _fresh_fallback_state(monkeypatch):
    """Every test starts undecided and leaves no fallback decision behind for GPU tests."""
    compiled_flex_attention = layer_module._compiled_flex_attention
    compiled_flex_attention.cache_clear()
    monkeypatch.setattr(layer_module, "_STREAMED_FALLBACK_REASON", None)
    yield
    compiled_flex_attention.cache_clear()


def _forbid_flex_attention(monkeypatch) -> None:
    """Fail on any call into torch's FlexAttention, from this module or from torch's."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the fallback must not call torch.nn.attention.flex_attention")

    monkeypatch.setattr(layer_module, "flex_attention", forbidden)
    monkeypatch.setattr(torch_flex_module, "flex_attention", forbidden)


def _triton_installed(monkeypatch, installed: bool) -> None:
    real_find_spec = layer_module.importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "triton":
            return types.SimpleNamespace(name="triton") if installed else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(layer_module.importlib.util, "find_spec", find_spec)


def _qkv(n_heads=4, kv_heads=1, n_query=16, n_keys=32, head_dim=8, dtype=torch.float32):
    torch.manual_seed(3)
    q = torch.randn(2, n_heads, n_query, head_dim).to(dtype)
    k = torch.randn(2, kv_heads, n_keys, head_dim).to(dtype)
    v = torch.randn(2, kv_heads, n_keys, head_dim).to(dtype)
    return q, k, v


def _einsum_reference(q, k, v, *, scale):
    """Softmax attention written out, K/V heads repeated for grouped-query attention."""
    repeats = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(repeats, dim=1)
    v = v.float().repeat_interleave(repeats, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k) * scale
    return torch.einsum("bhqk,bhkd->bhqd", torch.softmax(scores, dim=-1), v), torch.logsumexp(scores, dim=-1)


def _flex_reference(q, k, v, *, scale, enable_gqa):
    """Eager ``flex_attention`` on CPU, or a skip where this torch cannot run it."""
    kwargs = {"scale": scale, "enable_gqa": enable_gqa}
    try:
        if layer_module._FLEX_LSE_REQUEST is not None:
            out, aux = _REAL_FLEX_ATTENTION(q, k, v, return_aux=layer_module._FLEX_LSE_REQUEST, **kwargs)
            return out, aux.lse
        return _REAL_FLEX_ATTENTION(q, k, v, return_lse=True, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on the installed torch
        pytest.skip(f"eager flex_attention is unavailable on this torch/CPU: {type(exc).__name__}: {exc}")


def _warnings(caplog):
    return [r for r in caplog.records if r.name == _LAYER_LOGGER and r.levelno == logging.WARNING]


_HEAD_LAYOUTS = pytest.mark.parametrize(
    ("n_heads", "kv_heads"),
    [(4, 4), (4, 1), (8, 2)],
    ids=["mha", "mqa", "gqa-8-2"],
)


@_HEAD_LAYOUTS
def test_fallback_matches_eager_flex_attention(monkeypatch, n_heads, kv_heads):
    q, k, v = _qkv(n_heads=n_heads, kv_heads=kv_heads)
    enable_gqa = n_heads != kv_heads
    expected_out, expected_lse = _flex_reference(q, k, v, scale=0.25, enable_gqa=enable_gqa)
    _forbid_flex_attention(monkeypatch)

    out, lse = layer_module._math_attention_with_lse(q, k, v, scale=0.25, enable_gqa=enable_gqa)

    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse, expected_lse.float(), rtol=1e-5, atol=1e-5)


@_HEAD_LAYOUTS
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fallback_matches_softmax_attention(n_heads, kv_heads, dtype):
    q, k, v = _qkv(n_heads=n_heads, kv_heads=kv_heads, dtype=dtype)

    out, lse = layer_module._math_attention_with_lse(q, k, v, scale=0.3, enable_gqa=n_heads != kv_heads)

    expected_out, expected_lse = _einsum_reference(q, k, v, scale=0.3)
    assert out.dtype == dtype
    assert lse.dtype == torch.float32
    assert lse.shape == q.shape[:-1]
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out.float(), expected_out, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)


def test_fallback_rejects_mismatched_heads_without_gqa():
    q, k, v = _qkv(n_heads=4, kv_heads=2)
    with pytest.raises(ValueError, match="cannot attend"):
        layer_module._math_attention_with_lse(q, k, v, scale=0.3, enable_gqa=False)


def test_split_blocks_merge_to_the_whole_by_lse():
    q, k, v = _qkv(n_heads=8, kv_heads=2, n_keys=40)
    whole_out, whole_lse = layer_module._math_attention_with_lse(q, k, v, scale=0.3, enable_gqa=True)
    first = layer_module._math_attention_with_lse(q, k[:, :, :17], v[:, :, :17], scale=0.3, enable_gqa=True)
    second = layer_module._math_attention_with_lse(q, k[:, :, 17:], v[:, :, 17:], scale=0.3, enable_gqa=True)

    merged_lse = torch.logaddexp(first[1], second[1])
    merged_out = sum(out * torch.exp(lse - merged_lse).unsqueeze(-1) for out, lse in (first, second))

    torch.testing.assert_close(merged_lse, whole_lse, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(merged_out, whole_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("kv_heads", "head_dim"), [(4, 8), (1, 8), (2, 44)])
def test_streamed_merge_in_fallback_mode_matches_one_softmax(monkeypatch, kv_heads, head_dim):
    """The streaming merge over fallback blocks equals one softmax over every context row."""
    torch.manual_seed(5)
    n_heads, n_query, context_rows = 4, 3, 20
    q = torch.randn(1, n_query, n_heads, head_dim)
    kv = torch.randn(1, context_rows, 2, kv_heads, head_dim)
    cache = BlockwiseSeqKV.from_row_slices(
        iter([kv]),
        1,
        2,
        quantize=False,
        device=torch.device("cpu"),
        dtype=kv.dtype,
    )
    # A 7-row score workspace forces three blocks through the real fallback.
    monkeypatch.setattr(
        layer_module,
        "BLOCKWISE_SCORE_WORKSPACE_BYTES",
        7 * n_heads * n_query * layer_module.FP32_ELEMENT_BYTES,
    )
    _triton_installed(monkeypatch, installed=False)
    _forbid_flex_attention(monkeypatch)
    attn = MultiheadAttention(embed_dim=n_heads * head_dim, num_heads=n_heads, qkv_combined=False, dropout=0.0).eval()
    with torch.inference_mode():
        got = attn._compute_attention_blockwise_flex(q, cache)

    # The streaming path runs its blocks in bf16; build the reference from the same inputs.
    q_ref = q.to(layer_module.FLASH_ATTN_DTYPE).permute(0, 2, 1, 3)
    k_ref, v_ref = (t.to(layer_module.FLASH_ATTN_DTYPE).permute(0, 2, 1, 3) for t in kv.unbind(dim=2))
    expected, _ = _einsum_reference(q_ref, k_ref, v_ref, scale=head_dim**-0.5)
    assert cache.max_staged_rows == 7
    torch.testing.assert_close(got, expected.permute(0, 2, 1, 3), rtol=1e-2, atol=1e-2)


def test_missing_triton_uses_the_fallback_without_compiling(monkeypatch, caplog):
    _triton_installed(monkeypatch, installed=False)

    def forbid_compile():
        raise AssertionError("FlexAttention must not be compiled without triton")

    monkeypatch.setattr(layer_module, "_compiled_flex_attention", forbid_compile)
    _forbid_flex_attention(monkeypatch)
    q, k, v = _qkv()
    with caplog.at_level(logging.WARNING, logger=_LAYER_LOGGER):
        out, lse = layer_module._flex_attention_with_lse(q, k, v, scale=0.25, enable_gqa=True)
        layer_module._flex_attention_with_lse(q, k, v, scale=0.25, enable_gqa=True)

    expected_out, expected_lse = _einsum_reference(q, k, v, scale=0.25)
    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "triton is not installed" in warnings[0].getMessage()
    assert "plain PyTorch fallback" in warnings[0].getMessage()


def _inductor_error():
    return inductor_exc.InductorError(ModuleNotFoundError("No module named 'triton'"), None)


def _backend_compiler_failed():
    return dynamo_exc.BackendCompilerFailed(
        "inductor",
        ModuleNotFoundError("No module named 'triton'"),
        None,
    )


def _bare_missing_triton():
    return ModuleNotFoundError("No module named 'triton'")


@pytest.mark.parametrize("make_error", [_inductor_error, _backend_compiler_failed, _bare_missing_triton])
def test_compile_failure_on_first_call_falls_back_once(monkeypatch, caplog, make_error):
    _triton_installed(monkeypatch, installed=True)
    compiles = []
    compiled_calls = []

    def fake_compiled(*args, **kwargs):
        compiled_calls.append(kwargs)
        raise make_error()

    def fake_compile():
        compiles.append(1)
        return fake_compiled

    monkeypatch.setattr(layer_module, "_compiled_flex_attention", fake_compile)
    _forbid_flex_attention(monkeypatch)
    q, k, v = _qkv()
    with caplog.at_level(logging.WARNING, logger=_LAYER_LOGGER):
        first = layer_module._flex_attention_with_lse(q, k, v, scale=0.5, enable_gqa=True)
        second = layer_module._flex_attention_with_lse(q, k, v, scale=0.5, enable_gqa=True)

    expected_out, expected_lse = _einsum_reference(q, k, v, scale=0.5)
    for out, lse in (first, second):
        torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)
    assert len(compiles) == 1
    assert len(compiled_calls) == 1
    assert len(_warnings(caplog)) == 1
    assert layer_module._STREAMED_FALLBACK_REASON is not None


@pytest.mark.parametrize(
    "error",
    [RuntimeError("shape mismatch"), torch.OutOfMemoryError("CUDA out of memory")],
    ids=["runtime-error", "cuda-oom"],
)
def test_non_compile_errors_propagate_and_keep_compiling(monkeypatch, error):
    _triton_installed(monkeypatch, installed=True)

    def fake_compiled(*args, **kwargs):
        raise error

    monkeypatch.setattr(layer_module, "_compiled_flex_attention", lambda: fake_compiled)
    q, k, v = _qkv()
    with pytest.raises(type(error)):
        layer_module._flex_attention_with_lse(q, k, v, scale=0.5, enable_gqa=True)
    assert layer_module._STREAMED_FALLBACK_REASON is None


def test_triton_present_uses_the_compiled_kernel(monkeypatch):
    _triton_installed(monkeypatch, installed=True)
    expected_out, expected_lse = torch.zeros(1), torch.ones(1)

    def fake_compiled(*args, **kwargs):
        if "return_aux" in kwargs:
            return expected_out, types.SimpleNamespace(lse=expected_lse)
        return expected_out, expected_lse

    monkeypatch.setattr(layer_module, "_compiled_flex_attention", lambda: fake_compiled)
    q, k, v = _qkv()
    out, lse = layer_module._flex_attention_with_lse(q, k, v, scale=0.5, enable_gqa=True)
    assert out is expected_out
    assert lse is expected_lse
    assert layer_module._STREAMED_FALLBACK_REASON is None


@pytest.mark.parametrize("fallback", [True, False])
def test_fallback_blocks_are_bounded_by_the_score_workspace(monkeypatch, fallback):
    n_heads, head_dim, n_query, context_rows = 2, 8, 3, 20
    q = torch.randn(1, n_query, n_heads, head_dim)
    kv = torch.randn(1, context_rows, 2, n_heads, head_dim)
    cache = BlockwiseSeqKV.from_row_slices(
        iter([kv]),
        1,
        2,
        quantize=False,
        device=torch.device("cpu"),
        dtype=kv.dtype,
    )
    # One key row costs BG * H * Q FP32 scores; a workspace of 7 rows forces three blocks.
    monkeypatch.setattr(
        layer_module,
        "BLOCKWISE_SCORE_WORKSPACE_BYTES",
        7 * n_heads * n_query * layer_module.FP32_ELEMENT_BYTES,
    )
    monkeypatch.setattr(layer_module, "_streamed_attention_uses_fallback", lambda: fallback)
    block_widths = []

    def fake_block_attention(q_block, k_block, v_block, *, scale, enable_gqa):
        block_widths.append(k_block.shape[2])
        return layer_module._math_attention_with_lse(q_block, k_block, v_block, scale=scale, enable_gqa=enable_gqa)

    monkeypatch.setattr(layer_module, "_flex_attention_with_lse", fake_block_attention)
    attn = MultiheadAttention(embed_dim=n_heads * head_dim, num_heads=n_heads, qkv_combined=False, dropout=0.0).eval()
    with torch.inference_mode():
        attn._compute_attention_blockwise_flex(q, cache)
    assert block_widths == ([7, 7, 6] if fallback else [context_rows])


def test_lazy_compile_failure_bounds_the_first_and_remaining_math_blocks(monkeypatch):
    """The iterator is already open when lazy compilation first discovers a broken backend."""
    _triton_installed(monkeypatch, installed=True)
    torch.manual_seed(23)
    n_heads, head_dim, n_query, context_rows = 2, 8, 3, 20
    q = torch.randn(1, n_query, n_heads, head_dim)
    kv = torch.randn(1, context_rows, 2, 1, head_dim)
    cache = BlockwiseSeqKV.from_row_slices(
        iter([kv[:, :11], kv[:, 11:]]), 1, 2, quantize=False, device=torch.device("cpu"), dtype=kv.dtype
    )
    monkeypatch.setattr(
        layer_module, "BLOCKWISE_SCORE_WORKSPACE_BYTES", 7 * n_heads * n_query * layer_module.FP32_ELEMENT_BYTES
    )

    def broken_compiler(*args, **kwargs):
        raise _backend_compiler_failed()

    monkeypatch.setattr(layer_module, "_compiled_flex_attention", lambda: broken_compiler)
    math_attention = layer_module._math_attention_with_lse
    block_widths = []

    def record_math(q_block, k_block, v_block, *, scale, enable_gqa):
        block_widths.append(k_block.shape[2])
        return math_attention(q_block, k_block, v_block, scale=scale, enable_gqa=enable_gqa)

    monkeypatch.setattr(layer_module, "_math_attention_with_lse", record_math)
    attn = MultiheadAttention(embed_dim=n_heads * head_dim, num_heads=n_heads, qkv_combined=False, dropout=0.0).eval()
    with torch.inference_mode():
        actual = attn._compute_attention_blockwise_flex(q, cache)
    q_ref = q.to(layer_module.FLASH_ATTN_DTYPE).permute(0, 2, 1, 3)
    k_ref, v_ref = (t.to(layer_module.FLASH_ATTN_DTYPE).permute(0, 2, 1, 3) for t in kv.unbind(dim=2))
    expected, _ = _einsum_reference(q_ref, k_ref, v_ref, scale=head_dim**-0.5)
    assert block_widths == [7, 4, 7, 2]
    torch.testing.assert_close(actual, expected.permute(0, 2, 1, 3), rtol=1e-2, atol=5e-3)
