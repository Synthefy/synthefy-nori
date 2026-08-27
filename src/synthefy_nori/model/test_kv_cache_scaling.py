"""Standalone tests for WS1 KV-cache scaling (torch-only, CPU-runnable)."""

import importlib.util, os, torch

_spec = importlib.util.spec_from_file_location(
    "kv_cache_scaling", os.path.join(os.path.dirname(__file__), "kv_cache_scaling.py")
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
ScalableSeqKV, scale_caches = _m.ScalableSeqKV, _m.scale_caches
quantize_int8, dequantize_int8 = _m.quantize_int8, _m.dequantize_int8


def test_quant_roundtrip():
    t = torch.randn(4, 100, 2, 3, 16)
    q, s = quantize_int8(t)
    assert q.dtype == torch.int8 and s.shape[-1] == 1
    rel = (dequantize_int8(q, s, t.dtype) - t).abs().max() / t.abs().max()
    assert rel < 0.02, f"int8 rel err too high: {rel}"


def test_identity_when_off():
    kv = torch.randn(2, 50, 2, 4, 8)
    c = ScalableSeqKV(kv, 2, 5, quantize=False, offload=False, device=torch.device("cpu"))
    assert c["batch"] == 2 and c["groups"] == 5
    assert torch.equal(c["kv"], kv)


def test_quantized_close():
    kv = torch.randn(2, 50, 2, 4, 8)
    c = ScalableSeqKV(kv, 2, 5, quantize=True, offload=False, device=torch.device("cpu"))
    rel = (c["kv"] - kv).abs().max() / kv.abs().max()
    assert rel < 0.02, rel


def test_offload_transparent():
    kv = torch.randn(2, 50, 2, 4, 8)
    c = ScalableSeqKV(kv, 2, 5, quantize=False, offload=True, device=torch.device("cpu"))
    assert torch.equal(c["kv"], kv) and c.resident_gpu_bytes() == 0


def test_scale_caches_inplace_and_noop():
    mk = lambda: [{"seq_kv": {"kv": torch.randn(1, 10, 2, 2, 4), "batch": 1, "groups": 1}}]
    c0 = mk()
    assert scale_caches(c0, quantize=False, offload=False) is c0
    assert isinstance(c0[0]["seq_kv"], dict)  # untouched
    c1 = mk()
    scale_caches(c1, quantize=True, offload=True, device=torch.device("cpu"))
    assert isinstance(c1[0]["seq_kv"], ScalableSeqKV) and c1[0]["seq_kv"]["kv"].shape == (1, 10, 2, 2, 4)


def _chunks(kv, size):
    """Split kv along the row axis (dim 1) the way the fit-time build does."""
    return [kv[:, i : i + size] for i in range(0, kv.shape[1], size)]


def test_rowchunked_int8_is_bit_identical_to_quantizing_the_whole_tensor():
    """Quantizing per row-slice must equal quantizing after the concat.

    This is what licenses the memory win: the builder never holds a
    full-precision copy of the whole cache, and it is allowed to do that only
    because the absmax scale is taken over the LAST axis (head_dim), so a row's
    scale depends on that row alone. If quantization ever became global -- or
    per-tensor, as TabPFN-3 does it -- this test fails and the row-chunked build
    would have to change with it.
    """
    kv = torch.randn(3, 257, 2, 4, 16)  # 257 rows: deliberately not a multiple
    whole_q, whole_scale = quantize_int8(kv)
    built = ScalableSeqKV.from_row_slices(
        iter(_chunks(kv, 64)), 3, 4, quantize=True, offload=False, device=torch.device("cpu"), dtype=kv.dtype
    )
    assert torch.equal(built._q, whole_q), "int8 payload differs from whole-tensor quant"
    assert torch.equal(built._scale, whole_scale), "scales differ from whole-tensor quant"
    # ...and therefore the dequantized read is identical too.
    assert torch.equal(built["kv"], dequantize_int8(whole_q, whole_scale, kv.dtype))


def test_rowchunked_bf16_reassembles_exactly():
    kv = torch.randn(2, 100, 2, 3, 8)
    out = ScalableSeqKV.from_row_slices(
        iter(_chunks(kv, 32)), 2, 3, quantize=False, offload=False, device=torch.device("cpu"), dtype=kv.dtype
    )
    assert torch.equal(out["kv"], kv)


def test_rowchunked_offload_keeps_nothing_resident():
    kv = torch.randn(2, 100, 2, 3, 8)
    out = ScalableSeqKV.from_row_slices(
        iter(_chunks(kv, 32)), 2, 3, quantize=True, offload=True, device=torch.device("cpu"), dtype=kv.dtype
    )
    assert out.resident_gpu_bytes() == 0
    assert out["kv"].shape == kv.shape


def test_rowchunked_matches_the_unchunked_scalable_cache():
    """One slice == no chunking: the builder and ScalableSeqKV must agree."""
    kv = torch.randn(2, 64, 2, 3, 8)
    direct = ScalableSeqKV(kv, 2, 3, quantize=True, offload=False, device=torch.device("cpu"))
    built = ScalableSeqKV.from_row_slices(
        iter([kv]), 2, 3, quantize=True, offload=False, device=torch.device("cpu"), dtype=kv.dtype
    )
    assert torch.equal(built["kv"], direct["kv"])


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("PASS", n)
    print("all kv-cache-scaling tests passed")


def test_from_row_slices_rejects_an_empty_iterator():
    # A cache with no rows is an upstream bug, not a state worth representing.
    try:
        ScalableSeqKV.from_row_slices(iter([]), 1, 1, quantize=True, device=torch.device("cpu"), dtype=torch.float32)
    except ValueError:
        return
    raise AssertionError("expected ValueError on an empty slice iterator")
