"""WS1 (memory engineering): make Nori's cached-regression KV cache RAM-efficient
so dense/isab/delta serving scales to far more context rows on a fixed GPU.

The train-side sequence-attention K/V cache is O(N_train) — one K/V per training
row per layer — and is the resident-memory bottleneck at large row counts. This
module lets that cache be:
  * int8-quantized  (per-(row,head) absmax scale -> ~1.9x smaller than bf16/fp16,
    ~3.8x than fp32: one byte per element plus one fp32 scale per head_dim
    vector, which is ~+6% at head_dim 64 — count it or the resident budget
    reads optimistic), and/or
  * host-offloaded  (kept in CPU RAM; streamed to the GPU on demand),
dequantizing + staging to the compute device lazily when the attention code reads
``cache["kv"]`` — so it is transparent to ``MultiheadAttention.forward_with_kv_cache``.

This mirrors TabPFN-3's serving recipe (int8 KV + host offload + chunked prefill):
GPU high-water-mark stays ~flat in N (weights + per-chunk activations + a staged
slice), while the O(N) cache lives in host RAM. Compute is unchanged; this is a
memory lever only.
"""
from __future__ import annotations

import torch


def quantize_int8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-vector (last-dim) absmax int8 quantization. Returns (int8, scale).

    scale has t's shape with the last dim = 1; dequant is q.float() * scale.
    Per-(…, head_dim-vector) scaling keeps error low without a big scale tensor.
    """
    scale = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (q.to(torch.float32) * scale).to(dtype)


class ScalableSeqKV:
    """Dict-like drop-in for the ``{"kv","batch","groups"}`` seq-attn cache.

    Stores ``kv`` int8-quantized and/or on host RAM; ``self["kv"]`` returns the
    full-precision tensor on ``device`` (dequantized + moved), computed lazily so
    the GPU copy is transient. ``self["batch"]``/``self["groups"]`` are plain ints.
    """

    def __init__(self, kv: torch.Tensor, batch: int, groups: int, *,
                 quantize: bool = False, offload: bool = False,
                 device: torch.device | None = None):
        self.batch = int(batch)
        self.groups = int(groups)
        self.device = torch.device(device) if device is not None else kv.device
        self.dtype = kv.dtype
        self.quantized = quantize
        self.offloaded = offload
        store_dev = torch.device("cpu") if offload else self.device
        if quantize:
            q, scale = quantize_int8(kv)
            self._q = q.to(store_dev, non_blocking=False)
            self._scale = scale.to(store_dev, non_blocking=False)
            self._kv = None
        else:
            self._q = self._scale = None
            self._kv = kv.to(store_dev, non_blocking=False)

    def __getitem__(self, key: str):
        if key == "batch":
            return self.batch
        if key == "groups":
            return self.groups
        if key == "kv":
            if self.quantized:
                q = self._q.to(self.device, non_blocking=True) if self.offloaded else self._q
                scale = self._scale.to(self.device, non_blocking=True) if self.offloaded else self._scale
                return dequantize_int8(q, scale, self.dtype)
            kv = self._kv
            return kv.to(self.device, non_blocking=True) if self.offloaded else kv
        raise KeyError(key)

    @classmethod
    def from_row_slices(cls, slices, batch: int, groups: int, *,
                        quantize: bool = False, offload: bool = False,
                        device: torch.device | None = None,
                        dtype: torch.dtype | None = None) -> "ScalableSeqKV":
        """Build from row-axis slices, folding each one in as it arrives.

        ``__init__`` takes the finished tensor, which means the fit-time row-chunked
        build (``layer.py``) would have to assemble it at full precision first — so at
        the ``torch.cat`` the slice list *and* its copy are both resident: two
        full-precision copies of the whole cache, on the one code path entered
        *because* memory was already tight. This consumes an iterator instead and
        quantizes/moves each slice immediately, so full precision never exists beyond
        one slice.

        Safe because quantization is per-(row, head) absmax over the LAST axis: a
        row's scale depends on that row alone, so quantizing slices and concatenating
        the int8 payloads is bit-identical to quantizing the concatenation.
        ``test_kv_cache_scaling.py`` pins that equivalence, so if quantization ever
        became per-tensor (as TabPFN-3 does it) the test fails rather than this build
        silently changing meaning.

        Args:
            slices: iterable of ``kv`` tensors, row axis at dim 1, in row order.
            batch: batch size the cache was built for.
            groups: feature-group count the cache was built for.
            quantize: store int8 rather than full precision.
            offload: store on host RAM rather than the compute device.
            device: compute device reads are staged back to.
            dtype: dtype to dequantize back to on read; inferred from the first slice
                when omitted.

        Returns:
            The assembled cache.

        Raises:
            ValueError: if ``slices`` is empty — a cache with no rows is a bug
                upstream, not something to represent.
        """
        store = torch.device("cpu") if offload else None
        q_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        kv_parts: list[torch.Tensor] = []
        for kv_slice in slices:
            if dtype is None:
                dtype = kv_slice.dtype
            if device is None:
                device = kv_slice.device
            if quantize:
                q, scale = quantize_int8(kv_slice)
                q_parts.append(q if store is None else q.to(store))
                scale_parts.append(scale if store is None else scale.to(store))
            else:
                kv_parts.append(kv_slice if store is None else kv_slice.to(store))
        if not (q_parts or kv_parts):
            raise ValueError("from_row_slices got no slices; nothing to cache")

        obj = cls.__new__(cls)
        obj.batch = int(batch)
        obj.groups = int(groups)
        obj.device = device if device is not None else torch.device("cpu")
        obj.dtype = dtype
        obj.quantized = bool(quantize)
        obj.offloaded = bool(offload)
        if quantize:
            obj._q = torch.cat(q_parts, dim=1)
            obj._scale = torch.cat(scale_parts, dim=1)
            obj._kv = None
        else:
            obj._q = obj._scale = None
            obj._kv = torch.cat(kv_parts, dim=1)
        return obj


    def resident_gpu_bytes(self) -> int:
        """Bytes this cache holds resident on the GPU (0 if offloaded)."""
        if self.offloaded:
            return 0
        if self.quantized:
            return self._q.numel() + self._scale.numel() * 4
        return self._kv.numel() * self._kv.element_size()


def scale_caches(caches: list[dict], *, quantize: bool = False, offload: bool = False,
                 device: torch.device | None = None) -> list[dict]:
    """Wrap each layer cache's ``seq_kv`` in a ScalableSeqKV (in place). No-op
    when quantize=offload=False. Returns the same list for convenience."""
    if not (quantize or offload):
        return caches
    for layer_cache in caches:
        skv = layer_cache.get("seq_kv")
        if skv is None or isinstance(skv, ScalableSeqKV):
            continue
        if "kv" not in skv:   # delta/isab cache is the O(M) landmark memory, already tiny
            continue
        layer_cache["seq_kv"] = ScalableSeqKV(
            skv["kv"], skv["batch"], skv["groups"],
            quantize=quantize, offload=offload, device=device,
        )
    return caches
