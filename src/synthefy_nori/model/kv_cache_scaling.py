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

``ScalableSeqKV`` implements the legacy int8/offload recipe and may still stage a
full layer during decode. ``BlockwiseSeqKV`` is the fail-closed representation
for bounded prefill: full-N activations and self-attention K/V blocks stay in host RAM, and online-softmax
attention stages only a bounded row slice on the compute device. Explicit context
streaming also keeps the reusable cross-cache on the host; the private hybrid
prefill keeps that cache resident in directly preallocated INT8 storage.

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

    def __init__(
        self,
        kv: torch.Tensor,
        batch: int,
        groups: int,
        *,
        quantize: bool = False,
        offload: bool = False,
        device: torch.device | None = None,
    ):
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
    def from_row_slices(
        cls,
        slices,
        batch: int,
        groups: int,
        *,
        quantize: bool = False,
        offload: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ScalableSeqKV":
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


class BlockwiseSeqKV:
    """Blockwise K/V storage for genuinely bounded-device attention.

    ``ScalableSeqKV`` lowers *resident* GPU memory, but ``cache["kv"]`` still
    reconstructs one complete layer on the compute device before SDPA runs. This
    representation has no full-tensor accessor: consumers must iterate bounded
    blocks through :meth:`iter_kv_blocks`.

    Explicit context streaming stores the blocks on the host. The hybrid resident
    INT8 prefill stores one directly preallocated quantized payload on the compute
    device and exposes disjoint row views of it. Both avoid a full-layer BF16/FP32
    reconstruction, and an accidental ``cache["kv"]`` read fails loudly.
    """

    def __init__(
        self,
        blocks: list[tuple[torch.Tensor, torch.Tensor | None]],
        batch: int,
        groups: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        quantized: bool,
        offloaded: bool = True,
    ):
        if not blocks:
            raise ValueError("BlockwiseSeqKV requires at least one K/V block")
        self.batch = int(batch)
        self.groups = int(groups)
        self.device = torch.device(device)
        self.dtype = dtype
        self.quantized = bool(quantized)
        self.offloaded = bool(offloaded)
        self._blocks = blocks
        self.n_rows = sum(int(payload.shape[1]) for payload, _ in blocks)
        self.max_block_rows = max(int(payload.shape[1]) for payload, _ in blocks)
        # Diagnostic used by reports/tests. It is reset at the start of each scan.
        self.max_staged_rows = 0

    def __getitem__(self, key: str):
        if key == "batch":
            return self.batch
        if key == "groups":
            return self.groups
        if key == "kv":
            raise RuntimeError(
                "BlockwiseSeqKV cannot materialize a full layer on the compute "
                "device; consume iter_kv_blocks() with online-softmax attention"
            )
        raise KeyError(key)

    @classmethod
    def from_row_slices(
        cls,
        slices,
        batch: int,
        groups: int,
        *,
        quantize: bool,
        device: torch.device,
        dtype: torch.dtype | None = None,
        offload: bool = True,
        n_rows: int | None = None,
    ) -> "BlockwiseSeqKV":
        """Consume projected row slices without a concatenation peak.

        Host-offloaded caches retain separate CPU blocks. Resident INT8 caches
        preallocate their final payload and scale tensors once, then copy each
        quantized row slice directly into its destination. The exposed blocks are
        disjoint views of that storage, so online attention never reconstructs a
        full BF16 or FP32 layer.
        """
        if not offload:
            if not quantize:
                raise ValueError("resident blockwise K/V requires quantize=True")
            if n_rows is None or n_rows <= 0:
                raise ValueError("resident blockwise K/V requires a positive n_rows")
            iterator = iter(slices)
            try:
                first_slice = next(iterator)
            except StopIteration as exc:
                raise ValueError("from_row_slices got no slices; nothing to cache") from exc
            if dtype is None:
                dtype = first_slice.dtype
            target_device = torch.device(device)
            first_payload, first_scale = quantize_int8(first_slice)
            payload_shape = (*first_payload.shape[:1], n_rows, *first_payload.shape[2:])
            scale_shape = (*first_scale.shape[:1], n_rows, *first_scale.shape[2:])
            payload_storage = torch.empty(payload_shape, dtype=first_payload.dtype, device=target_device)
            scale_storage = torch.empty(scale_shape, dtype=first_scale.dtype, device=target_device)
            blocks: list[tuple[torch.Tensor, torch.Tensor | None]] = []
            offset = 0

            def store(payload: torch.Tensor, scale: torch.Tensor) -> None:
                nonlocal offset
                width = int(payload.shape[1])
                end = offset + width
                if end > n_rows:
                    raise ValueError(f"row slices exceed declared n_rows={n_rows}")
                payload_view = payload_storage[:, offset:end]
                scale_view = scale_storage[:, offset:end]
                payload_view.copy_(payload)
                scale_view.copy_(scale)
                blocks.append((payload_view, scale_view))
                offset = end

            store(first_payload, first_scale)
            del first_slice, first_payload, first_scale
            for kv_slice in iterator:
                payload, scale = quantize_int8(kv_slice)
                store(payload, scale)
                del kv_slice, payload, scale
            if offset != n_rows:
                raise ValueError(f"row slices covered {offset} rows, expected n_rows={n_rows}")
            return cls(
                blocks,
                batch,
                groups,
                device=target_device,
                dtype=dtype,
                quantized=True,
                offloaded=False,
            )

        blocks: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for kv_slice in slices:
            if dtype is None:
                dtype = kv_slice.dtype
            if quantize:
                payload, scale = quantize_int8(kv_slice)
                host_payload = payload.detach().to("cpu")
                host_scale = scale.detach().to("cpu")
                blocks.append((host_payload, host_scale))
                del payload, scale, host_payload, host_scale
            else:
                host_payload = kv_slice.detach().to("cpu")
                blocks.append((host_payload, None))
                del host_payload
            del kv_slice
        if dtype is None:
            raise ValueError("from_row_slices got no slices; nothing to cache")
        return cls(
            blocks,
            batch,
            groups,
            device=device,
            dtype=dtype,
            quantized=quantize,
            offloaded=True,
        )

    def iter_kv_blocks(self, block_rows: int | None = None):
        """Yield full-precision K/V blocks on the compute device, in row order."""
        if block_rows is not None and block_rows <= 0:
            raise ValueError("block_rows must be positive")
        self.max_staged_rows = 0
        for payload, stored_scale in self._blocks:
            stored_rows = int(payload.shape[1])
            step = stored_rows if block_rows is None else min(block_rows, stored_rows)
            for start in range(0, stored_rows, step):
                end = min(start + step, stored_rows)
                # Slice while still on CPU. Transferring then slicing would recreate
                # the full-layer staging allocation this path promises not to make.
                staged_payload = payload[:, start:end].to(self.device, non_blocking=True)
                if self.quantized:
                    assert stored_scale is not None
                    staged_scale = stored_scale[:, start:end].to(self.device, non_blocking=True)
                    kv = dequantize_int8(staged_payload, staged_scale, self.dtype)
                else:
                    kv = staged_payload
                self.max_staged_rows = max(self.max_staged_rows, end - start)
                yield kv
                # A generator retains locals across ``yield``. Release the staged
                # block before the next host slice is transferred so peak staging is
                # one block, not the previous and current block together.
                del kv, staged_payload
                if self.quantized:
                    del staged_scale

    def resident_gpu_bytes(self) -> int:
        """Bytes retained on the compute device between iterator steps."""
        if self.offloaded:
            return 0
        total = 0
        for payload, scale in self._blocks:
            total += payload.numel() * payload.element_size()
            if scale is not None:
                total += scale.numel() * scale.element_size()
        return total

    def host_bytes(self) -> int:
        """Bytes retained by the host block list."""
        if not self.offloaded:
            return 0
        total = 0
        for payload, scale in self._blocks:
            total += payload.numel() * payload.element_size()
            if scale is not None:
                total += scale.numel() * scale.element_size()
        return total


def scale_caches(
    caches: list[dict], *, quantize: bool = False, offload: bool = False, device: torch.device | None = None
) -> list[dict]:
    """Wrap each layer cache's ``seq_kv`` in a ScalableSeqKV (in place). No-op
    when quantize=offload=False. Returns the same list for convenience."""
    if not (quantize or offload):
        return caches
    for layer_cache in caches:
        skv = layer_cache.get("seq_kv")
        if skv is None or isinstance(skv, (ScalableSeqKV, BlockwiseSeqKV)):
            continue
        if "kv" not in skv:  # delta/isab cache is the O(M) landmark memory, already tiny
            continue
        layer_cache["seq_kv"] = ScalableSeqKV(
            skv["kv"],
            skv["batch"],
            skv["groups"],
            quantize=quantize,
            offload=offload,
            device=device,
        )
    return caches
