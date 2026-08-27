from __future__ import annotations

from contextlib import nullcontext
import functools
import inspect
from typing import Callable, Literal, Optional
import math
import os

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.attention.flex_attention as flex_attention_module
from torch.utils.checkpoint import checkpoint
from synthefy_nori.model.kv_cache_scaling import BlockwiseSeqKV, ScalableSeqKV, scale_caches
from functools import partial
from torch.amp import autocast

# Hard ceiling for the only block-local tensor whose width is Q x K: the FP32
# attention score/weight workspace. The row cap below derives K from the actual
# [batch*groups, heads, query_rows] shape, so a caller may request a larger
# context_row_chunk without making this tensor grow past 256 MiB.
BLOCKWISE_SCORE_WORKSPACE_BYTES = 256 * 1024**2
FP32_ELEMENT_BYTES = 4

flex_attention = flex_attention_module.flex_attention
_FLEX_SUPPORTS_RETURN_AUX = "return_aux" in inspect.signature(flex_attention).parameters
_FLEX_AUX_REQUEST = getattr(flex_attention_module, "AuxRequest", None)
if _FLEX_SUPPORTS_RETURN_AUX and _FLEX_AUX_REQUEST is None:
    raise RuntimeError("FlexAttention exposes return_aux without AuxRequest")
_FLEX_LSE_REQUEST = _FLEX_AUX_REQUEST(lse=True) if _FLEX_SUPPORTS_RETURN_AUX else None


@functools.lru_cache(maxsize=1)
def _compiled_flex_attention():
    """Compile FlexAttention once for streamed CUDA K/V blocks."""
    return torch.compile(flex_attention)


def _flex_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused exact attention and retain each block's softmax normalizer."""
    compiled = _compiled_flex_attention()
    if _FLEX_LSE_REQUEST is not None:
        out, aux = compiled(
            q,
            k,
            v,
            scale=scale,
            enable_gqa=enable_gqa,
            return_aux=_FLEX_LSE_REQUEST,
        )
        if aux.lse is None:
            raise RuntimeError("FlexAttention did not return the requested LSE")
        return out, aux.lse
    out, lse = compiled(
        q,
        k,
        v,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=True,
    )
    return out, lse


# Internal defines HAVE_FLASH_ATTN / HAVE_FLASH_ATTN_4 here and dispatches
# cached attention to flash-attn when present. This tier has no flash path, so
# the probes are omitted rather than left as unread constants. It also disables
# cuDNN SDPA process-wide at import; the per-call sdpa_kernel guard below does
# the same job without mutating global torch state, so that block is omitted too.

#: Working dtype for the FlexAttention blockwise kernel. Named for the flash
#: kernels it was introduced alongside; the blockwise path uses it whether or
#: not flash-attn is installed.
FLASH_ATTN_DTYPE = torch.bfloat16

from typing_extensions import override

Activation = Literal["gelu"]

_ALLOW_CUDNN_SDP = os.environ.get("SYNTHEFY_NORI_ALLOW_CUDNN_SDP", "0") == "1"
_SDPA_BACKENDS_WITHOUT_CUDNN = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

# PyTorch SDPA grids one CUDA dimension over batch/head pairs. Keep each
# launch within that dimension's portable limit; larger independent batches
# are split in ``compute_attention_by_torch``.
SDPA_BATCH_HEAD_LIMIT = 65_535

ACTIVATION_FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
}


class LayerNormMixedPrecision(nn.LayerNorm):
    """
    When the embedding dimension is below 512, use half precision for computation to improve performance.
    If the embedding dimension exceeds 512, it may cause training instability.
    """

    def forward(self, input: torch.Tensor):
        if input.dtype == torch.float16 and sum(self.normalized_shape) < 512:
            with autocast(device_type="cuda" if input.is_cuda else "cpu", enabled=False):
                return super().forward(input)
        else:
            return super().forward(input)


class RMSNorm(nn.Module):
    """RMSNorm — faster than LayerNorm, no mean computation."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=False, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # Training can opt into PyTorch's fused native implementation without
        # changing the checkpoint schema or default numerical path.
        self.use_native = False
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_native:
            return nn.functional.rms_norm(x, self.normalized_shape, weight=self.weight, eps=self.eps)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms
        if self.weight is not None:
            x = x * self.weight
        return x


class SwiGLUMLP(nn.Module):
    """SwiGLU gated FFN: (SiLU(xW_gate) * xW_up) @ W_down.
    Uses hidden=2/3 of original to match param count."""

    def __init__(self, in_features, hidden_size, out_features, has_bias, device, dtype):
        super().__init__()
        swiglu_hidden = int(2 * hidden_size / 3)
        self.w_gate = nn.Linear(in_features, swiglu_hidden, bias=has_bias, device=device, dtype=dtype)
        self.w_up = nn.Linear(in_features, swiglu_hidden, bias=has_bias, device=device, dtype=dtype)
        self.w_down = nn.Linear(swiglu_hidden, out_features, bias=has_bias, device=device, dtype=dtype)
        torch.nn.init.normal_(self.w_down.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(nn.functional.silu(self.w_gate(x)) * self.w_up(x))


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        has_bias: bool,
        device: torch.device | None,
        dtype: torch.dtype | None,
        activation: Activation = "gelu",
        depth: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.layers = []

        if depth == 1:
            self.layers.append(nn.Linear(in_features, out_features, bias=has_bias, device=device, dtype=dtype))
        else:
            # input layer
            self.layers.append(nn.Linear(in_features, hidden_size, bias=has_bias, device=device, dtype=dtype))
            self.layers.append(ACTIVATION_FN[self.activation])
            # hidden layers
            for i in range(depth - 2):
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=has_bias, device=device, dtype=dtype))
                self.layers.append(ACTIVATION_FN[self.activation])
            # output layer — small init for stable gradient flow in deep post-norm transformers
            self.layers.append(nn.Linear(hidden_size, out_features, bias=has_bias, device=device, dtype=dtype))
            torch.nn.init.normal_(self.layers[-1].weight, std=0.02)
        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class QASSMaxScaling(nn.Module):
    """Query-aware scalable softmax implemented as query rescaling.

    We keep the critical log(n) trend from SSMax as the default behavior and let
    two small MLPs learn head-/dimension-specific deviations around it:

      q_scaled = q * [log(n) * (1 + tanh(base(log n)))] * [1 + tanh(gate(q))]

    This keeps the scaling positive and initializes to SSMax-like behavior
    (base=0, gate=0 -> q_scaled = q * log(n)).
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        hidden_dim: int = 64,
        qass_mode: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # qass_mode selects which learned components are active:
        #   - "log_only":  q * log(n)                  (no learned components)
        #   - "base_only": q * base_scale              (no query gate)
        #   - "full":      q * base_scale * gate_scale
        # Honoring it is required for older "log_only" checkpoints that still
        # carry trained base/gate weights: running them "full" silently applies
        # the wrong attention temperature.
        #
        # Precedence: explicit argument > SYNTHEFY_QASS_MODE > "full".
        # `build_model` resolves the mode from the architecture config once and
        # passes it down explicitly, so a model's attention temperature is a
        # function of its own config and nothing else. The env var stays as a
        # deliberate experiment override for callers that construct this module
        # directly; it must NOT be able to override a config-resolved mode,
        # because that is process-global state deciding model behaviour.
        if qass_mode is None:
            qass_mode = os.environ.get("SYNTHEFY_QASS_MODE", "full")
        self.qass_mode = str(qass_mode).strip().lower()
        if self.qass_mode not in {"full", "base_only", "log_only"}:
            raise ValueError(
                "qass_mode (argument or SYNTHEFY_QASS_MODE) must be one of "
                f"full, base_only, log_only; got {self.qass_mode!r}"
            )

        # Build both historical components in their original order before
        # retaining only the ones this mode executes. Besides creating dead
        # checkpoint state, the old constructors consumed random numbers between
        # live attention/MLP initializations. Skipping those draws made a nominally
        # identical seed initialize almost every later live tensor differently.
        # Construct-and-discard keeps new state dicts slim while preserving the
        # established initialization stream exactly. This is deliberately safer
        # than manually advancing the RNG by a hard-coded number of draws, whose
        # backend/dtype behavior is not a stable contract.
        legacy_base_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * head_dim, device=device, dtype=dtype),
        )
        legacy_gate_mlp = nn.Sequential(
            nn.Linear(head_dim, hidden_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, head_dim, device=device, dtype=dtype),
        )
        self.base_mlp = legacy_base_mlp if self.qass_mode != "log_only" else None
        self.gate_mlp = legacy_gate_mlp if self.qass_mode == "full" else None

        # Start from SSMax-like behavior: base=0 -> log(n), gate=0 -> 1.
        if self.base_mlp is not None:
            nn.init.zeros_(self.base_mlp[-1].weight)
            nn.init.zeros_(self.base_mlp[-1].bias)
        if self.gate_mlp is not None:
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Ignore only legacy weights for components absent in this mode.

        Old log_only checkpoints contain frozen ``base_mlp`` and ``gate_mlp``
        tensors; old base_only checkpoints contain a frozen ``gate_mlp``. They
        were never read by those modes. Filtering exactly those prefixes keeps
        strict checkpoint loading backward compatible while preserving strict
        errors for every live architecture tensor.
        """
        ignored_prefixes = []
        if self.base_mlp is None:
            ignored_prefixes.append(f"{prefix}base_mlp.")
        if self.gate_mlp is None:
            ignored_prefixes.append(f"{prefix}gate_mlp.")
        if ignored_prefixes:
            state_dict = state_dict.copy()
            for key in tuple(state_dict):
                if key.startswith(tuple(ignored_prefixes)):
                    state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, q: torch.Tensor, key_len: int) -> torch.Tensor:
        if key_len <= 1:
            return q

        # Take the log in Python doubles and cast only the RESULT to q.dtype.
        # Materializing `key_len` itself in q.dtype first overflows fp16 — its
        # largest finite value is 65504, so any context of 65520+ rows becomes
        # `inf` before log() ever runs, and inf propagates through base_scale
        # into q, making the whole prediction non-finite. log(n) is ~11 for a
        # 65k-row context, which every supported dtype represents exactly
        # enough. Inference autocasts to fp16 on CUDA (the trainer uses bf16,
        # which has the range to hide this), so the overflow only ever showed
        # up at long-context inference. See issue #439.
        log_n = torch.tensor(math.log(float(max(key_len, 2))), device=q.device, dtype=q.dtype)
        if self.qass_mode == "log_only":
            return q * log_n.view(1, 1, 1, 1)
        assert self.base_mlp is not None
        base_delta = self.base_mlp(log_n.view(1, 1)).view(1, 1, self.num_heads, self.head_dim)
        base_scale = log_n.view(1, 1, 1, 1) * (1.0 + torch.tanh(base_delta))
        if self.qass_mode == "base_only":
            return q * base_scale
        assert self.gate_mlp is not None
        gate_scale = 1.0 + torch.tanh(self.gate_mlp(q))
        return q * base_scale * gate_scale


class MultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qkv_combined: bool = True,
        dropout: float = 0,
        recompute: bool = False,
        use_qassmax: bool = False,
        use_logn_attention: bool = False,
        use_learnable_attn_temperature: bool = False,
        attn_n_ref: float = 1024.0,
        # Appended, not inserted: this signature is not keyword-only, so adding a
        # parameter mid-list would silently shift any positional caller.
        qass_mode: Optional[str] = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_combined = qkv_combined
        self.dropout = dropout
        self.recompute = recompute
        self.device = device
        self.dtype = dtype
        self.use_qassmax = use_qassmax

        # logN attention scaling and learnable per-layer temperature.
        # The standard scale is 1/sqrt(head_dim). We pre-multiply Q by an
        # additional factor so SDPA's internal scaling produces the desired
        # final scale of (temperature * log(n_keys) / log(n_ref)) / sqrt(d).
        # When both flags are off, behaves identically to standard attention.
        self.use_logn_attention = use_logn_attention
        self.use_learnable_attn_temperature = use_learnable_attn_temperature
        self.attn_n_ref = float(attn_n_ref)
        self._attn_n_ref_log = math.log(max(self.attn_n_ref, 2.0))
        if use_learnable_attn_temperature:
            # Init at 1.0 so step-0 behavior matches standard attention.
            # Use abs() in forward to keep positive — gradient is well-defined
            # everywhere except 0, which is practically never visited.
            self.attn_temperature = nn.Parameter(torch.ones(1, device=device, dtype=dtype if dtype else torch.float32))
        else:
            self.attn_temperature = None

        self.out_proj_weight = torch.nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, self.embed_dim, device=self.device, dtype=self.dtype)
        )
        self.qkv_proj_weight = torch.nn.Parameter(
            torch.empty(3, self.num_heads, self.head_dim, self.embed_dim, device=device, dtype=dtype)
        )

        torch.nn.init.normal_(self.out_proj_weight, std=0.02)
        nn.init.xavier_uniform_(self.qkv_proj_weight)

        self.q_proj_weight = None
        self.kv_proj_weight = None
        self.qassmax = (
            QASSMaxScaling(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                hidden_dim=64,
                qass_mode=qass_mode,
                device=device,
                dtype=dtype,
            )
            if use_qassmax
            else None
        )

        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore

    def apply_qassmax(self, q: torch.Tensor, key_len: int) -> torch.Tensor:
        if self.qassmax is None:
            return q
        return self.qassmax(q, key_len)

    def _apply_extra_attn_scale(self, q: torch.Tensor, n_keys: int) -> torch.Tensor:
        """Pre-multiply Q by (temperature * logN factor) so SDPA's internal
        1/sqrt(d) scaling combines to the desired final attention scale.

        - logN factor: log(max(n_keys, 2)) / log(n_ref). When n_keys==n_ref,
          factor==1 and attention is identical to standard. As n grows, the
          factor grows (sharpens softmax) to compensate for soft-distribution
          dilution at long context.
        - Learnable temperature (if enabled): a per-layer scalar nn.Parameter,
          initialized to 1.0, multiplied via .abs() for positivity. Lets the
          model learn per-layer attention sharpness.

        For static-shape tracing under torch.compile, n_keys is a Python int
        (one trace per shape bucket), so math.log is constant in the trace.
        Returns Q multiplied by the factor — or Q unchanged when both flags
        are off.
        """
        if not self.use_logn_attention and not self.use_learnable_attn_temperature:
            return q

        factor: float | torch.Tensor = 1.0
        if self.use_logn_attention:
            n_keys_int = max(int(n_keys), 2)
            log_n = math.log(n_keys_int)
            factor = factor * (log_n / self._attn_n_ref_log)
        if self.use_learnable_attn_temperature and self.attn_temperature is not None:
            # Tensor multiplication; gradient flows through the temperature.
            factor = self.attn_temperature.abs() * factor
        return q * factor

    def _attention_dropout_p(self) -> float:
        """Return functional-attention dropout for the current module mode.

        Functional SDPA does not inspect ``Module.training``; it applies the
        probability the caller passes on every invocation. The module therefore
        has to turn dropout off explicitly during evaluation.
        """
        return float(self.dropout) if self.training else 0.0

    # Below this head count, broadcasting a single K/V head through SDPA's
    # enable_gqa is slower than handing it explicit heads. The cache stays
    # single-head either way; only the transient per-call tensor changes.
    GQA_MIN_HEADS = 4

    def _kv_for_kernel(self, kv: torch.Tensor, *, needs_explicit_heads: bool) -> torch.Tensor:
        """Broadcast single-head MQA K/V as required by the consumer."""
        if kv.size(-2) == self.num_heads:
            return kv
        expanded = kv.expand(*kv.shape[:-2], self.num_heads, kv.shape[-1])
        if needs_explicit_heads:
            return expanded
        if self.num_heads < self.GQA_MIN_HEADS:
            return expanded.contiguous()
        return kv

    def project_kv_cache(
        self,
        x_kv: torch.Tensor,
        *,
        copy_first_head_kv: bool = False,
    ) -> dict[str, torch.Tensor | int]:
        """Project and cache K/V for qkv_combined=False attention.

        The sequence-attention path calls this with x_kv shaped like the
        normal forward input after transposing rows/features: [B, G, N, E].
        We cache the flattened projected K/V tensor so repeated test chunks do
        not re-run train-side K/V projection.
        """
        if self.qkv_combined:
            raise ValueError("project_kv_cache is only valid for separate Q/KV attention")
        B, S, _, _ = x_kv.shape
        x_kv_flat = x_kv.reshape(-1, *x_kv.shape[-2:])
        kv_proj_weight = self.qkv_proj_weight[1:]
        if copy_first_head_kv:
            # Keep the cache single-head. Expanding before the contiguous copy
            # below materializes num_heads identical K/V copies and defeats
            # the memory-saving purpose of MQA.
            kv_weights = kv_proj_weight[:, :1]
            # This head slice is strided in the packed [Q,K,V] parameter.
            # Reshaping it for F.linear copies the weights on every cache
            # build, which is slower than einsum for this MQA-only branch.
            kv = torch.einsum("... s, j h d s -> ... j h d", x_kv_flat, kv_weights)
        else:
            kv = torch.nn.functional.linear(
                x_kv_flat,
                kv_proj_weight.reshape(2 * self.num_heads * self.head_dim, self.embed_dim),
            ).view(*x_kv_flat.shape[:-1], 2, self.num_heads, self.head_dim)
        return {"kv": kv.contiguous(), "batch": B, "groups": S}

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        kv_cache: dict[str, torch.Tensor | int] | BlockwiseSeqKV,
        *,
        calculate_sample_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run cross-attention using a projected train K/V cache."""
        if self.qkv_combined:
            raise ValueError("forward_with_kv_cache is only valid for separate Q/KV attention")
        B, S, _, _ = x.shape
        if int(kv_cache["batch"]) != B or int(kv_cache["groups"]) != S:
            raise ValueError(
                "KV cache shape does not match query shape: "
                f"cache=({kv_cache['batch']}, {kv_cache['groups']}), query=({B}, {S})"
            )
        x_flat = x.reshape(-1, *x.shape[-2:])
        q_proj_weight = self.qkv_proj_weight[0]
        q = torch.nn.functional.linear(
            x_flat,
            q_proj_weight.reshape(self.num_heads * self.head_dim, self.embed_dim),
        ).view(*x_flat.shape[:-1], self.num_heads, self.head_dim)
        blockwise = isinstance(kv_cache, BlockwiseSeqKV)
        kv = None if blockwise else kv_cache["kv"]
        if kv is not None:
            assert isinstance(kv, torch.Tensor)
        n_keys = kv_cache.n_rows if blockwise else int(kv.shape[1])
        if self.use_qassmax:
            q = self.apply_qassmax(q, n_keys)

        if blockwise:
            if calculate_sample_attention:
                raise NotImplementedError("sample-attention diagnostics are not available for streamed K/V")
            if self._attention_dropout_p() != 0.0:
                raise RuntimeError("blockwise K/V attention is inference-only (dropout must be zero)")
            # ``compute_attention_by_torch`` applies the logN / learnable
            # temperature to q itself, which the blockwise path does not go
            # through — so apply it here to give both paths the same effective
            # query. Internal instead folds that factor into an ``sdpa_scale``
            # argument; passing None keeps this tier's already-scaled q on the
            # blockwise kernel's default 1/sqrt(head_dim).
            q = self._apply_extra_attn_scale(q, n_keys)
            atten_out = self.compute_attention_blockwise(q, kv_cache, sdpa_scale=None)
        else:
            kv_perhead = self._kv_for_kernel(kv, needs_explicit_heads=False)
            atten_out = self.compute_attention_by_torch(None, q, kv_perhead, None)

        atten_out = atten_out.reshape(x_flat.shape[0], x_flat.shape[1], self.num_heads, self.head_dim)
        sample_attention = None
        if calculate_sample_attention:
            # Broadcast a single-head (MQA) cache so the diagnostic score has
            # the same per-head shape it had when the cache was materialized.
            assert kv is not None
            k, _ = self._kv_for_kernel(kv, needs_explicit_heads=True).unbind(dim=2)
            sample_attention = self.caculate_attention_score(q[-1], k[-1])
        out = torch.einsum(
            "... h d, h d s -> ... s",
            atten_out,
            self.out_proj_weight,
        )
        return out.reshape(B, S, *out.shape[1:]), sample_attention

    def compute_attention_blockwise(
        self,
        q: torch.Tensor,
        kv_cache: BlockwiseSeqKV,
        *,
        sdpa_scale: float | None = None,
    ) -> torch.Tensor:
        """Exact-context attention with a stable online softmax over CPU K/V blocks.

        ``q`` is ``[BG, Q, H, D]`` and every yielded K/V block is
        ``[BG, Kb, 2, Hkv, D]`` (``Hkv`` is either ``H`` or one for MQA). The
        accumulator is FP32 and has no key-length dimension, so GPU working memory
        is O(Q * H * D + block_rows * Hkv * D), independent of total context rows.
        The total key count was already used for QASS/log-N/temperature before this
        method is called; block size never changes model scaling.
        """
        if self._attention_dropout_p() != 0.0:
            raise RuntimeError("blockwise K/V attention is inference-only (dropout must be zero)")

        if kv_cache.device != q.device:
            raise RuntimeError(
                "streamed K/V cache device does not match the query/model device: "
                f"cache={kv_cache.device}, query={q.device}; rebuild the context cache"
            )
        if q.device.type == "cuda":
            return self._compute_attention_blockwise_flex(
                q,
                kv_cache,
                sdpa_scale=sdpa_scale,
            )
        batch_groups, n_query, n_heads, head_dim = q.shape
        score_bytes_per_key_row = batch_groups * n_heads * n_query * FP32_ELEMENT_BYTES
        # At least one key row must be processed. If even one row exceeds the
        # workspace target, adaptive query chunking is the remaining lever.
        score_limited_rows = max(
            1,
            BLOCKWISE_SCORE_WORKSPACE_BYTES // max(score_bytes_per_key_row, 1),
        )
        block_rows = min(kv_cache.max_block_rows, score_limited_rows)
        running_max = torch.full(
            (batch_groups, n_heads, n_query, 1),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        denominator = torch.zeros_like(running_max)
        numerator = torch.zeros(
            (batch_groups, n_heads, n_query, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        scale = (1.0 / math.sqrt(float(head_dim))) if sdpa_scale is None else sdpa_scale
        q_fp32 = q.to(torch.float32)

        for kv_block in kv_cache.iter_kv_blocks(block_rows=block_rows):
            if kv_block.ndim != 5 or kv_block.shape[0] != batch_groups:
                raise ValueError(f"streamed K/V block must be [BG, K, 2, Hkv, D], got {tuple(kv_block.shape)}")
            k, v = kv_block.unbind(dim=2)
            k_fp32 = k.to(torch.float32)
            v_fp32 = v.to(torch.float32)
            logits = torch.einsum("bqhd,bkhd->bhqk", q_fp32, k_fp32) * scale
            block_max = logits.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.where(
                denominator > 0,
                torch.exp(running_max - new_max),
                torch.zeros_like(new_max),
            )
            # Reuse the score allocation as softmax weights. Keeping both FP32
            # [BG,H,Q,K] tensors doubled the very workspace this path bounds.
            logits.sub_(new_max).exp_()
            block_denominator = logits.sum(dim=-1, keepdim=True)
            block_numerator = torch.einsum("bhqk,bkhd->bhqd", logits, v_fp32)
            numerator = numerator * old_scale + block_numerator
            denominator = denominator * old_scale + block_denominator
            running_max = new_max
            del (
                kv_block,
                k,
                v,
                k_fp32,
                v_fp32,
                logits,
                block_max,
                new_max,
                old_scale,
                block_denominator,
                block_numerator,
            )

        # BlockwiseSeqKV rejects an empty cache and this path has no attention mask,
        # so every query has positive mass. Avoid a device sync just to re-prove it.
        return (numerator / denominator).to(q.dtype).permute(0, 2, 1, 3)

    def _compute_attention_blockwise_flex(
        self,
        q: torch.Tensor,
        kv_cache: BlockwiseSeqKV,
        *,
        sdpa_scale: float | None = None,
    ) -> torch.Tensor:
        """Fuse each K/V block and merge its normalized output by exact LSE.

        FlexAttention never materializes ``[BG, H, Q, K]`` scores. Each call
        returns the block-local normalized output and log-sum-exp; combining
        those pairs in FP32 is algebraically identical to one softmax over all
        context rows while retaining only O(BG * H * Q * D) accumulators.
        """
        batch_groups, n_query, n_heads, head_dim = q.shape
        scale = (1.0 / math.sqrt(float(head_dim))) if sdpa_scale is None else sdpa_scale
        q_flex = q.to(FLASH_ATTN_DTYPE).permute(0, 2, 1, 3).contiguous()
        running_lse = torch.full(
            (batch_groups, n_heads, n_query),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        running_output = torch.zeros(
            (batch_groups, n_heads, n_query, head_dim),
            device=q.device,
            dtype=torch.float32,
        )

        for kv_block in kv_cache.iter_kv_blocks():
            if kv_block.ndim != 5 or kv_block.shape[0] != batch_groups:
                raise ValueError(f"streamed K/V block must be [BG, K, 2, Hkv, D], got {tuple(kv_block.shape)}")
            k, v = kv_block.unbind(dim=2)
            k_flex = k.to(FLASH_ATTN_DTYPE).permute(0, 2, 1, 3).contiguous()
            v_flex = v.to(FLASH_ATTN_DTYPE).permute(0, 2, 1, 3).contiguous()
            block_output, block_lse = _flex_attention_with_lse(
                q_flex,
                k_flex,
                v_flex,
                scale=scale,
                enable_gqa=k_flex.shape[1] != n_heads,
            )
            if block_lse.shape != running_lse.shape:
                raise RuntimeError(
                    "FlexAttention returned an unexpected LSE shape: "
                    f"expected {tuple(running_lse.shape)}, "
                    f"got {tuple(block_lse.shape)}"
                )
            block_lse = block_lse.to(torch.float32)
            new_lse = torch.logaddexp(running_lse, block_lse)
            old_weight = torch.exp(running_lse - new_lse).unsqueeze(-1)
            block_weight = torch.exp(block_lse - new_lse).unsqueeze(-1)
            running_output.mul_(old_weight)
            running_output.add_(block_output.to(torch.float32) * block_weight)
            running_lse = new_lse
            del (
                kv_block,
                k,
                v,
                k_flex,
                v_flex,
                block_output,
                block_lse,
                new_lse,
                old_weight,
                block_weight,
            )

        return running_output.to(q.dtype).permute(0, 2, 1, 3)

    def compute_attention_by_torch(
        self, qkv: torch.Tensor | None, q: torch.Tensor | None, kv: torch.Tensor | None, attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Compute attention with PyTorch scaled_dot_product_attention (supports attn_mask)."""
        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None and q is not None:
            k, v = kv.unbind(dim=-3)
        else:
            raise ValueError("When qkv is None, q and kv cannot both be None at the same time")
        assert q is not None and k is not None and v is not None, "q, k, and v must not be None"

        # Apply logN + learnable-temperature scaling to Q before SDPA.
        # SDPA's internal 1/sqrt(d) scale is unchanged; pre-scaling Q gives
        # the desired final attention scale.
        n_keys = k.size(1)  # k shape: [B, n_keys, num_heads, head_dim]
        q = self._apply_extra_attn_scale(q, n_keys)

        # A single-head MQA K/V is broadcast across Q's heads by SDPA. Equal
        # head counts leave GQA disabled, preserving the ordinary MHA path.
        enable_gqa = k.size(2) != q.size(2)

        # Newer torch releases can prefer cuDNN SDPA for these shapes. That
        # backend has been slow/intermittently broken for Nori's dynamic table
        # sizes and small head_dim=16. Exclude it only for this call rather than
        # mutating process-wide torch backend state at import time.
        backend_context = nullcontext() if _ALLOW_CUDNN_SDP else sdpa_kernel(_SDPA_BACKENDS_WITHOUT_CUDNN)
        with backend_context:
            # q/k/v: [B, seq, num_heads, head_dim]. SDPA grids over (B * num_heads);
            # above CUDA's 65535 grid-dimension limit it raises "invalid
            # configuration argument". Feature/sample attention is independent
            # across the batch dimension, so chunking B and concatenating is
            # mathematically exact. Keep every legal launch whole: a lower 32768
            # cutoff needlessly split production shapes such as batch=20,
            # rows=1536, heads=2 (61440), adding a second SDPA launch and
            # concatenation in every encoder layer.
            #
            # The QASS temperature is already folded into q by
            # ``_apply_extra_attn_scale`` above, so no explicit ``scale=`` is
            # passed here — internal instead threads an ``sdpa_scale`` argument,
            # which is the same quantity applied one step later.
            B = q.size(0)
            dropout_p = self._attention_dropout_p()
            if B * self.num_heads > SDPA_BATCH_HEAD_LIMIT:
                chunk = max(1, SDPA_BATCH_HEAD_LIMIT // max(self.num_heads, 1))
                mask_batched = attn_mask is not None and attn_mask.dim() >= 1 and attn_mask.size(0) == B
                outs = []
                for s in range(0, B, chunk):
                    e = min(s + chunk, B)
                    m = attn_mask[s:e] if mask_batched else attn_mask
                    o = torch.nn.functional.scaled_dot_product_attention(
                        q[s:e].transpose(1, 2),
                        k[s:e].transpose(1, 2),
                        v[s:e].transpose(1, 2),
                        attn_mask=m,
                        dropout_p=dropout_p,
                        enable_gqa=enable_gqa,
                    )
                    outs.append(o.transpose(1, 2))
                return torch.cat(outs, dim=0)

            attention_outputs = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                enable_gqa=enable_gqa,
            )
        attention_outputs = attention_outputs.transpose(1, 2)
        return attention_outputs

    def caculate_attention_score(self, q: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
        if len(q.shape) == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
        logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
        logits *= torch.sqrt(torch.tensor(1.0 / q.shape[-1])).to(k.device)
        ps = torch.softmax(logits.float(), dim=2).to(torch.float16).mean(dim=-1)
        del logits
        return ps

    @override
    def forward(
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        copy_first_head_kv: bool = False,
        attn_mask: torch.Tensor | None = None,
        calculate_sample_attention: bool = False,
        calculate_feature_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        x: [batch_size, seq_len, feature, embed_dim]
        kv: Optional[batch_size, seq_len_kv, feature, embed_dim] — only needed if qkv_combined=False
        copy_first_head: Reuse the results from the first attention head
        """
        # feature attention: [B S F E]
        # item attention: [B F S E]
        # B, T, C = x.shape
        B, S, _, _ = x.shape
        assert x.shape[-1] == self.embed_dim

        x = x.reshape(-1, *x.shape[-2:])
        BS, F, E = x.shape

        qkv = None
        q = None
        kv = None
        feature_attention = None
        sample_attention = None
        # batch_size = None
        # seqlen = None
        if self.qkv_combined:
            qkv = torch.einsum("... s, j h d s -> ... j h d", x, self.qkv_proj_weight)
            if self.use_qassmax:
                q_scaled = self.apply_qassmax(qkv[:, :, 0], qkv.shape[1])
                qkv = torch.stack((q_scaled, qkv[:, :, 1], qkv[:, :, 2]), dim=2)
        else:
            self.q_proj_weight = self.qkv_proj_weight[0]
            self.kv_proj_weight = self.qkv_proj_weight[1:]
            assert x_kv is not None, "kv combined attention requires kv input"
            x_kv = x_kv.reshape(-1, *x_kv.shape[-2:])
            q = torch.einsum("... s, h d s -> ... h d", x, self.q_proj_weight)
            if copy_first_head_kv:
                kv_weights = self.kv_proj_weight[:, :1]
                kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, kv_weights)
            else:
                kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, self.kv_proj_weight)
            if self.use_qassmax:
                q = self.apply_qassmax(q, kv.shape[1])

        kv_perhead = kv if kv is None else self._kv_for_kernel(kv, needs_explicit_heads=False)
        atten_out = self.compute_attention_by_torch(qkv, q, kv_perhead, attn_mask)

        atten_out = atten_out.reshape(BS, F, self.num_heads, self.head_dim)

        if qkv is not None:
            q, k, v = qkv.unbind(dim=2)
        else:
            k, v = self._kv_for_kernel(kv, needs_explicit_heads=True).unbind(dim=2)
        if calculate_feature_attention:
            feature_attention = self.caculate_attention_score(q, k)

        if calculate_sample_attention:
            sample_attention = self.caculate_attention_score(q[-1], k[-1])
        out = torch.einsum(
            "... h d, h d s -> ... s",
            atten_out,
            self.out_proj_weight,
        )

        return out.reshape(B, S, *out.shape[1:]), feature_attention, sample_attention


class EncoderBaseLayer(nn.Module):
    "Base encoder layer of the Transformer model"

    def __init__(
        self,
        nhead: int,
        embed_dim: int,
        hid_dim: int,
        dropout: float = 0,
        pre_norm: bool = False,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        recompute_attn: bool = False,
        mlp_use_residual: bool = False,
        layer_arch: str = "fmfmsm",
        seq_attn_isolated: bool = False,
        seq_attn_serial: bool = False,
        self_share_all_kv_heads: bool = False,
        cross_share_all_kv_heads: bool = True,
        use_qassmax: bool = False,
        norm_type: str = "layernorm",
        deepnorm_alpha: float | None = None,
        use_logn_attention: bool = False,
        use_learnable_attn_temperature: bool = False,
        attn_n_ref: float = 1024.0,
        # Appended, not inserted — this signature is not keyword-only.
        qass_mode: str | None = None,
    ):
        super().__init__()
        if mlp_use_residual:
            raise ValueError(
                "mlp_use_residual=True is unsupported: every MLP sublayer already "
                "uses the transformer's outer residual connection. This legacy "
                "flag was a no-op; leave it false or remove it from the config."
            )
        self.use_logn_attention = use_logn_attention
        self.use_learnable_attn_temperature = use_learnable_attn_temperature
        self.attn_n_ref = float(attn_n_ref)
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.device = device
        self.dtype = dtype
        self.layer_arch = layer_arch
        self.head_dim = self.embed_dim // self.nhead
        self.recompute_attn = recompute_attn

        self.feature_attentions = []
        self.sequence_attentions = []
        self.mlp = []
        self.feature_attn_num = 1  # feature attention number
        self.seq_attn_num = 1  # sequence attention number
        self.mlp_num = 1  # mlp number
        if layer_arch == "fmfmsm":
            self.feature_attn_num = 2
            self.mlp_num = 3

        self.norm_type = norm_type
        if deepnorm_alpha is not None:
            self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha, dtype=torch.float32))
        else:
            self.deepnorm_alpha = None
        self.self_share_all_kv_heads = self_share_all_kv_heads
        self.cross_share_all_kv_heads = cross_share_all_kv_heads
        self.seq_attn_serial = seq_attn_serial
        self.seq_attn_isolated = seq_attn_isolated
        self.use_qassmax = use_qassmax

        if self.seq_attn_isolated:
            self.seq_attn_num *= 2

        # attention+MLP
        self.feature_attentions = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.nhead,
                    device=self.device,
                    dtype=self.dtype,
                    qkv_combined=True,
                    dropout=self.dropout,
                    recompute=self.recompute_attn,
                    use_qassmax=False,
                    use_logn_attention=use_logn_attention,
                    use_learnable_attn_temperature=use_learnable_attn_temperature,
                    attn_n_ref=attn_n_ref,
                )
                for _ in range(self.feature_attn_num)
            ]
        )
        self.sequence_attentions = nn.ModuleList(
            [
                MultiheadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.nhead,
                    device=self.device,
                    dtype=self.dtype,
                    qkv_combined=False,
                    dropout=self.dropout,
                    recompute=self.recompute_attn,
                    use_qassmax=use_qassmax,
                    qass_mode=qass_mode,
                    use_logn_attention=use_logn_attention,
                    use_learnable_attn_temperature=use_learnable_attn_temperature,
                    attn_n_ref=attn_n_ref,
                )
                for _ in range(self.seq_attn_num)
            ]
        )
        if self.activation == "swiglu":
            mlp_creator = lambda: SwiGLUMLP(
                in_features=self.embed_dim,
                hidden_size=self.hid_dim,
                out_features=self.embed_dim,
                has_bias=False,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            mlp_creator = lambda: MLP(
                in_features=self.embed_dim,
                hidden_size=self.hid_dim,
                out_features=self.embed_dim,
                has_bias=False,
                device=self.device,
                dtype=self.dtype,
                activation=self.activation,
                depth=2,
            )
        self.mlp = nn.ModuleList([mlp_creator() for _ in range(self.mlp_num)])

        self.layer_steps = []
        if self.layer_arch == "fmfmsm":
            assert len(self.feature_attentions) >= 2 and len(self.sequence_attentions) >= 1 and len(self.mlp) >= 3
            self.layer_steps = [
                partial(self.call_features_attention, index=0),
                self.mlp[0],
                partial(self.call_features_attention, index=1),
                self.mlp[1],
                partial(self.call_sequence_attention, index=0),
                self.mlp[2],
            ]
        elif self.layer_arch == "smf":
            assert len(self.feature_attentions) >= 1 and len(self.sequence_attentions) >= 1 and len(self.mlp) >= 1
            self.layer_steps = [
                partial(self.call_sequence_attention, index=0),
                self.mlp[0],
                partial(self.call_features_attention, index=0),
            ]
        else:
            raise ValueError(f"Unsupport layr arch: {self.layer_arch}")

        # Locate diagnostic capture points from the actual architecture rather
        # than numeric step positions. smf puts sequence attention at step 0;
        # fmfmsm puts it at step 4 and has two feature-attention steps. The
        # intended map is from the final attention operation of each kind.
        self._feature_capture_idx = next(
            (
                i
                for i in reversed(range(len(self.layer_steps)))
                if isinstance(self.layer_steps[i], functools.partial)
                and self.layer_steps[i].func == self.call_features_attention
            ),
            None,
        )
        self._sequence_capture_idx = next(
            (
                i
                for i in reversed(range(len(self.layer_steps)))
                if isinstance(self.layer_steps[i], functools.partial)
                and self.layer_steps[i].func == self.call_sequence_attention
            ),
            None,
        )

        if self.norm_type == "rmsnorm":
            norm_creator = lambda: RMSNorm(
                self.embed_dim, eps=self.layer_norm_eps, elementwise_affine=False, device=self.device, dtype=self.dtype
            )
        else:
            norm_creator = lambda: LayerNormMixedPrecision(
                normalized_shape=self.embed_dim,
                eps=self.layer_norm_eps,
                elementwise_affine=False,
                device=self.device,
                dtype=self.dtype,
            )
        self.layer_norms = nn.ModuleList([norm_creator() for _ in range(len(self.layer_steps))])

    def create_attn_mask(self, q_mask: torch.Tensor, k_mask: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask

        Args:
            q_mask (torch.Tensor): Query sequence mask, with shape [batch_size, head_count, q_seq_len]
            k_mask (torch.Tensor): Key sequence mask, with shape   [batch_size, head_count, k_seq_len]

        Returns:
            torch.Tensor: attention mask, with shape [batch_size, head_count, q_seq_len, k_seq_len]
        """
        _, _, q_seq_len = q_mask.shape
        _, _, k_seq_len = k_mask.shape

        q_mask_bool = q_mask.bool()  # [batch_size, head_count, q_seq_len]
        k_mask_bool = k_mask.bool()  # [batch_size, head_count, k_seq_len]

        q_expanded = q_mask_bool.unsqueeze(-1)
        k_expanded = k_mask_bool.unsqueeze(-2)

        # PyTorch SDPA's boolean-mask contract is True == participates in
        # attention (the opposite of nn.MultiheadAttention's key-padding mask).
        # Return the valid pairs directly; inverting them makes every padded
        # feature attend while every real feature is excluded.
        attn_mask = q_expanded & k_expanded
        _, _, q_seq_len, k_seq_len = attn_mask.shape
        attn_mask = attn_mask.reshape(-1, q_seq_len, k_seq_len)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)

        return attn_mask

    def call_features_attention(
        self,
        x: torch.Tensor,
        feature_atten_mask: torch.Tensor | None,
        eval_pos: int,
        index: int = 0,
        calculate_feature_attention: bool = False,
    ):
        assert len(self.feature_attentions) > index
        attn_mask = None
        if feature_atten_mask is not None:
            attn_mask = self.create_attn_mask(feature_atten_mask, feature_atten_mask)
        return self.feature_attentions[index](
            x, x_kv=None, attn_mask=attn_mask, calculate_feature_attention=calculate_feature_attention
        )

    def call_sequence_attention(
        self,
        x: torch.Tensor,
        feature_atten_mask: torch.Tensor | None,
        eval_pos: int,
        index: int = 0,
        calculate_sample_attention: bool = False,
    ):
        assert len(self.sequence_attentions) > index
        sample_attention = None
        index1 = index * 2 if self.seq_attn_isolated else index
        index2 = index1 + 1 if self.seq_attn_isolated else index1
        assert index2 < len(self.sequence_attentions), (
            f"Error: index2({index2}) >= len(self.sequence_attentions)({len(self.sequence_attentions)})"
        )

        x_train = self.sequence_attentions[index1](
            x=x[:, :eval_pos].transpose(1, 2),
            x_kv=x[:, :eval_pos].transpose(1, 2),
            copy_first_head_kv=True if self.self_share_all_kv_heads else False,
        )[0].transpose(1, 2)

        if eval_pos < x.shape[1]:
            # KV source: updated x_train when serial, original context otherwise
            kv = x_train.transpose(1, 2) if self.seq_attn_serial else x[:, :eval_pos].transpose(1, 2)
            x_test, _, sample_attention = self.sequence_attentions[index2](
                x=x[:, eval_pos:].transpose(1, 2),
                x_kv=kv,
                copy_first_head_kv=True if self.cross_share_all_kv_heads else False,
                calculate_sample_attention=calculate_sample_attention,
            )
            x_test = x_test.transpose(1, 2)
            return torch.cat([x_train, x_test], dim=1), None, sample_attention
        else:
            return x_train, None, None

    def _residual_add(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        if self.deepnorm_alpha is not None:
            return residual + self.deepnorm_alpha * update
        return residual + update

    def _run_non_sequence_step(
        self,
        x: torch.Tensor,
        *,
        step_idx: int,
        feature_atten_mask: torch.Tensor | None,
        eval_pos: int,
    ) -> torch.Tensor:
        sublayer = self.layer_steps[step_idx]
        layer_norm = self.layer_norms[step_idx]
        if self.pre_norm:
            residual = x
            x_norm = layer_norm(x)
            if isinstance(sublayer, functools.partial):
                out = sublayer(x_norm, feature_atten_mask, eval_pos)
                if isinstance(out, tuple):
                    out = out[0]
            else:
                out = sublayer(x_norm)
                if isinstance(out, tuple):
                    out = out[0]
            return self._residual_add(residual, out)

        residual = x
        if isinstance(sublayer, functools.partial):
            out = sublayer(x, feature_atten_mask, eval_pos)
            if isinstance(out, tuple):
                out = out[0]
        else:
            out = sublayer(x)
            if isinstance(out, tuple):
                out = out[0]
        return layer_norm(out + residual)

    def _project_kv_cache_rowchunked(
        self,
        attn_mod,
        x_kv_src,
        copy_kv,
        row_chunk,
        *,
        norm=None,
        offload=False,
        quantize=False,
        device=None,
        streaming=False,
    ):
        """Build attn_mod's K/V cache over the row axis in chunks.

        Mirrors ``project_kv_cache(full)`` bit-for-bit (K/V are per-row
        independent), but never materialises the full-N projection transient
        (the fp32 einsum spike that OOMs at large N*groups) -- it projects
        ``row_chunk`` rows at a time.

        Each slice is quantized and/or moved to host **as it is produced**
        (``ScalableSeqKV.from_row_slices``), so the full-precision cache is
        never resident in whole -- neither as a slice list nor as a
        concatenated copy. That matters because this path is entered precisely
        when memory is already tight: assembling at full precision and
        quantizing afterwards would peak at two full copies of the cache.

        Args:
            attn_mod: the attention module whose K/V projection to run.
            x_kv_src: source activations, ``[B, N, groups, E]``.
            copy_kv: forwarded to ``project_kv_cache`` as ``copy_first_head_kv``.
            row_chunk: context rows to project per step.
            norm: optional layer norm applied per-slice before projecting (the
                ``pre_norm`` path); applied inside the loop so normalising does
                not itself materialise all N rows.
            offload: store the finished cache on host RAM.
            quantize: store the finished cache int8.
            device: compute device reads are staged back to.
            streaming: retain separate blocks and forbid full-device staging.

        Returns:
            A blockwise cache when streaming, ``ScalableSeqKV`` when merely
            quantizing/offloading, else the plain cache dict.
        """
        N = x_kv_src.shape[1]
        B, groups = x_kv_src.shape[0], x_kv_src.shape[2]
        compute_device = torch.device(device) if device is not None else attn_mod.qkv_proj_weight.device

        def _row_slices():
            """Yield each row-slice's projected K/V, one at a time.

            A generator so the consumer folds each slice in and drops it before the
            next is projected -- the whole point of this path.
            """
            for r0 in range(0, N, row_chunk):
                sl = slice(r0, min(r0 + row_chunk, N))
                rows = x_kv_src[:, sl]
                if rows.device != compute_device:
                    rows = rows.to(compute_device)
                if norm is not None:
                    rows = norm(rows)
                projected = attn_mod.project_kv_cache(rows.transpose(1, 2), copy_first_head_kv=copy_kv)["kv"]
                del rows
                yield projected
                del projected

        if streaming:
            return BlockwiseSeqKV.from_row_slices(
                _row_slices(),
                B,
                groups,
                quantize=quantize,
                device=compute_device,
                dtype=x_kv_src.dtype,
                offload=offload,
                n_rows=N,
            )
        if not (quantize or offload):
            # Nothing to fold in; the plain dict is what the uninstrumented path uses.
            return {"kv": torch.cat(list(_row_slices()), dim=1), "batch": B, "groups": groups}
        return ScalableSeqKV.from_row_slices(
            _row_slices(),
            B,
            groups,
            quantize=quantize,
            offload=offload,
            device=compute_device,
            dtype=x_kv_src.dtype,
        )

    def _forward_train_cache_streaming(
        self,
        x_train,
        feature_atten_mask,
        pre_seq_steps,
        seq_step,
        post_seq_steps,
        index1,
        index2,
        row_chunk,
        quantize,
        device,
        *,
        cross_cache_offload=True,
    ):
        """Run one context layer with only row-bounded tensors on the GPU.

        The complete layer input/output and self-attention K/V representation live
        on CPU. Each row block is staged for row-local work; self-attention scans
        CPU K/V blocks with FP32 online softmax, so neither an N-row activation nor
        a full layer's BF16 K/V is ever materialized on the GPU. Explicit context
        streaming also offloads the reusable cross-cache. The private hybrid path
        instead writes that cache directly into preallocated resident INT8 storage.

        Sequence-serial layers are rejected by the caller: their cross-cache source
        is the complete self-attention output, which needs a different two-pass
        schedule to preserve the same semantics.
        """
        compute_device = torch.device(device) if device is not None else next(self.parameters()).device
        if feature_atten_mask is not None:
            raise NotImplementedError(
                "stream_context does not yet support feature_atten_mask; staging "
                "a full mask would violate the bounded-GPU-memory contract"
            )
        if x_train.device.type != "cpu":
            raise ValueError(
                "stream_context requires CPU-resident context activations; the "
                "predictor must not upload the full training matrix"
            )
        seq_attn_train = self.sequence_attentions[index1]
        seq_attn_test = self.sequence_attentions[index2]
        seq_norm = self.layer_norms[seq_step]
        n_rows = x_train.shape[1]

        def _write_host(source, transform):
            if n_rows <= 0:
                raise ValueError("stream_context received an empty context")
            for row_start in range(0, n_rows, row_chunk):
                row_end = min(row_start + row_chunk, n_rows)
                row_slice = slice(row_start, row_end)
                staged = source[:, row_slice].to(compute_device)
                staged = transform(staged)
                host = staged.detach().to("cpu")
                source[:, row_slice].copy_(host)
                del staged, host
            return source

        def _run_steps(staged, steps):
            for step_idx in steps:
                staged = self._run_non_sequence_step(
                    staged,
                    step_idx=step_idx,
                    feature_atten_mask=feature_atten_mask,
                    eval_pos=staged.shape[1],
                )
            return staged

        # FMFMSM's feature/MLP steps are row-local, but every row must reach the
        # sequence-attention input before any K/V is projected.
        prepared = _write_host(x_train, lambda rows: _run_steps(rows, pre_seq_steps)) if pre_seq_steps else x_train
        del x_train
        norm = seq_norm if self.pre_norm else None

        def _project(module, copy_first_head, *, quantize_cache, offload_cache):
            return self._project_kv_cache_rowchunked(
                module,
                prepared,
                copy_first_head,
                row_chunk,
                norm=norm,
                quantize=quantize_cache,
                offload=offload_cache,
                device=compute_device,
                streaming=True,
            )

        # Prefill self-attention remains full precision. Only the reusable cross
        # cache follows cache_dtype=int8, matching the existing cached path.
        train_cache = _project(
            seq_attn_train,
            self.self_share_all_kv_heads,
            quantize_cache=False,
            offload_cache=True,
        )
        test_cache = _project(
            seq_attn_test,
            self.cross_share_all_kv_heads,
            quantize_cache=quantize,
            offload_cache=cross_cache_offload,
        )

        def _self_attention_and_post(rows):
            residual = rows
            if self.pre_norm:
                attention = seq_attn_train.forward_with_kv_cache(
                    seq_norm(rows).transpose(1, 2),
                    train_cache,
                )[0].transpose(1, 2)
                rows = self._residual_add(residual, attention)
            else:
                attention = seq_attn_train.forward_with_kv_cache(
                    rows.transpose(1, 2),
                    train_cache,
                )[0].transpose(1, 2)
                rows = seq_norm(attention + residual)
            return _run_steps(rows, post_seq_steps)

        output = _write_host(prepared, _self_attention_and_post)
        return output, {"seq_kv": test_cache}

    def _forward_train_cache_memsaving(
        self,
        x_train,
        feature_atten_mask,
        pre_seq_steps,
        seq_step,
        post_seq_steps,
        index1,
        index2,
        row_chunk,
        quantize,
        offload,
        device,
    ):
        """Memory-bounded fit-time cache build (TabPFN-3-style, non-serial only).

        Keeps only ONE full-N K/V tensor resident: the self-attention cache.
        Everything else -- queries, attention output, MLP hidden, and the
        stored test cache -- is processed / built in row-chunks and written
        back into ``x_train`` in place, so the peak is ~ K/V + input instead of
        the ~4-5x working set the monolithic path materialises. Bit-exact: the
        self-attention becomes project-once + chunked cached-query reads (each
        query row attends to the full K/V), and the per-row MLP/feature-attn
        steps are unchanged."""
        seq_attn_train = self.sequence_attentions[index1]
        seq_attn_test = self.sequence_attentions[index2]
        seq_norm = self.layer_norms[seq_step]
        N = x_train.shape[1]

        def _rows_inplace(steps):
            for r0 in range(0, N, row_chunk):
                sl = slice(r0, min(r0 + row_chunk, N))
                xc = x_train[:, sl]
                for st in steps:
                    xc = self._run_non_sequence_step(
                        xc, step_idx=st, feature_atten_mask=feature_atten_mask, eval_pos=xc.shape[1]
                    )
                x_train[:, sl] = xc

        # pre-seq per-row steps (fmfmsm) -- must finish for all rows before we
        # project K/V from the post-pre-step activations.
        _rows_inplace(pre_seq_steps)

        # Build the self-attention cache (resident) and the stored test cache
        # (quantized/offloaded during build). Non-serial => both read the same
        # source. The norm, when this layer is pre_norm, is applied per-slice
        # inside the projector so it does not materialise all N rows either.
        norm = seq_norm if self.pre_norm else None

        def _proj(mod, copy_kv, **scale):
            return self._project_kv_cache_rowchunked(
                mod, x_train, copy_kv, row_chunk, norm=norm, device=device, **scale
            )

        train_cache = _proj(seq_attn_train, self.self_share_all_kv_heads)
        test_cache = _proj(seq_attn_test, self.cross_share_all_kv_heads, quantize=quantize, offload=offload)

        # Self-attention as chunked cached-query reads + post steps, in place.
        for r0 in range(0, N, row_chunk):
            sl = slice(r0, min(r0 + row_chunk, N))
            xr = x_train[:, sl]
            if self.pre_norm:
                attn_r = seq_attn_train.forward_with_kv_cache(seq_norm(xr).transpose(1, 2), train_cache)[0].transpose(
                    1, 2
                )
                xr = self._residual_add(xr, attn_r)
            else:
                attn_r = seq_attn_train.forward_with_kv_cache(xr.transpose(1, 2), train_cache)[0].transpose(1, 2)
                xr = seq_norm(attn_r + xr)
            for st in post_seq_steps:
                xr = self._run_non_sequence_step(
                    xr, step_idx=st, feature_atten_mask=feature_atten_mask, eval_pos=xr.shape[1]
                )
            x_train[:, sl] = xr
        del train_cache
        return x_train, {"seq_kv": test_cache}

    def forward_train_cache(
        self,
        x_train: torch.Tensor,
        feature_atten_mask: torch.Tensor | None = None,
        fit_row_chunk: int | None = None,
        quantize_kv_cache: bool = False,
        offload_kv_cache: bool = False,
        device=None,
        stream_context: bool = False,
        _hybrid_resident_int8_prefill: bool = False,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor | int]]]:
        """Run the train rows through one layer and cache train K/V for tests.

        This mirrors forward(..., eval_pos=n_train) for the train side, but it
        also stores projected K/V for the subsequent test cross-attention. It
        is intentionally inference-only; training still uses the standard path.

        fit_row_chunk switches to a
        memory-bounded build (TabPFN-3-style: chunked queries + in-place
        residual + chunked/offloaded cache projection) that keeps only the
        resident K/V tensor full-size. Supported for non-serial seq attention;
        serial raises rather than silently ignoring the requested memory cap.

        stream_context additionally keeps full-N activations and every K/V layer
        on CPU; only ``fit_row_chunk`` rows are staged on the compute device.
        """

        if self.layer_arch == "fmfmsm":
            pre_seq_steps = (0, 1, 2, 3)
            seq_step = 4
            post_seq_steps = (5,)
            seq_index = 0
        elif self.layer_arch == "smf":
            pre_seq_steps = ()
            seq_step = 0
            post_seq_steps = (1, 2)
            seq_index = 0
        else:
            raise NotImplementedError("Cached inference currently supports layer_arch='fmfmsm' and 'smf'")

        index1 = seq_index * 2 if self.seq_attn_isolated else seq_index
        index2 = index1 + 1 if self.seq_attn_isolated else index1

        # Memory-bounded fit build (TabPFN-3-style). Non-serial only: the serial
        # variant's test cache reads the self-attention OUTPUT, which the chunked
        # path would have to fully materialise -- defeating the purpose.
        #
        # Raise rather than fall through to the monolithic path. Silently ignoring a
        # requested memory lever is how a caller ends up believing the cap applied
        # while the build still peaks at O(N); the predictor catches this to decide
        # whether to escalate elsewhere.
        bounded_prefill = stream_context or _hybrid_resident_int8_prefill
        if (fit_row_chunk or bounded_prefill) and self.seq_attn_serial:
            requested = "stream_context" if stream_context else "context_row_chunk"
            raise NotImplementedError(
                f"{requested} is not supported for serial sequence attention: "
                "the serial variant's test cache reads the self-attention output, so "
                "the bounded build needs a different two-pass schedule. Use a "
                "non-serial checkpoint for streaming, or disable context_row_chunk "
                "only when the full context safely fits."
            )
        if bounded_prefill:
            if not fit_row_chunk:
                requested = "stream_context" if stream_context else "hybrid resident INT8 prefill"
                raise ValueError(f"{requested} requires a concrete context_row_chunk")
            if stream_context and not offload_kv_cache:
                raise ValueError("stream_context requires offload_kv_cache=True")
            if _hybrid_resident_int8_prefill and (stream_context or offload_kv_cache or not quantize_kv_cache):
                raise ValueError(
                    "hybrid resident INT8 prefill requires stream_context=False, "
                    "offload_kv_cache=False, and quantize_kv_cache=True"
                )
            return self._forward_train_cache_streaming(
                x_train,
                feature_atten_mask,
                pre_seq_steps,
                seq_step,
                post_seq_steps,
                index1,
                index2,
                fit_row_chunk,
                quantize_kv_cache,
                device,
                cross_cache_offload=not _hybrid_resident_int8_prefill,
            )
        if fit_row_chunk:
            return self._forward_train_cache_memsaving(
                x_train,
                feature_atten_mask,
                pre_seq_steps,
                seq_step,
                post_seq_steps,
                index1,
                index2,
                fit_row_chunk,
                quantize_kv_cache,
                offload_kv_cache,
                device,
            )

        eval_pos = x_train.shape[1]
        for step_idx in pre_seq_steps:
            x_train = self._run_non_sequence_step(
                x_train, step_idx=step_idx, feature_atten_mask=feature_atten_mask, eval_pos=eval_pos
            )

        seq_attn_train = self.sequence_attentions[index1]
        seq_attn_test = self.sequence_attentions[index2]
        seq_norm = self.layer_norms[seq_step]

        if self.pre_norm:
            residual = x_train
            x_norm = seq_norm(x_train)
            train_attn = seq_attn_train(
                x=x_norm.transpose(1, 2),
                x_kv=x_norm.transpose(1, 2),
                copy_first_head_kv=True if self.self_share_all_kv_heads else False,
            )[0].transpose(1, 2)
            kv_source = train_attn if self.seq_attn_serial else x_norm
            seq_kv_cache = seq_attn_test.project_kv_cache(
                kv_source.transpose(1, 2),
                copy_first_head_kv=True if self.cross_share_all_kv_heads else False,
            )
            x_train = self._residual_add(residual, train_attn)
        else:
            residual = x_train
            train_attn = seq_attn_train(
                x=x_train.transpose(1, 2),
                x_kv=x_train.transpose(1, 2),
                copy_first_head_kv=True if self.self_share_all_kv_heads else False,
            )[0].transpose(1, 2)
            kv_source = train_attn if self.seq_attn_serial else x_train
            seq_kv_cache = seq_attn_test.project_kv_cache(
                kv_source.transpose(1, 2),
                copy_first_head_kv=True if self.cross_share_all_kv_heads else False,
            )
            x_train = seq_norm(train_attn + residual)

        for step_idx in post_seq_steps:
            x_train = self._run_non_sequence_step(
                x_train, step_idx=step_idx, feature_atten_mask=feature_atten_mask, eval_pos=eval_pos
            )
        return x_train, {"seq_kv": seq_kv_cache}

    def forward_test_with_cache(
        self,
        x_test: torch.Tensor,
        cache: dict[str, dict[str, torch.Tensor | int]],
        feature_atten_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run test rows through one layer using train K/V from forward_train_cache."""
        if self.layer_arch == "fmfmsm":
            pre_seq_steps = (0, 1, 2, 3)
            seq_step = 4
            post_seq_steps = (5,)
            seq_index = 0
        elif self.layer_arch == "smf":
            pre_seq_steps = ()
            seq_step = 0
            post_seq_steps = (1, 2)
            seq_index = 0
        else:
            raise NotImplementedError("Cached inference currently supports layer_arch='fmfmsm' and 'smf'")

        eval_pos = x_test.shape[1]
        for step_idx in pre_seq_steps:
            x_test = self._run_non_sequence_step(
                x_test, step_idx=step_idx, feature_atten_mask=feature_atten_mask, eval_pos=eval_pos
            )

        index1 = seq_index * 2 if self.seq_attn_isolated else seq_index
        index2 = index1 + 1 if self.seq_attn_isolated else index1
        seq_attn_test = self.sequence_attentions[index2]
        seq_norm = self.layer_norms[seq_step]
        if self.pre_norm:
            residual = x_test
            x_norm = seq_norm(x_test)
            test_attn, _ = seq_attn_test.forward_with_kv_cache(
                x_norm.transpose(1, 2),
                cache["seq_kv"],
            )
            test_attn = test_attn.transpose(1, 2)
            x_test = self._residual_add(residual, test_attn)
        else:
            residual = x_test
            test_attn, _ = seq_attn_test.forward_with_kv_cache(
                x_test.transpose(1, 2),
                cache["seq_kv"],
            )
            test_attn = test_attn.transpose(1, 2)
            x_test = seq_norm(test_attn + residual)

        for step_idx in post_seq_steps:
            x_test = self._run_non_sequence_step(
                x_test, step_idx=step_idx, feature_atten_mask=feature_atten_mask, eval_pos=eval_pos
            )
        return x_test

    def forward(
        self, x: torch.Tensor, feature_atten_mask: torch.Tensor, eval_pos: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        calculate_sample_attention = kwargs.get("calculate_sample_attention", False)
        calculate_feature_attention = kwargs.get("calculate_feature_attention", False)
        layer_idx = kwargs.get("layer_idx", 11)
        # LayerStack marks the real final layer for arbitrary depths. Retain the
        # legacy layer-11 fallback for callers that drive an EncoderBaseLayer
        # directly and therefore cannot provide stack position metadata.
        is_capture_layer = kwargs.get("is_last_layer", layer_idx == 11)

        feature_attention = None
        sample_attention = None
        for idx, (sublayer, layer_norm) in enumerate(zip(self.layer_steps, self.layer_norms)):
            if self.pre_norm:
                residual = x
                x = layer_norm(x)
                if idx == self._feature_capture_idx and calculate_feature_attention and is_capture_layer:
                    x, feature_attention, _ = sublayer(
                        x, feature_atten_mask, eval_pos, calculate_feature_attention=True
                    )
                elif idx == self._sequence_capture_idx and calculate_sample_attention and is_capture_layer:
                    x, _, sample_attention = sublayer(x, feature_atten_mask, eval_pos, calculate_sample_attention=True)
                else:
                    if isinstance(sublayer, functools.partial):
                        x = sublayer(x, feature_atten_mask, eval_pos)
                        if isinstance(x, tuple):
                            x = x[0]
                    else:
                        x = sublayer(x)
                        if isinstance(x, tuple):
                            x = x[0]
                if self.deepnorm_alpha is not None:
                    x = residual + self.deepnorm_alpha * x
                else:
                    x = x + residual
            else:
                residual = x
                if idx == self._feature_capture_idx and calculate_feature_attention and is_capture_layer:
                    x, feature_attention, _ = sublayer(
                        x, feature_atten_mask, eval_pos, calculate_feature_attention=True
                    )
                    x = x + residual
                elif idx == self._sequence_capture_idx and calculate_sample_attention and is_capture_layer:
                    x, _, sample_attention = sublayer(x, feature_atten_mask, eval_pos, calculate_sample_attention=True)
                    x = x + residual
                else:
                    if isinstance(sublayer, functools.partial):
                        x = sublayer(x, feature_atten_mask, eval_pos)
                        if isinstance(x, tuple):
                            x = x[0]
                        x = x + residual
                    else:
                        x = sublayer(x)
                        if isinstance(x, tuple):
                            x = x[0]
                        x = x + residual
                x = layer_norm(x)
        return x, feature_attention, sample_attention


class LayerStack(nn.Module):
    """
    A flexible container module similar to ``nn.Sequential`` that allows
    keyword arguments to be passed through to each layer.
    """

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.gradient_checkpointing = False

    def forward(self, x, **kwargs):
        n_layers = len(self.layers)
        feature_attention = None
        sample_attention = None
        for idx, layer in enumerate(self.layers):
            kwargs["layer_idx"] = idx
            kwargs["is_last_layer"] = idx == n_layers - 1
            if self.gradient_checkpointing and self.training:
                x, layer_feature_attention, layer_sample_attention = checkpoint(layer, x, use_reentrant=False, **kwargs)
            else:
                x, layer_feature_attention, layer_sample_attention = layer(x, **kwargs)
            if layer_feature_attention is not None:
                feature_attention = layer_feature_attention
            if layer_sample_attention is not None:
                sample_attention = layer_sample_attention
        return x, feature_attention, sample_attention

    def build_train_cache(self, x_train, **kwargs):
        """Build the per-layer train K/V caches for chunked inference.

        WS1 Stage 2 memory lever: with quantize/offload set, each layer's O(N)
        seq-K/V cache is int8-quantized and/or moved to host RAM *immediately*
        after that layer produces it — before the next layer runs. This is what
        actually bounds the GPU high-water-mark: otherwise all len(layers) caches
        accumulate full-precision on the GPU and the prefill peak is O(L*N), so a
        post-hoc offload (after the whole build) can't lower it. Doing it in the
        loop keeps at most ~one layer's cache resident during the build.
        """
        caches = []
        feature_atten_mask = kwargs.get("feature_atten_mask", None)
        quantize = kwargs.get("quantize_kv_cache", False)
        offload = kwargs.get("offload_kv_cache", False)
        device = kwargs.get("device", None)
        fit_row_chunk = kwargs.get("fit_row_chunk", None)
        stream_context = kwargs.get("stream_context", False)
        hybrid_resident_int8_prefill = kwargs.get("_hybrid_resident_int8_prefill", False)
        for layer in self.layers:
            x_train, layer_cache = layer.forward_train_cache(
                x_train,
                feature_atten_mask=feature_atten_mask,
                fit_row_chunk=fit_row_chunk,
                quantize_kv_cache=quantize,
                offload_kv_cache=offload,
                device=device,
                stream_context=stream_context,
                _hybrid_resident_int8_prefill=hybrid_resident_int8_prefill,
            )
            if (quantize or offload) and not stream_context and not hybrid_resident_int8_prefill:
                # Scale in place, per layer: the full-precision GPU tensor is
                # dropped as soon as the CPU/int8 copy is stored, so the next
                # layer builds against a freed allocator slot. (No-op if the
                # fit_row_chunk path already offloaded it -- scale_caches skips
                # an already-wrapped ScalableSeqKV.)
                scale_caches([layer_cache], quantize=quantize, offload=offload, device=device)
            caches.append(layer_cache)
        return x_train, caches

    def forward_test_with_cache(self, x_test, caches, **kwargs):
        feature_atten_mask = kwargs.get("feature_atten_mask", None)
        if len(caches) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} layer caches, got {len(caches)}")
        for layer, layer_cache in zip(self.layers, caches):
            x_test = layer.forward_test_with_cache(
                x_test,
                layer_cache,
                feature_atten_mask=feature_atten_mask,
            )
        return x_test
