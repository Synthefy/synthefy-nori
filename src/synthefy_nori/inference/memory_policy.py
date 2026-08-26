"""How much memory inference may use, and where the K/V cache lives.

Nori does in-context regression: the context table is *input*, not weights, so one
``predict`` call reads all N context rows and keeps a per-layer key/value cache that
every query row then attends to. That cache is O(nlayers x N) and stays resident for
the whole decode phase, which makes it the dominant serving cost at large N.

:class:`MemoryPolicy` is the single object describing what to do about that.
:meth:`MemoryPolicy.resolve` measures the request and returns **another
MemoryPolicy** with the decided fields concrete — same type in, same type out, so
there is no second "decision" type to keep straight.

Every field's declared default is its **real value**, not a sentinel: reading the
signature tells you what happens without looking anything up. The adaptive part is
expressed as *permissions* (``allow_quantization``, ``offload_to_host``) rather than
an opaque ``"auto"``, so "what is the default precision" and "may it change under
pressure" are two separate, individually readable questions.

The ladder, cheapest rung first. Each rung is used only when the one above it cannot
serve the request:

=================  ========================================================
rung               meaning
=================  ========================================================
no_cache           the cached path does not apply (e.g. the query set is small
                   enough that inference never chunks). Not a degradation.
resident_bf16      the full-precision cache fits the GPU budget. BIT-EXACT.
resident_int8      bf16 would not fit but int8 does, so the cache stays on the
                   fast on-GPU path instead of paying PCIe streaming. Costs
                   |dR2| ~ 6e-6 (per-(row, head) absmax quantization).
                   Requires ``allow_quantization``.
offload_bf16       cannot stay resident and quantizing is not allowed -> the
                   full-precision cache lives in host RAM. Bit-exact, slower.
offload_int8       cannot stay resident at any precision -> quantized cache in
                   host RAM, each layer's slice streamed back on demand.
context_row_chunk  an actual OOM happened -> re-run bounding the prefill
                   working set too (bit-exact; see layer.py).
plain_loop         nothing above worked. **No cache at all: every query chunk
                   recomputes the context K/V, so this is several times slower,
                   and if the context alone exceeds the element budget the
                   caller must SUBSAMPLE it — silently losing rows unless
                   ``allow_subsample=False`` turns that into an error.** The
                   predictor logs a warning naming the rung and, when rows are
                   dropped, how many.
=================  ========================================================

**Only the int8 rungs are lossy, and ``resident_int8`` is reached only when the
full-precision cache does not fit.** That ordering is the point: a table that serves
correctly today keeps bit-exact predictions, and accuracy is spent only to avoid a
fallback that would otherwise be slower or fatal. A measured cost for int8
(|dR2| = 5.8e-6) is not a licence to charge it to requests that had no memory
problem in the first place.

``context_row_chunk`` and ``plain_loop`` are *reactions* to an OutOfMemoryError, so the
caller escalates into them via :meth:`MemoryPolicy.escalated`; :meth:`resolve` picks
the opening rung.

Budgets are **fractions of the hardware** rather than absolute GB so one setting is
portable from a 24 GB laptop card to a 143 GB H200. The ``*_absolute_gb`` overrides
exist for the one case a fraction cannot express: a co-tenanted GPU, where what
matters is a hard ceiling rather than a share.

**This module reads no environment variables.** Configuration comes from
``NoriPredictor(memory_policy=...)`` / ``NoriRegressor(memory_policy=...)``; the only env var in
the whole path is the ``SYNTHEFY_DISABLE_CACHED_INFERENCE`` kill switch, handled by
the predictor. Resolution is pure arithmetic — no torch allocations, no device calls
— so every rung boundary is unit-testable on CPU rather than only exercised by
whoever happens to run a 500k-row table on a big card.
"""

from __future__ import annotations

import contextlib
import os
import threading
import warnings
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Named starting points, for callers who do not want to set fields individually.
#: Each one CHANGES something. There is deliberately no "auto"/"default" preset:
#: omitting ``memory_policy=`` already means the defaults, so naming that would just be a
#: second spelling of nothing.
MemoryPreset = Literal["exact", "max_context", "off"]
MEMORY_PRESETS: tuple[str, ...] = ("exact", "max_context", "off")

CacheDtype = Literal["bf16", "int8"]
CACHE_DTYPE_CHOICES: tuple[str, ...] = ("bf16", "int8")

RUNGS: tuple[str, ...] = (
    "no_cache",
    "resident_bf16",
    "resident_int8",
    "offload_bf16",
    "offload_int8",
    "context_row_chunk",
    "plain_loop",
)

# --- unit + shape constants (no bare numbers in the arithmetic below) ---------

#: Bytes per GiB, for every GB figure in this module.
BYTES_PER_GIB = 1024**3

#: The cache stores one key AND one value vector per (layer, group, row).
KV_TENSORS_PER_ROW = 2

#: int8 payload width.
INT8_BYTES_PER_ELEMENT = 1

#: Width of the fp32 absmax scale int8 quantization adds, one per head_dim vector.
#: Counting it keeps the resident threshold honest: against bf16 the real saving is
#: ~1.9x, not the 4x that holds only against fp32.
INT8_SCALE_BYTES = 4

# --- policy defaults ---------------------------------------------------------

#: Share of TOTAL VRAM the resident cache may occupy before we offload. 0.4 is the
#: aggressive ceiling: the co-resident non-cache working set measures ~0.72x the bf16
#: cache, which puts a resident peak near full VRAM at this fraction.
DEFAULT_GPU_BUDGET_FRAC = 0.4

#: Share of total physical RAM the offloaded cache may occupy.
DEFAULT_HOST_BUDGET_FRAC = 0.25

#: Fit-time row chunk engaged automatically after an OOM on the cached path.
FIT_ROW_CHUNK_ON_OOM = 2048

#: Fields that only mean anything while the cached path is in use. Asking for any of
#: them with ``cache=False`` is incoherent, so it is rejected rather than silently
#: dropped — the failure mode where a caller spells everything correctly and still
#: gets none of what they asked for.
CACHE_ONLY_FIELDS: tuple[str, ...] = (
    "cache_dtype",
    "allow_quantization",
    "offload_to_host",
    "context_row_chunk",
    # adaptive_query_chunk is passed only to forward_cached_regression, so it is
    # inert on the plain loop as well -- it belongs here for the same reason.
    "adaptive_query_chunk",
    # The budgets bound where the CACHE is placed. With no cache to place, tuning
    # them does nothing, which is the same silent no-op as the levers above.
    "gpu_budget_frac",
    "gpu_budget_absolute_gb",
    "host_budget_frac",
    "host_budget_absolute_gb",
)

#: Stand-in when the device cannot be introspected (CPU inference, or a device string
#: torch will not describe). Conservative on purpose; the OOM fallback is the real
#: safety net regardless of how good this estimate is.
ASSUMED_VRAM_GB = 24.0

#: Config warnings already emitted this process. These describe a *configuration*, not
#: a request, so saying it once is enough -- and the alternative is genuine spam:
#: resolve() runs once per inference pipeline (16 on the default config), and because
#: ``stacklevel`` attributes the warning to a varying caller frame, Python's own
#: per-location de-duplication never engages. Measured 32 copies of one warning from
#: two predict() calls before this existed.
_WARNED_ONCE: set[str] = set()


def warn_once(message: str) -> None:
    """Emit ``message`` as a UserWarning the first time it is seen this process.

    Args:
        message: the full warning text; identical text is emitted only once.
    """
    if message in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(message)
    # Prefixed because these fire inside pydantic validation, so Python attributes them
    # to pydantic/main.py rather than to the caller's line. The prefix keeps it obvious
    # whose warning this is; the text is self-contained about which fields are involved.
    warnings.warn(f"Nori memory policy: {message}", UserWarning, stacklevel=3)


class ContextTooLargeError(RuntimeError):
    """The context does not fit the element budget and may not be subsampled.

    A **caller** condition, not a server fault: the context size, ``elements_budget`` and
    ``allow_subsample`` are all things the caller chose, and changing any one of them
    makes the same request work. The message names which setting forbade the shrink and
    what to raise.

    Typed so that a server can tell it apart from the other ``RuntimeError`` that reaches
    the same place -- a CUDA OOM, which really is a server condition. Without the
    distinction a server has to choose between turning its own faults into 4xx or
    reporting a caller's own configuration back to them as a 500 with the remedy buried
    in it.

    Subclasses ``RuntimeError`` because that is what this used to raise, so code already
    catching ``RuntimeError`` keeps working.
    """


#: The one field family a shared server must bound rather than honour verbatim, named
#: here beside the fields themselves so a serving target cannot go stale: adding a
#: budget field means editing this tuple, not every server.
HOST_BUDGET_FIELDS: tuple[str, ...] = ("host_budget_frac", "host_budget_absolute_gb")

#: Field a shared server must FLOOR rather than honour verbatim. Unlike the budgets above,
#: whose cost is memory, this one's cost is TIME: it is the step size of the prefill loop
#: that builds the K/V cache, so ``context_row_chunk=1`` turns one request into O(N) kernel
#: launches per layer over the caller's own context rows. That runs while holding the
#: inference lock, so it delays every other caller rather than only its own -- which is
#: exactly the "hurts nobody else" premise the other fields rely on, and the reason this
#: one cannot ride along with them.
CONTEXT_ROW_CHUNK_FIELD = "context_row_chunk"

#: Guards the warning capture in :func:`capture_policy_notes`.
#: ``warnings.catch_warnings`` swaps process-global filter state, so two callers
#: capturing at once would otherwise cross-attribute each other's notes.
#:
#: Reentrant because a caller legitimately nests: a server holds one capture across a
#: whole request and :meth:`MemoryPolicy.coerce_for_service` opens its own inside that.
#: A plain Lock deadlocks on that nesting -- and it deadlocks while holding the
#: inference lock, so the replica stops answering rather than failing one request.
_WARNING_CAPTURE_LOCK = threading.RLock()


@contextlib.contextmanager
def capture_policy_notes():
    """Collect the policy warnings emitted in this block, instead of logging them.

    A memory-policy warning is *advice to whoever set the policy*. In a script that is
    the person reading stderr, so :func:`warn_once` is right. In a server the person
    who set it is on the other end of an HTTP request and never sees stderr, so the
    warnings have to be captured and returned -- as ``memory_report.notes``.

    Yields the list the notes accumulate into, so a caller reads it after the block:

        with capture_policy_notes() as notes:
            ...
        return tuple(notes)

    Two things this handles that an inline ``catch_warnings`` does not:

    - **Per-request de-duplication.** :func:`warn_once` suppresses repeats for the
      life of the process, which is right for a script and wrong for a server: the
      first caller to misconfigure something absorbs the only warning and everyone
      after them is told nothing about their own policy. So the registry is cleared
      on entry.
    - **Serialisation.** The filter state this swaps is process-global, so overlapping
      captures cross-attribute. Note the lock alone is NOT sufficient for a server:
      it serialises captures against each other, but not a capture against a
      concurrent forward pass, which also emits warnings. A server must therefore
      hold this INSIDE its inference lock -- see ``serving/nori_serving/engine.py``.

    Yields:
        The list of warning messages, filled as the block runs.
    """
    with _WARNING_CAPTURE_LOCK:
        forget_emitted_warnings()
        notes: list = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                yield notes
            finally:
                # In the finally so a raising block still reports the warnings that
                # preceded it: an incoherent policy is exactly the case where the
                # notes explain the error.
                notes.extend(str(entry.message) for entry in caught)


def _as_pydantic_would_coerce(value) -> "float | None":
    """The float pydantic's lax mode will produce for ``value``, or None if it will not.

    Exists because clamping has to reason about the value that ENDS UP in the model, and
    an ``isinstance(value, (int, float))`` test does not: pydantic v2 without
    ``strict=True`` accepts ``"0.95"`` and ``True`` and turns them into ``0.95`` and
    ``1.0``. Bools are deliberately included rather than skipped -- ``True`` became 100%
    of host RAM, unclamped and reported as honoured.
    """
    if value is None or isinstance(value, (list, tuple, dict, set)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def forget_emitted_warnings() -> None:
    """Clear the once-per-process registry behind :func:`warn_once`.

    Process-lifetime de-duplication is right for a script or a notebook: one copy of
    "this budget cannot bite", not one per inference chunk. It is wrong for a
    **long-lived server**, where each request carries a different caller's config —
    the first caller to make a mistake absorbs the only warning, and every caller
    after them is told nothing about their own. Worse, the warning goes to the
    server's log, not to the caller who could act on it.

    So a server calls this once per request, captures the warnings that follow, and
    returns them to that request. ``serving/nori_serving/engine.py`` does exactly
    that, surfacing them as ``memory_report.notes``. Tests use it for the same reason
    in miniature: without it, a ``pytest.warns`` assertion passes or fails depending
    on what an earlier test in the session already emitted.

    Safe to call at any time; it only forgets what has been *said*, never any
    configuration.
    """
    _WARNED_ONCE.clear()


def estimate_cache_gb(
    *,
    n_context_rows: int,
    n_groups: int,
    nlayers: int,
    embed_dim: int,
    bytes_per_element: int,
) -> float:
    """Estimate the stored key/value cache footprint in GiB at full precision.

    Args:
        n_context_rows: context (train) rows; the N the cache scales in.
        n_groups: feature groups the model splits the columns into.
        nlayers: transformer layers, each keeping its own cache.
        embed_dim: model embedding width.
        bytes_per_element: 2 under mixed precision, 4 in fp32.

    Returns:
        Footprint in GiB.
    """
    elements = nlayers * n_groups * n_context_rows * KV_TENSORS_PER_ROW * embed_dim
    return (elements * bytes_per_element) / BYTES_PER_GIB


def int8_footprint_gb(est_cache_gb: float, *, bytes_per_element: int, head_dim: int) -> float:
    """Footprint of the same cache stored int8, including its scale tensor.

    Args:
        est_cache_gb: full-precision footprint from :func:`estimate_cache_gb`.
        bytes_per_element: bytes per element in that full-precision figure.
        head_dim: ``embed_dim // nhead``; the vector length each scale covers.

    Returns:
        Footprint in GiB.
    """
    payload_gb = est_cache_gb * (INT8_BYTES_PER_ELEMENT / max(bytes_per_element, 1))
    scale_ratio = INT8_SCALE_BYTES / max(head_dim, 1)
    return payload_gb * (1.0 + scale_ratio)


#: cgroup v2 then v1. A container's memory ceiling lives here and NOWHERE that
#: ``sysconf`` can see.
_CGROUP_LIMIT_PATHS = (
    "/sys/fs/cgroup/memory.max",  # v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # v1
)

#: v1 spells "unlimited" as a huge sentinel rather than a word. Anything at or above
#: this is "no limit set", not a 8-exabyte container.
_CGROUP_UNLIMITED_SENTINEL = 1 << 62


def _physical_ram_gb() -> float:
    """What the KERNEL reports, which inside a container is the whole node's RAM."""
    if not hasattr(os, "sysconf"):
        return 0.0
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 0.0
    if not page_size or not page_count or page_size < 0 or page_count < 0:
        return 0.0
    return (page_size * page_count) / BYTES_PER_GIB


def _cgroup_memory_limit_gb() -> float:
    """The container's own memory ceiling in GiB, or 0.0 if there is not one.

    Read because ``sysconf("SC_PHYS_PAGES")`` derives from the host's ``MemTotal`` and
    is **not** namespaced by any container runtime — so a process in a 234 GiB cgroup on
    a 2 TiB node sees 2 TiB. Budgeting against that number is how you get SIGKILLed: the
    cgroup limit is the one the kernel actually enforces, and exceeding it is not a
    catchable error.
    """
    for path in _CGROUP_LIMIT_PATHS:
        try:
            with open(path) as limit_file:
                raw = limit_file.read().strip()
        except (OSError, ValueError):
            continue
        if raw == "max":  # v2's "no limit"
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= _CGROUP_UNLIMITED_SENTINEL:
            continue
        return value / BYTES_PER_GIB
    return 0.0


def total_host_ram_gb() -> float:
    """Host RAM this process may actually use, in GiB, or 0.0 if unknowable.

    The **smaller** of what the kernel reports and what the container's cgroup allows.
    Both are needed: `sysconf` alone over-reports inside a container (it is not
    namespaced), and the cgroup limit alone is absent when there is no limit set.

    Returns 0.0 rather than raising, so an unknown platform degrades to "no host
    budget" — never offload — instead of inventing a number and getting OOM-killed.
    """
    physical = _physical_ram_gb()
    limit = _cgroup_memory_limit_gb()
    if physical and limit:
        return min(physical, limit)
    return limit or physical


class MemoryPolicy(BaseModel):
    """What inference may spend on memory, and where the K/V cache goes.

    One object covers the whole ladder (see the module docstring)::

        MemoryPolicy()                                        # the defaults below
        MemoryPolicy.coerce("exact")                          # never quantize
        MemoryPolicy.coerce({"gpu_budget_frac": 0.25})        # e.g. parsed from YAML
        MemoryPolicy(cache_dtype="int8", context_row_chunk=2048)

    ``extra="forbid"`` is deliberate: a mistyped key on a knob whose whole job is "do
    not lose accuracy" must fail loudly rather than be ignored forever.
    ``frozen=True`` makes instances hashable and safe to share — :meth:`resolve`
    returns a new object rather than mutating.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cache: bool = Field(
        True,
        description="Build a key/value cache over the context rows at all. False re-reads "
        "the whole context for every batch of query rows instead: correct, but "
        "several times slower on a large query set, and it may have to drop "
        "context rows to fit.",
    )
    reuse_context_cache: bool = Field(
        True,
        description="Retain and reuse the encoded context across separate predict() calls "
        "on this estimator when the context and cache parameters are exactly "
        "unchanged. False still uses the K/V cache within each prediction, but "
        "does not retain context-derived state afterwards. Shared serving "
        "processes force this off; local fit-once/predict-many use keeps it on.",
    )
    cache_dtype: CacheDtype = Field(
        "bf16",
        description="Precision the K/V cache STARTS at. bf16 is bit-exact and is the "
        "default; set 'int8' to quantize from the outset (~1.9x smaller, "
        "|dR2| ~ 6e-6) when you would rather trade that for context. "
        "Whether bf16 may be downgraded under memory pressure is a "
        "separate question — see allow_quantization.",
    )
    allow_quantization: bool = Field(
        True,
        description="May the cache be quantized to int8 when full precision would "
        "NOT stay resident? This is the only way int8 gets used by "
        "default, and it is strictly better than offloading (which costs "
        "40-175% latency). False keeps every rung bit-exact — the "
        "'exact' preset — at the cost of offloading sooner.",
    )
    gpu_budget_frac: float = Field(
        DEFAULT_GPU_BUDGET_FRAC,
        gt=0,
        le=1,
        description="Share of TOTAL VRAM the resident cache may occupy before we "
        "offload. A fraction so one setting is portable across GPUs. "
        "Total rather than free VRAM: free is racy, and budgeting against "
        "it would make the same call take different rungs on different "
        "runs.",
    )
    gpu_budget_absolute_gb: float | None = Field(
        None,
        ge=0,
        description="Hard VRAM ceiling in GiB, overriding gpu_budget_frac. For a "
        "co-tenanted GPU, where a fixed cap is the requirement and a "
        "share of the card is the wrong unit. None = use the fraction; "
        "0 = never keep the cache in GPU memory. 0 is deliberately "
        "legal, not a typo guard: it is a real request, and it is also "
        "the value the resolved budget can take.",
    )
    offload_to_host: bool = Field(
        True,
        description="May the cache be moved to host RAM (streamed back per layer) "
        "when it cannot stay resident? Bit-exact transport, but 40-175% "
        "slower. False reproduces the legacy behaviour of skipping the "
        "cache entirely instead, which is slower still.",
    )
    host_budget_frac: float = Field(
        DEFAULT_HOST_BUDGET_FRAC,
        gt=0,
        le=1,
        description="Share of total physical RAM the offloaded cache may occupy. A "
        "fraction because a flat GB default is a latent bug: 128 GB "
        "'fits' on a 32 GB laptop by arithmetic, so offload proceeds and "
        "the kernel OOM-kills the process instead of the policy falling "
        "to the plain loop.",
    )
    host_budget_absolute_gb: float | None = Field(
        None,
        ge=0,
        description="Hard host-RAM ceiling in GiB, overriding host_budget_frac. "
        "None = use the fraction; 0 = never offload. 0 is deliberately "
        "legal: it is also what a platform that will not report its RAM "
        "resolves to, and that case has to degrade to 'no offload' rather "
        "than fail.",
    )
    context_row_chunk: int | None = Field(
        None,
        gt=0,
        description="Bound the fit-time build working set to this many context rows "
        f"(bit-exact). None = off, with {FIT_ROW_CHUNK_ON_OOM} engaged "
        "automatically after an OOM. Pinning a value uses it from the "
        "first attempt — which also costs you that escalation, since "
        "there is then nothing left to escalate to.",
    )
    adaptive_query_chunk: bool = Field(
        True,
        description="On a decode OOM, halve the QUERY chunk and retry instead of "
        "raising. Distinct from context_row_chunk, which caps CONTEXT rows "
        "during prefill.",
    )
    elements_budget: int | None = Field(
        None,
        gt=0,
        description="Per-forward element cap driving query chunk size and context "
        "subsampling. None = derive it from available VRAM. Upstream of "
        "everything else here: the cached path engages only when the "
        "query set exceeds the chunk size this implies, so raising it far "
        "enough disables the cache knobs by making inference stop "
        "chunking at all.",
    )
    allow_subsample: bool = Field(
        True,
        description="Permit dropping context rows when the request will not fit "
        "otherwise. False makes that an error instead of a silent "
        "accuracy loss.",
    )

    # --- populated by resolve(); None on an unresolved policy ----------------
    rung: str | None = Field(
        None,
        description="Which fallback the server used, in decreasing memory cost: "
        "resident_bf16 (cache in GPU memory, exact), resident_int8 (quantized), "
        "offload_bf16 / offload_int8 (cache in host RAM, streamed back per layer, "
        "exact transport), context_row_chunk (cache built in row chunks), "
        "plain_loop (no cache; the context re-read per query batch), no_cache "
        "(the cached path did not apply). Decided per request, not requestable.",
    )
    est_cache_gb: float | None = Field(
        None, ge=0, description="Full-precision cache footprint for this request's context, in GiB. Reported, not set."
    )
    resident_gb: float | None = Field(
        None,
        ge=0,
        description="Footprint at the chosen precision (GiB) — the figure actually "
        "compared against the budget. Reported, not set.",
    )
    query_chunk: int | None = Field(
        None,
        gt=0,
        description="Query rows per decode forward, as the cached path was entered "
        "with. Reported, not set. NOTE: a decode OOM may halve this "
        "further inside the model, and that further reduction is not "
        "currently reported here.",
    )
    dropped_context_rows: int = Field(
        0,
        ge=0,
        description="Context rows the caller had to subsample away to fit. Non-zero "
        "only on the plain_loop rung; recorded so a shrunk context is "
        "visible in memory_report_ rather than inferred from the score.",
    )

    @model_validator(mode="after")
    def _check_rung(self) -> "MemoryPolicy":
        """Reject an unrecognised rung so a typo cannot masquerade as a decision."""
        if self.rung is not None and self.rung not in RUNGS:
            raise ValueError(f"rung must be one of {RUNGS}, got {self.rung!r}")
        return self

    @model_validator(mode="after")
    def _check_coherent(self) -> "MemoryPolicy":
        """Reject a request that asks for cache-only levers with the cache off.

        Only applies BEFORE resolution. While ``rung is None`` these fields are a
        *request* the caller wrote, and ``cache=False`` plus (say) ``context_row_chunk``
        is a contradiction: fit-time row chunking bounds the prefill build, which only
        happens on the cached path, so the request could never be honoured. Once
        :meth:`resolve` has run the same fields are *outputs* — a ``plain_loop``
        policy legitimately reports ``cache=False`` alongside a concrete dtype — so
        the check is skipped.

        Uses ``model_fields_set`` so only levers the caller EXPLICITLY set count:
        ``offload_to_host`` defaults to True, and ``MemoryPolicy(cache=False)`` (the
        ``"off"`` preset) must stay legal.

        Returns:
            ``self`` when coherent.

        Raises:
            ValueError: naming every field that could not be honoured.
        """
        if self.rung is not None or self.cache:
            return self
        if "reuse_context_cache" in self.model_fields_set and self.reuse_context_cache:
            raise ValueError(
                "cache=False cannot be combined with reuse_context_cache=True: "
                "there is no K/V cache to retain across calls. Set "
                "reuse_context_cache=False or remove the field."
            )
        requested = sorted(f for f in CACHE_ONLY_FIELDS if f in self.model_fields_set)
        if requested:
            raise ValueError(
                f"cache=False cannot be combined with {requested}: those only affect "
                f"the cached path, so nothing would honour them. In particular "
                f"context_row_chunk caps the K/V build, which only runs when "
                f"the cache is on — 'row chunking without KV caching' is not a "
                f"reachable configuration. Either drop {requested}, or set cache=True "
                f"and use memory_policy={{'context_row_chunk': N}} to cap the build."
            )
        return self

    @model_validator(mode="after")
    def _check_no_dead_settings(self) -> "MemoryPolicy":
        """Warn where one setting makes another unreachable.

        These are WARNINGS, not errors, unlike :meth:`_check_coherent`. The split
        follows one rule: **error when we cannot know what the caller meant, or when
        guessing wrong would cost accuracy; warn when intent is clear and the extra
        setting is merely inert.** A layered config (base YAML sets a fraction, a
        per-run override sets an absolute) reaches these innocently, and refusing it
        would make layering unusable for no safety gain.

        Same principle as :meth:`_check_coherent`, for pairs rather than the cache
        switch: a value that cannot take effect is rejected instead of silently
        ignored. Skipped after resolution, where these fields are outputs.

        Returns:
            ``self`` when every setting can take effect.

        Raises:
            ValueError: naming the pair and which of the two to drop.
        """
        if self.rung is not None:
            return self
        given = self.model_fields_set
        for frac, absolute in (
            ("gpu_budget_frac", "gpu_budget_absolute_gb"),
            ("host_budget_frac", "host_budget_absolute_gb"),
        ):
            if frac in given and absolute in given:
                warn_once(
                    f"{frac} is ignored because {absolute} is also set and takes "
                    f"precedence. Set only one: the fraction for a config that travels "
                    f"between GPUs, the absolute for a hard cap on a shared one.",
                )
        if "offload_to_host" in given and not self.offload_to_host:
            dead = sorted(f for f in ("host_budget_frac", "host_budget_absolute_gb") if f in given)
            if dead:
                warn_once(
                    f"{dead} is ignored because offload_to_host=False disables host "
                    f"offload entirely, so no host budget is ever consulted.",
                )
        # A cache that is configured never to be usable: statically decidable, so say
        # so rather than letting every call quietly take the slowest rung.
        if self.cache and not self.offload_to_host and self.gpu_budget_absolute_gb == 0:
            warn_once(
                "cache=True but gpu_budget_absolute_gb=0 with offload_to_host=False "
                "means the cache can never be placed, so every call will take the "
                "plain_loop rung (several times slower, and it may subsample "
                "context). Raise the GPU budget or allow host offload.",
            )
        # Rule 6, the half that is decidable without hardware: when BOTH budgets are
        # absolute, we can compare them now. Offload only engages above the GPU budget
        # and only succeeds within the host budget, so a host budget at or below the
        # GPU budget makes offload unreachable. The fractional case depends on the
        # box, so resolve() carries the same check for it.
        if (
            self.offload_to_host
            and self.gpu_budget_absolute_gb is not None
            and self.host_budget_absolute_gb is not None
            and self.host_budget_absolute_gb <= self.gpu_budget_absolute_gb
        ):
            warn_once(
                f"offload_to_host=True cannot engage: host_budget_absolute_gb="
                f"{self.host_budget_absolute_gb} is not above gpu_budget_absolute_gb="
                f"{self.gpu_budget_absolute_gb}, so any cache too big to stay resident "
                f"is also too big to offload. Raise the host budget above the GPU "
                f"budget for offload to be reachable.",
            )
        if "allow_quantization" in given and not self.allow_quantization and self.cache_dtype == "int8":
            raise ValueError(
                "allow_quantization=False with cache_dtype='int8' is contradictory: "
                "the first forbids quantizing, the second asks for a quantized cache "
                "outright. Use cache_dtype='bf16' with allow_quantization=False to "
                "stay bit-exact, or cache_dtype='int8' alone to start quantized."
            )
        return self

    def _revalidated_copy(self, **update) -> "MemoryPolicy":
        """Copy with ``update`` applied, re-running validation.

        ``model_copy(update=...)`` does NOT re-validate, so writing a bad rung or a
        negative budget through it would succeed silently and the validators above
        would be decorative on the only path that actually uses them.

        Args:
            **update: field values to overwrite.

        Returns:
            The validated copy.
        """
        return type(self).model_validate({**self.model_dump(), **update})

    # ------------------------------------------------------------------ build

    @classmethod
    def coerce(cls, value: "MemoryPreset | dict | MemoryPolicy | None") -> "MemoryPolicy":
        """Build a policy from a preset name, a dict, an instance, or None.

        Args:
            value: ``None`` for the defaults; ``"exact"``, ``"max_context"`` or
                ``"off"`` for a named starting point; a dict of field values (e.g.
                parsed from a config file); or an existing policy, returned unchanged.

        Returns:
            The policy.

        Raises:
            ValueError: on an unknown preset name.
            TypeError: on any other type.
            pydantic.ValidationError: on an unknown or invalid dict key.
        """
        if isinstance(value, MemoryPolicy):
            if value.rung is not None:
                raise ValueError(
                    f"memory_policy= was given an already-RESOLVED policy (rung="
                    f"{value.rung!r}). Its decided outputs would be re-used as inputs, "
                    f"and a policy carrying a rung skips every coherence check. Pass a "
                    f"fresh MemoryPolicy with only the fields you want to set; "
                    f"memory_report_ is for inspection, not for feeding back in."
                )
            return value
        if value is None:
            return cls()
        if isinstance(value, dict):
            if value.get("rung") is not None:
                raise ValueError(
                    f"memory_policy= was given a resolved report (rung={value['rung']!r}), "
                    f"probably a memory_report_ dict. Those are outputs; feeding them "
                    f"back in re-uses decided values as configuration and skips every "
                    f"coherence check. Pass only the fields you want to set."
                )
            return cls(**value)
        if isinstance(value, str):
            if value not in MEMORY_PRESETS:
                raise ValueError(
                    f"unknown memory preset {value!r}; expected one of {MEMORY_PRESETS}, a dict, or a MemoryPolicy"
                )
            if value == "exact":
                # Never trade accuracy: offload (bit-exact) rather than quantize.
                return cls(allow_quantization=False)
            if value == "max_context":
                # Fit the largest table possible; start quantized to free VRAM at once.
                return cls(cache_dtype="int8")
            return cls(cache=False, reuse_context_cache=False)  # "off"
        raise TypeError(f"memory must be a preset name, dict, MemoryPolicy or None, got {type(value).__name__}")

    @classmethod
    def coerce_for_service(
        cls,
        value: "MemoryPreset | dict | MemoryPolicy | None",
        *,
        max_host_budget_frac: float,
        min_context_row_chunk: int | None = None,
        total_ram_gb: float | None = None,
    ) -> "tuple[MemoryPolicy | None, tuple[str, ...], tuple[str, ...]]":
        """Coerce a policy that arrived from a caller who does not own this process.

        :meth:`coerce` is enough when the policy was written by whoever runs the
        process: a bad value is their own problem, and a warning goes to a terminal
        they are looking at. A **server** taking policies from requests needs three
        more things, and all three depend on this module's own fields — which is why
        they live here rather than in each serving target:

        1. **Bound the host-RAM budgets** (:data:`HOST_BUDGET_FIELDS`). Every other
           field spends only the requesting caller's GPU memory, so overspending is
           self-inflicted and passed through. Host RAM is different in kind:
           exceeding the container's cgroup limit is a SIGKILL, not a catchable
           error, so it takes down the replica and charges the *next* caller a cold
           start. Clamped values are returned by name so the server can report them
           instead of applying them silently.
        2. **Clamp before validating**, so the coherence rules below run against the
           numbers that will actually be used — a warning about a budget that cannot
           bite is only true of the clamped figure.
        3. **Capture the warnings for this caller.** :func:`warn_once` de-duplicates
           for the life of the process, so on a long-lived server the first caller to
           make a given mistake absorbs the only copy and everyone after them is told
           nothing — and it goes to the server's log either way, never to the person
           who could act on it. The registry is cleared and the warnings collected
           per call, for the server to return in its response.

        Args:
            value: what the request carried, unchanged. ``None`` means the request
                said nothing, and is returned as ``None`` rather than as the default
                policy, so a server can tell "not asked for" from "asked for the
                defaults" and keep its response unchanged in the first case.
            max_host_budget_frac: the largest share of host RAM this deployment will
                grant. A deployment-shaped decision (it depends on what else lives in
                the container), so it has no default here.
            min_context_row_chunk: the smallest prefill step this deployment will run.
                ``None`` leaves it unbounded, which is right in-process and wrong on a
                shared server: a tiny value costs TIME rather than memory, and it spends
                that time holding the inference lock. Also deployment-shaped, hence no
                default.
            total_ram_gb: host RAM to size an absolute budget against.
                ``None`` measures it with :func:`total_host_ram_gb`, which is
                cgroup-aware — a container sees its limit, not the machine's.

        Returns:
            ``(policy, clamped_field_names, notes)``. ``policy`` is None only when
            ``value`` was. ``notes`` are the coherence warnings, already formatted.

        Raises:
            ValueError, TypeError: exactly as :meth:`coerce` — the caller maps them
                onto its own status code, keeping the message, which already names
                the offending field.
        """
        if value is None:
            return None, (), ()

        clamped: list[str] = []
        ceilings = {
            "host_budget_frac": max_host_budget_frac,
            "host_budget_absolute_gb": max_host_budget_frac
            * (total_host_ram_gb() if total_ram_gb is None else total_ram_gb),
        }
        if isinstance(value, dict):
            value = dict(value)  # never mutate the caller's parsed request body
            for field in HOST_BUDGET_FIELDS:
                numeric = _as_pydantic_would_coerce(value.get(field))
                # Compare what VALIDATION will produce, not what the JSON happens to look
                # like. An isinstance check here let `"0.95"` and `true` straight through:
                # pydantic runs in lax mode, so it coerces both to floats AFTER the clamp
                # has already skipped them -- and `clamped` then came back empty, which the
                # response documents as "honoured verbatim". A quote mark defeated the rail.
                if numeric is None:
                    continue  # not numeric at all -> pydantic's error
                if numeric > ceilings[field]:
                    value[field] = ceilings[field]
                    clamped.append(field)

            if min_context_row_chunk is not None:
                # A FLOOR, not a ceiling: small is the dangerous direction here, because
                # the cost is one prefill step per `context_row_chunk` rows and the loop
                # holds the lock. Same reporting as the budgets -- capped, never silent.
                given = _as_pydantic_would_coerce(value.get(CONTEXT_ROW_CHUNK_FIELD))
                if given is not None and given < min_context_row_chunk:
                    value[CONTEXT_ROW_CHUNK_FIELD] = min_context_row_chunk
                    clamped.append(CONTEXT_ROW_CHUNK_FIELD)
        # A preset needs no clamping: none of them sets a host budget (they set
        # allow_quantization / cache_dtype / cache), so the safe default fraction
        # applies. Asserted by a test, so a new preset cannot quietly break it.

        # Construction-time warnings only -- coherence checks that pydantic runs during
        # validation. The ones resolve() emits fire later, during predict, and a server
        # that wants those too holds its own capture across the whole request.
        with capture_policy_notes() as notes:
            policy = cls.coerce(value)
        return policy, tuple(clamped), tuple(notes)

    # --------------------------------------------------------------- resolve

    def gpu_budget(self, total_vram_gb: float) -> float:
        """GPU budget in GiB: the absolute override if set, else the fraction."""
        if self.gpu_budget_absolute_gb is not None:
            return self.gpu_budget_absolute_gb
        return self.gpu_budget_frac * total_vram_gb

    def host_budget(self, total_ram_gb: float) -> float:
        """Host budget in GiB: the absolute override if set, else the fraction."""
        if self.host_budget_absolute_gb is not None:
            return self.host_budget_absolute_gb
        return self.host_budget_frac * total_ram_gb

    @property
    def is_resolved(self) -> bool:
        """Whether a rung has been chosen for a specific request."""
        return self.rung is not None

    @property
    def is_bit_exact(self) -> bool:
        """Whether the cache is stored losslessly.

        Offload moves bytes without changing them, so only precision decides this.
        """
        return self.cache_dtype != "int8"

    @property
    def is_degraded(self) -> bool:
        """Whether this rung is a fallback rather than the intended fast path.

        ``no_cache`` is not degraded — it means the request never needed a cache.
        Used by the predictor to decide whether to log at warning level.
        """
        return self.rung in ("offload_bf16", "offload_int8", "context_row_chunk", "plain_loop")

    def resolve(
        self,
        *,
        est_cache_gb: float,
        bytes_per_element: int,
        head_dim: int,
        total_vram_gb: float | None = None,
        total_ram_gb: float | None = None,
        cache_eligible: bool = True,
    ) -> "MemoryPolicy":
        """Decide the opening rung for one request, returning a concrete policy.

        Args:
            est_cache_gb: full-precision cache footprint, from
                :func:`estimate_cache_gb`.
            bytes_per_element: bytes per element behind ``est_cache_gb``.
            head_dim: ``embed_dim // nhead``, for the int8 scale overhead.
            total_vram_gb: total VRAM; defaults to :data:`ASSUMED_VRAM_GB` when the
                device cannot be introspected.
            total_ram_gb: total host RAM; defaults to a probe that yields 0.0 — and
                therefore no offload — when the platform will not say.
            cache_eligible: False when the cached path cannot apply at all (for
                example the query set is small enough that inference never chunks),
                giving rung ``no_cache``.

        Returns:
            A new :class:`MemoryPolicy` with ``rung``, ``cache``, ``cache_dtype``,
            ``offload_to_host``, both absolute budgets, ``est_cache_gb`` and
            ``resident_gb`` concrete.
        """
        vram = ASSUMED_VRAM_GB if total_vram_gb is None else total_vram_gb
        ram = total_host_ram_gb() if total_ram_gb is None else total_ram_gb
        gpu_budget_gb = self.gpu_budget(vram)
        host_budget_gb = self.host_budget(ram)
        bf16_gb = est_cache_gb
        int8_gb = int8_footprint_gb(est_cache_gb, bytes_per_element=bytes_per_element, head_dim=head_dim)

        def decided(
            rung: str, dtype: str, offload: bool, resident: float, cache: bool, budgets: bool = True
        ) -> "MemoryPolicy":
            return self._revalidated_copy(
                cache=cache,
                cache_dtype=dtype,
                offload_to_host=offload,
                reuse_context_cache=bool(cache and self.reuse_context_cache),
                gpu_budget_absolute_gb=gpu_budget_gb if budgets else None,
                host_budget_absolute_gb=host_budget_gb if budgets else None,
                rung=rung,
                est_cache_gb=est_cache_gb,
                resident_gb=resident,
            )

        if not self.cache or not cache_eligible:
            # Not eligible, or switched off: nothing to place, and no budget was
            # consulted. Report the budgets as None rather than a number computed
            # from a VRAM figure nobody looked at -- a diagnostic that states
            # "budget 9.6 GiB" on an 80 GB card is worse than one that says nothing.
            return decided("no_cache", self.cache_dtype, False, 0.0, cache=False, budgets=False)

        # Precisions this request may use, cheapest-accuracy-cost first. Starting at
        # int8 means bf16 is not a candidate at all — the caller asked for the smaller
        # cache; allow_quantization only governs DOWNGRADING from bf16.
        if self.cache_dtype == "int8":
            candidates = [("int8", int8_gb)]
        elif self.allow_quantization:
            candidates = [("bf16", bf16_gb), ("int8", int8_gb)]
        else:
            candidates = [("bf16", bf16_gb)]

        for dtype, footprint in candidates:
            if footprint <= gpu_budget_gb:
                return decided(f"resident_{dtype}", dtype, False, footprint, cache=True)

        if self.offload_to_host:
            # Offload the LARGEST (i.e. most precise) candidate host RAM can hold, not
            # the smallest. Offload transport is bit-exact at either precision, so
            # quantizing here buys only PCIe bandwidth -- and spending accuracy for
            # speed is exactly what this ladder exists not to do. int8 is used only
            # when bf16 will not fit host either. Callers who would rather have the
            # faster stream ask for it: cache_dtype="int8" or memory_policy="max_context".
            for dtype, footprint in candidates:
                if footprint <= host_budget_gb:
                    return decided(f"offload_{dtype}", dtype, True, footprint, cache=True)

        # Rule 6: we needed a fallback and offload could not provide one. Warn HERE
        # rather than up-front: a host budget below the GPU budget makes offload
        # unreachable, but that only matters once a request actually overflows VRAM.
        # Warning at the top of resolve() fired on untouched defaults for anyone whose
        # RAM is under ~1.6x their VRAM, for a fallback their request never reached.
        # Do not repeat what construction already said. When BOTH budgets are absolute,
        # rule 6a compared them at construction time and warned there; saying it again
        # here makes one root cause produce two warnings.
        already_warned_at_construction = (
            self.gpu_budget_absolute_gb is not None and self.host_budget_absolute_gb is not None
        )
        if self.offload_to_host and host_budget_gb <= gpu_budget_gb and not already_warned_at_construction:
            warn_once(
                f"This request needed to spill out of VRAM, but offload_to_host could "
                f"not help: the host budget ({host_budget_gb:.1f} GiB) is not larger "
                f"than the GPU budget ({gpu_budget_gb:.1f} GiB), so a cache too big to "
                f"stay resident is also too big to offload. Falling back to the plain "
                f"loop. Raise host_budget_frac / host_budget_absolute_gb above the GPU "
                f"budget to make offload reachable.",
            )
        worst_dtype, worst_footprint = candidates[-1]
        return decided("plain_loop", worst_dtype, False, worst_footprint, cache=False)

    def escalated(
        self,
        rung: str,
        *,
        context_row_chunk: int | None = None,
        dropped_context_rows: int | None = None,
        query_chunk: int | None = None,
    ) -> "MemoryPolicy":
        """Return a copy recording an escalation the caller made after an OOM.

        Args:
            rung: the rung reached, e.g. ``"context_row_chunk"`` or ``"plain_loop"``.
            context_row_chunk: the chunk actually used, when applicable.
            dropped_context_rows: context rows subsampled away, when applicable.
            query_chunk: query rows per decode forward, when applicable.

        Returns:
            The updated policy.
        """
        updates: dict[str, object] = {"rung": rung}
        if context_row_chunk is not None:
            updates["context_row_chunk"] = context_row_chunk
        if dropped_context_rows is not None:
            updates["dropped_context_rows"] = dropped_context_rows
        if query_chunk is not None:
            updates["query_chunk"] = query_chunk
        if rung == "plain_loop":
            updates["cache"] = False
        return self._revalidated_copy(**updates)

    def describe(self) -> str:
        """One-line human-readable summary, for logs and warnings.

        Returns:
            e.g. ``"resident_int8 (cache 48.8 GiB -> 25.9 GiB int8, GPU budget
            31.8 GiB, host budget 503.9 GiB)"``.
        """
        if not self.is_resolved:
            return "unresolved"
        parts = [self.rung]
        if self.est_cache_gb and self.gpu_budget_absolute_gb is not None:
            parts.append(
                f"(cache {self.est_cache_gb:.1f} GiB -> {self.resident_gb:.1f} GiB "
                f"{self.cache_dtype}, GPU budget {self.gpu_budget_absolute_gb:.1f} GiB, "
                f"host budget {self.host_budget_absolute_gb:.1f} GiB)"
            )
        if self.context_row_chunk:
            parts.append(f"context_row_chunk={self.context_row_chunk}")
        if self.dropped_context_rows:
            parts.append(f"DROPPED {self.dropped_context_rows} context rows")
        return " ".join(parts)
