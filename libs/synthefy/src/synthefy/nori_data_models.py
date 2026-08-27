"""Lightweight Nori client/serving data models.

``MemoryPolicy`` mirrors the authoritative model in ``synthefy-nori``. The
large-context types below keep the client transport-neutral: local mode may forward
a Python callable, while hosted modes send a policy-name string for the server's
authoritative resolver to validate.

**This is a copy, and the original is authoritative.** The policy is defined by
``synthefy_nori.inference.memory_policy.MemoryPolicy``, in the repo that also runs the
serving code, and that model is what actually validates a request. This file exists so a
client user gets a typed, documented object without installing ``synthefy-nori`` — which
would pull torch, numpy, scikit-learn and wandb into a thin API client.

So the honest description of this file is: **duplicated on purpose, policed by CI.** A
cross-repo sync check is specced in SynthefyPFN#119 and will compare this module's
``model_json_schema()`` against the library's, on every merge to either repo. Until that
exists, ``tests/test_nori_data_models.py`` performs the same comparison wherever both
packages happen to be installed.

What is copied, and what is deliberately not:

* **Copied:** the field names, types, bounds, enums, defaults and documentation. That is
  precisely what ``model_json_schema()`` covers, which is what makes a schema comparison a
  complete check on this file rather than a partial one.
* **NOT copied:** the coherence rules (which combinations are incoherent), the fallback
  ladder, and ``resolve()``. Those live server-side and run there. Duplicating behaviour
  would add divergence a schema comparison cannot see — the client would have to be *right*,
  not merely *matching*. An incoherent policy is rejected by the server with the library's
  own message, before any inference is paid for.

Two peers, deliberately at different versions: the **hosted** path runs the internal repo's
copy of the library (serving vendors it at deploy time), while ``mode="local"`` uses whatever
``synthefy-nori`` is pip-installed from PyPI. So this mirror tracks the *wire* contract, and
local mode gates separately on what is installed — see ``_local_memory_policy_available``.
"""

from __future__ import annotations

from typing import Any, Callable, List, Literal, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from typing_extensions import Annotated

#: Named starting points accepted wherever a policy is expected, instead of a field dict.
MEMORY_PRESETS = ("exact", "max_context", "off")

#: The fallback rungs the server may report, in decreasing memory cost.
MEMORY_RUNGS = (
    "no_cache",
    "resident_bf16",
    "resident_int8",
    "offload_bf16",
    "offload_int8",
    "context_row_chunk",
    "plain_loop",
)

#: The direct library's defaults, copied here because synthefy must remain usable
#: without importing the heavyweight synthefy-nori package.
DEFAULT_LARGE_CONTEXT_THRESHOLD = 50_000
DEFAULT_LARGE_CONTEXT_SEED = 0

#: Hosted serving owns its work bounds; these cap the two integer controls before
#: a request is sent.
MAX_LARGE_CONTEXT_THRESHOLD = 10_000_000
MAX_LARGE_CONTEXT_SEED = 2**32 - 1

MemoryPreset = Literal["exact", "max_context", "off"]
#: Local mode also accepts callables; hosted modes require a policy-name string and
#: let the server's installed synthefy-nori resolver decide whether it is valid.
LargeContextPolicy = Union[str, Callable[..., Any]]
MultiTargetPredictionStrategy = Literal["independent", "copula", "autoregressive"]
DEFAULT_MULTI_TARGET_PREDICTION_STRATEGY: MultiTargetPredictionStrategy = "copula"
MAX_MULTI_TARGET_RANDOM_STATE = 2**32 - 1

# Hosted-only work bounds. The public policy model intentionally does not apply
# these maxima because the same object is also accepted by ``mode="local"``.
MAX_MULTI_TARGET_DRAWS = 1_000
MAX_MULTI_TARGET_COPULA_CV = 20
MAX_MULTI_TARGET_COPULA_PIT_JITTER = 0.1
MAX_MULTI_TARGET_AUTOREGRESSIVE_ORDERS = 8


class MultiTargetPredictionPolicy(BaseModel):
    """Controls for multi-target joint draws and dependence fitting.

    Copula is the recommended/default strategy. Use independent for the lowest
    cost, or autoregressive to model order-sensitive conditional structure.
    Explicit autoregressive orders control factorization, not causality.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_draws: int = Field(
        300,
        ge=1,
        description="Joint Monte Carlo draws returned or averaged per query row.",
    )
    random_state: Optional[int] = Field(
        0,
        ge=0,
        le=MAX_MULTI_TARGET_RANDOM_STATE,
        description="Seed for reproducible joint sampling.",
    )
    copula_cv: int = Field(
        5,
        ge=2,
        description="Cross-fitting folds used to estimate copula residual ranks.",
    )
    copula_pit_jitter: float = Field(
        1e-4,
        ge=0.0,
        description="Uniform jitter applied to tied copula probability transforms.",
    )
    autoregressive_n_orders: int = Field(
        3,
        ge=1,
        description=(
            "Number of automatically generated unique target permutations; do not set with autoregressive_orders."
        ),
    )
    autoregressive_orders: Optional[List[List[int]]] = Field(
        None,
        description=(
            "Explicit complete target-index permutations, used exactly as supplied. "
            "Fit-dependent and not a causal claim."
        ),
    )

    @model_validator(mode="after")
    def _validate_explicit_orders(self):
        orders = self.autoregressive_orders
        if orders is None:
            return self
        if "autoregressive_n_orders" in self.model_fields_set:
            raise ValueError(
                "autoregressive_orders cannot be combined with explicitly supplied autoregressive_n_orders"
            )
        if not orders or any(not order for order in orders):
            raise ValueError("autoregressive_orders must contain non-empty target orders")
        if len({tuple(order) for order in orders}) != len(orders):
            raise ValueError("autoregressive_orders must not contain duplicate orders")
        return self


class MemoryPolicy(BaseModel):
    """How much memory Nori inference may use, and where the key/value cache lives.

    Nori does in-context regression, so your table is *input*: one prediction keeps a
    per-layer key/value cache over every context row, and that cache — not the ~6M-parameter
    model — is what exhausts GPU memory on a big table. This decides what to do about it.

    Every declared default is the real value, so reading the signature tells you what
    happens. Adaptivity is expressed as *permissions* (``allow_quantization``,
    ``offload_to_host``) rather than an opaque ``"auto"``, so "what is the default" and "may
    it change under pressure" stay separate, individually readable questions.

    Pass it to :meth:`synthefy.SynthefyNoriClient.predict` as ``memory_policy=``. Unknown
    fields are rejected here rather than silently dropped, matching the server.

    Note that ``NoriPredictRequest.memory_policy`` is deliberately NOT typed as this class.
    Pydantic would coerce a caller's partial dict into a full instance and send all thirteen
    fields — semantically identical today, but it would make the CLIENT pin the server's
    defaults, so a later change to a default would be silently overridden by every older
    client. Only the fields you actually set go on the wire; the server applies its own
    defaults to the rest.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cache: bool = Field(
        True,
        description=(
            "Build a key/value cache over the context rows at all. False re-reads the "
            "whole context for every batch of query rows instead: correct, but several "
            "times slower on a large query set, and it may have to drop context rows to "
            "fit. "
        ),
    )

    reuse_context_cache: bool = Field(
        True,
        description=(
            "Retain and reuse the encoded context across separate predict() calls "
            "on this estimator when the context and cache parameters are exactly "
            "unchanged. False still uses the K/V cache within each prediction, but "
            "does not retain context-derived state afterwards. Shared serving "
            "processes force this off; local fit-once/predict-many use keeps it on."
        ),
    )

    cache_dtype: Literal["bf16", "int8"] = Field(
        "bf16",
        description=(
            "Precision the K/V cache STARTS at. bf16 is bit-exact and is the default; set "
            "'int8' to quantize from the outset (~1.9x smaller, |dR2| ~ 6e-6) when you "
            "would rather trade that for context. Whether bf16 may be downgraded under "
            "memory pressure is a separate question — see allow_quantization. "
        ),
    )

    allow_quantization: bool = Field(
        True,
        description=(
            "May the cache be quantized to int8 when full precision would NOT stay "
            "resident? This is the only way int8 gets used by default, and it is strictly "
            "better than offloading (which costs 40-175% latency). False keeps every rung "
            "bit-exact — the 'exact' preset — at the cost of offloading sooner. "
        ),
    )

    gpu_budget_frac: float = Field(
        0.4,
        gt=0,
        le=1,
        description=(
            "Share of TOTAL VRAM the resident cache may occupy before we offload. A "
            "fraction so one setting is portable across GPUs. Total rather than free "
            "VRAM: free is racy, and budgeting against it would make the same call take "
            "different rungs on different runs. "
        ),
    )

    gpu_budget_absolute_gb: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Hard VRAM ceiling in GiB, overriding gpu_budget_frac. For a co-tenanted GPU, "
            "where a fixed cap is the requirement and a share of the card is the wrong "
            "unit. None = use the fraction; 0 = never keep the cache in GPU memory. 0 is "
            "deliberately legal, not a typo guard: it is a real request, and it is also "
            "the value the resolved budget can take. "
        ),
    )

    offload_to_host: bool = Field(
        True,
        description=(
            "May the cache be moved to host RAM (streamed back per layer) when it cannot "
            "stay resident? Bit-exact transport, but 40-175% slower. False reproduces the "
            "legacy behaviour of skipping the cache entirely instead, which is slower "
            "still. "
        ),
    )

    host_budget_frac: float = Field(
        0.25,
        gt=0,
        le=1,
        description=(
            "Share of total physical RAM the offloaded cache may occupy. A fraction "
            "because a flat GB default is a latent bug: 128 GB 'fits' on a 32 GB laptop "
            "by arithmetic, so offload proceeds and the kernel OOM-kills the process "
            "instead of the policy falling to the plain loop. "
        ),
    )

    host_budget_absolute_gb: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Hard host-RAM ceiling in GiB, overriding host_budget_frac. None = use the "
            "fraction; 0 = never offload. 0 is deliberately legal: it is also what a "
            "platform that will not report its RAM resolves to, and that case has to "
            "degrade to 'no offload' rather than fail. "
        ),
    )

    context_row_chunk: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Bound prefill to this many context rows (deterministic, with small "
            "floating-point reassociation differences). None = off, with 2048, 1024, "
            "then 512 tried after OOMs. An explicit value is a first-attempt cap; retries "
            "stay at or below it. "
        ),
    )

    adaptive_query_chunk: bool = Field(
        True,
        description=(
            "On a decode OOM, halve the QUERY chunk and retry instead of raising. "
            "Distinct from context_row_chunk, which caps CONTEXT rows during prefill. "
        ),
    )

    elements_budget: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Per-forward element cap driving query chunk size and context subsampling. "
            "None = derive it from available VRAM. Upstream of everything else here: the "
            "cached path engages only when the query set exceeds the chunk size this "
            "implies, so raising it far enough disables the cache knobs by making "
            "inference stop chunking at all. "
        ),
    )

    allow_subsample: bool = Field(
        True,
        description=(
            "Permit dropping context rows when the request will not fit otherwise. False "
            "makes that an error instead of a silent accuracy loss. "
        ),
    )


class MemoryAttempt(BaseModel):
    """One execution attempt contributing to a prediction's memory outcome."""

    model_config = ConfigDict(frozen=True, extra="allow")

    pipeline_ids: List[int] = Field(default_factory=list)
    path: Literal["pipeline_batch", "cached", "plain_loop"]
    rung: str
    cache_dtype: Literal["bf16", "int8"]
    offload_to_host: bool
    context_row_chunk: Optional[int] = Field(None, gt=0)
    outcome: Literal["success", "oom", "unsupported"]
    reason: Literal[
        "resolved",
        "oom_retry",
        "fallback_after_oom",
        "fallback_after_unsupported",
    ]
    dropped_context_rows: int = Field(0, ge=0)


class MemoryReport(BaseModel):
    """What the server did about ``memory_policy=`` — the resolved policy plus its outcome.

    Returned as ``memory_report`` and surfaced on the client as
    :attr:`synthefy.SynthefyNoriClient.last_memory_report`. Read it: the rung is decided by
    the replica's free VRAM at that moment, not by your request, so it is not knowable from
    the client side.

    ``extra="allow"`` on purpose, unlike :class:`MemoryPolicy`. This is a *response* model,
    and a server newer than your client may report a field this copy does not know about;
    dropping it would be more surprising than carrying it through.
    """

    model_config = ConfigDict(extra="allow")

    rung: Optional[str] = Field(
        None,
        description=(
            "Which fallback the server used, in decreasing memory cost: resident_bf16 "
            "(cache in GPU memory, exact), resident_int8 (quantized), offload_bf16 / "
            "offload_int8 (cache in host RAM, streamed back per layer, exact transport), "
            "context_row_chunk (cache built in row chunks), plain_loop (no cache; the "
            "context re-read per query batch), no_cache (the cached path did not apply). "
            "Decided per request, not requestable. "
        ),
    )

    est_cache_gb: Optional[float] = Field(
        None,
        description=("Full-precision cache footprint for this request's context, in GiB. Reported, not set. "),
    )

    resident_gb: Optional[float] = Field(
        None,
        description=(
            "Footprint at the chosen precision (GiB) — the figure actually compared "
            "against the budget. Reported, not set. "
        ),
    )

    query_chunk: Optional[int] = Field(
        None,
        description=(
            "Query rows per decode forward, as the cached path was entered with. "
            "Reported, not set. NOTE: a decode OOM may halve this further inside the "
            "model, and that further reduction is not currently reported here. "
        ),
    )

    dropped_context_rows: int = Field(
        0,
        ge=0,
        description=(
            "Context rows subsampled before cache resolution to fit the element budget; "
            "recorded on cached and plain-loop outcomes so a shrunk context is visible "
            "rather than inferred from the score. "
        ),
    )

    attempt_history: List[MemoryAttempt] = Field(
        default_factory=list,
        description="Chronological memory execution attempts, including runtime OOMs, "
                    "fit-row retries, and the successful fallback.",
    )

    clamped: List[str] = Field(
        default_factory=list,
        description=(
            "Fields the server capped rather than honouring verbatim; empty when the policy "
            "was used as sent. Only the host-RAM budgets are ever capped, because exceeding "
            "the container's memory limit kills the replica outright and charges the next "
            "caller a cold start."
        ),
    )
    notes: List[str] = Field(
        default_factory=list,
        description=(
            "Remarks about the policy you sent: settings that work but are probably not what "
            "you meant, e.g. a budget that cannot take effect. Empty when it was unambiguous. "
            "Locally these are Python warnings; over HTTP they are returned here instead, "
            "because a warning would land in the server's log rather than reaching you."
        ),
    )


class MultiTargetMemoryReport(MemoryReport):
    """Memory outcome for one internal multi-target marginal or chain call."""

    strategy: MultiTargetPredictionStrategy
    target: int = Field(ge=0)
    order: Optional[int] = Field(
        None,
        ge=0,
        description="Autoregressive order index, or None for marginal calls.",
    )


class LargeContextReport(BaseModel):
    """What one client call did about large_context_policy.

    This is a capability handshake as well as observability. A deployment older
    than the large-context wire contract ignores an unknown request field and can
    return plausible ordinary predictions, so a client that requested a policy
    must require this report.

    Extra response fields are allowed for forward compatibility.
    """

    model_config = ConfigDict(extra="allow")

    applied: bool = Field(
        description=(
            "Whether the request crossed its row threshold and entered the "
            "large-context policy path. False means ordinary full-context/memory-policy "
            "inference ran instead; see reason."
        )
    )
    policy: str = Field(
        description=("Resolved policy name when available; otherwise a display label for the skipped request.")
    )
    threshold: int = Field(
        ge=1,
        le=MAX_LARGE_CONTEXT_THRESHOLD,
        description="Context-row threshold honored for this request.",
    )
    seed: int = Field(
        ge=0,
        le=MAX_LARGE_CONTEXT_SEED,
        description="Deterministic policy seed honored for this request.",
    )
    reason: Optional[str] = Field(
        None,
        description="Why the policy was not applied, currently below_threshold, else null.",
    )
    window: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum context rows used by each internal Nori call when applied.",
    )
    n_train: int = Field(ge=0, description="Number of context rows in the request.")
    n_test: int = Field(ge=0, description="Number of query rows in the request.")
    shards_available: Optional[int] = Field(
        None,
        ge=0,
        description="How many full policy windows the context table contains.",
    )
    nori_calls: int = Field(
        ge=0,
        description="Internal Nori forward calls made for this client request.",
    )
    full_context: Optional[bool] = Field(
        None,
        description=(
            "Whether the applied policy used the complete context in one call. Null when the policy did not engage."
        ),
    )
    reused_train_state: bool = Field(
        description=(
            "Whether train-derived policy state was reused. Shared hosted requests are one-shot and report false."
        )
    )
    gate_winner: Optional[str] = Field(
        None,
        description=("Winning policy when a holdout gate was requested; otherwise null."),
    )


def _accept_another_packages_policy(value: Any) -> Any:
    """Let an instance of ``synthefy_nori``'s ``MemoryPolicy`` validate as this one.

    Anyone with ``synthefy-nori`` installed may reasonably build the library's own
    ``MemoryPolicy`` and pass it. It is a different class with the same name, so pydantic
    refuses it — and the message it produces ("input should be an instance of MemoryPolicy")
    is actively misleading to someone who is holding exactly that. Dump it to a dict and let
    normal validation take over.

    ``exclude_unset`` so only the fields that caller actually set survive: the server (or the
    library, in local mode) then applies its own defaults to the rest, instead of this client
    freezing whatever the defaults happened to be when it was installed.

    A BeforeValidator rather than a helper called from ``predict()``: attached to the type it
    applies on construction, on assignment, and anywhere the annotation is reused — there is no
    call site to forget.
    """
    if value is None or isinstance(value, (str, dict, MemoryPolicy)):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_unset=True)
    return value  # let pydantic produce the type error


#: What ``memory_policy=`` accepts: a preset name, a policy, a plain dict (validated into one),
#: or another package's equivalent. Anything else is a pydantic error before a request is sent.
MemoryPolicyInput = Annotated[Union[MemoryPreset, MemoryPolicy], BeforeValidator(_accept_another_packages_policy)]
