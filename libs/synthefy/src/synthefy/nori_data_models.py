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

So the honest description of this file is: **duplicated on purpose, policed by CI.** A CI
check compares this module's ``model_json_schema()`` against the library's, per field, on every
merge. This package's own tests cover the mirror's standalone behaviour.

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

The fine-tuning models at the end of this file (``FineTuneHyperparameters``, ``ChildCurve``,
``TrainingPlan`` and ``TrainingReport``) and the constants beside them describe the Nori fine-tuning
Algorithm: the hyperparameters a job takes, the plan it works out and the ``training_report.json`` it
writes. CI checks them against the
Algorithm. ``TrainingReport.validation_source`` is a plain ``str``, so a report from a newer
Algorithm version still parses.

The request and result schemas at the very end (``FineTuneJobRequest``, ``FineTuneJob`` and
``FineTunedEndpoint``) mirror nothing: they are the client's own description of a SageMaker
CreateTrainingJob request, of a training job as SageMaker describes it, and of an endpoint
:class:`~synthefy.SynthefyNoriFineTuner` deployed.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from synthefy._aws import check_sagemaker_resource_name, parse_s3_uri
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
    "stream_bf16",
    "stream_int8",
    "context_row_chunk",
    "plain_loop",
)

#: The direct library's defaults, copied here because synthefy must remain usable
#: without importing the heavyweight synthefy-nori package.
DEFAULT_LARGE_CONTEXT_THRESHOLD = 50_000
DEFAULT_LARGE_CONTEXT_SEED = 0
DEFAULT_LARGE_CONTEXT_HOLDOUT = "random"
MAX_LARGE_CONTEXT_CANDIDATES = 8

#: Hosted serving owns its work bounds; these cap the two integer controls before
#: a request is sent.
MAX_LARGE_CONTEXT_THRESHOLD = 10_000_000
MAX_LARGE_CONTEXT_SEED = 2**32 - 1

MemoryPreset = Literal["exact", "max_context", "off"]
#: Local mode also accepts a callable; hosted modes accept one built-in name or a
#: bounded list of names for the per-table holdout gate.
LargeContextPolicy = Union[str, List[str], Callable[..., Any]]
LargeContextHoldout = Literal["random", "tail"]
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
            "Precision the K/V cache STARTS at. bf16 stores projected K/V without "
            "additional quantization and is the default; set "
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
            "better than offloading (which costs 40-175% latency). False forbids cache quantization — the "
            "'exact' preset — at the cost of offloading sooner. It does not guarantee "
            "identical predictions across chunk sizes or cached/uncached paths, "
            "or prevent context subsampling. "
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

    stream_context: bool = Field(
        False,
        description=(
            "Keep the full context and K/V cache in host RAM and stage only bounded "
            "row slices on the GPU. This forces the host-only stream_bf16/stream_int8 "
            "path, disables cross-call context reuse, and uses "
            "context_row_chunk=2048 as the maximum staged-row cap when no cap is "
            "supplied. Runtime may split K/V blocks smaller to honor its fixed FP32 "
            "online-attention workspace ceiling. Streaming never silently falls back "
            "to plain_loop."
        ),
    )

    context_row_chunk: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Bound prefill to this many context rows (deterministic, with small "
            "floating-point reassociation differences). None = off, with 2048, 1024, 512, "
            "then 256 tried after OOMs. An explicit value is a first-attempt cap; retries "
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
    query_chunk: Optional[int] = Field(None, gt=0)
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
            "stream_bf16 / stream_int8 (explicit bounded host-only context streaming; "
            "not bit-exact), context_row_chunk (cache built in row chunks), plain_loop "
            "(no cache; the context re-read per query batch), no_cache (the cached path "
            "did not apply). Decided per request, not requestable. "
        ),
    )

    est_cache_gb: Optional[float] = Field(
        None,
        ge=0,
        description=("Full-precision cache footprint for this request's context, in GiB. Reported, not set. "),
    )

    resident_gb: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Footprint at the chosen precision (GiB) — the figure actually compared "
            "against the budget. Reported, not set. "
        ),
    )

    query_chunk: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Effective query rows per decode forward on the successful cached path. "
            "If decoding exhausted its retries and raised, this is the last attempted "
            "size. Every adaptive reduction is also recorded in attempt_history. "
            "Reported, not set. "
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
        description="Chronological memory execution attempts, including cache-build "
        "OOMs, decode query-chunk retries, fit-row retries, and the "
        "successful fallback.",
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
    holdout_strategy: Optional[LargeContextHoldout] = Field(
        None,
        description=(
            "Validation split used by a policy-list gate: random for IID rows or "
            "tail for chronologically ordered rows. Null for a single policy."
        ),
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


# ------------------------------------------------------------------ fine-tuning on SageMaker
#: A served model's name is at most this long: the job's ``model_name``, and the name the endpoint serves it under.
MAX_MODEL_NAME_CHARS = 63
#: The prefix that tells a fine-tuned name from a released one in the endpoint's request schema, which lists the
#: released names and admits a fine-tuned name only by this prefix.
FINE_TUNED_MODEL_NAME_PREFIX = "ft-"
_FINE_TUNED_NAME_BODY_CHARS = MAX_MODEL_NAME_CHARS - len(FINE_TUNED_MODEL_NAME_PREFIX)
#: A fine-tuned model's name, matched whole (``fullmatch``); the job holds ``model_name`` to the same pattern.
FINE_TUNED_MODEL_NAME_PATTERN = (
    rf"{FINE_TUNED_MODEL_NAME_PREFIX}[a-z0-9][a-z0-9-]{{0,{_FINE_TUNED_NAME_BODY_CHARS - 1}}}"
)
FINE_TUNED_MODEL_NAME_RE = re.compile(FINE_TUNED_MODEL_NAME_PATTERN)
#: FINE_TUNED_MODEL_NAME_PATTERN in words, the sentence every refusal of a name quotes.
FINE_TUNED_MODEL_NAME_RULE = (
    f"'{FINE_TUNED_MODEL_NAME_PREFIX}' followed by 1 to {_FINE_TUNED_NAME_BODY_CHARS} characters of lowercase "
    "letters, digits and '-', the first a letter or digit"
)

#: The released model names. A fine-tuned model may not take one, so an answer that names a
#: released model always comes from that released model.
RELEASED_MODEL_NAMES = ("nori-6m", "nori-30m", "nori-30m-thinking-medium", "nori-100m", "nori-1-5")

#: The released models a fine-tuning job accepts as ``base_model``.
BaseModelName = Literal["nori-6m", "nori-1-5", "nori-100m"]

#: What ended a child's training (``ChildCurve.stop_reason``) or the job's (``TrainingReport.stop_reason``).
StopReason = Literal["plan_done", "patience", "max_training_hours", "stop_request", "non_finite_loss"]
#: How the rows are split among the models (``FineTuneHyperparameters.row_partition_strategy``).
RowPartitionStrategy = Literal["shard", "folds"]

#: The hyperparameter string that turns off a setting whose default is on (``patience``, ``max_training_hours``).
_NONE_HYPERPARAMETER = "none"

#: The fine-tuning Algorithm's training instance types with 48 GB GPUs: one GPU on the first two, 4 on
#: ``ml.g6e.12xlarge`` and 8 on ``ml.g6e.48xlarge``. ``nori-100m`` needs one of them.
LARGE_GPU_INSTANCE_TYPES = ("ml.g6e.xlarge", "ml.g6e.2xlarge", "ml.g6e.12xlarge", "ml.g6e.48xlarge")
#: The fine-tuning Algorithm's training instance types with several GPUs and the number each has; every other type
#: has one. A job trains one model per GPU at a time.
MULTI_GPU_INSTANCE_TYPES = MappingProxyType(
    {"ml.g5.12xlarge": 4, "ml.g6e.12xlarge": 4, "ml.g5.48xlarge": 8, "ml.g6e.48xlarge": 8}
)

#: The job's input channels: the training table and the optional validation table.
TRAIN_CHANNEL = "train"
VALIDATION_CHANNEL = "validation"

#: The report inside a fine-tuning job's ``model.tar.gz``.
REPORT_FILENAME = "training_report.json"

#: The derived MaxRuntimeInSeconds: max_training_hours plus 30 minutes for the download and the model's writing, or
#: SageMaker's default limit on a training job's runtime (5 days) when there is no cap.
_RUNTIME_MARGIN_S = 1800
SAGEMAKER_MAX_RUNTIME_S = 432_000
#: The largest max_training_hours: SageMaker's default limit less the 30 minutes, so the derived runtime fits it.
MAX_TRAINING_HOURS = (SAGEMAKER_MAX_RUNTIME_S - _RUNTIME_MARGIN_S) / 3600


def get_runtime_for_training_cap(hours: float) -> int:
    """The derived ``max_runtime_s`` for a ``max_training_hours`` of ``hours``: the hours in seconds, rounded up,
    plus 1,800 s for the download and the model's writing."""
    return math.ceil(round(hours * 3600, 6)) + _RUNTIME_MARGIN_S


#: The fewest context rows a training step takes (the recipe's smallest episode context).
MIN_CONTEXT_ROWS = 16
#: How many rows of each kind a training step takes: one count, or ``{rows: weight}``, from which each step draws one
#: count with probability proportional to its weight. None leaves the count to the job.
RowCounts = Optional[Union[int, Dict[int, float]]]


def get_row_count_weights(value: RowCounts) -> Optional[Dict[int, float]]:
    """The ``{rows: weight}`` a row-count setting stands for: ``{n: 1.0}`` for one count ``n``, the mapping itself
    otherwise, and None for None."""
    if value is None:
        return None
    if isinstance(value, int):
        return {int(value): 1.0}
    return {int(rows): float(weight) for rows, weight in value.items()}


def _get_row_counts_from_text(value: Any) -> Any:
    """``value``, or the row counts its text holds: SageMaker hands over a string, which is a number, a JSON object
    of ``{rows: weight}``, or "none"."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() == "none":
        return None
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not a JSON object of row counts and weights: {exc}") from exc
    return text


def _check_row_counts(name: str, value: RowCounts, min_rows: int) -> None:
    """Raise ValueError unless ``value`` is None, or row counts of at least ``min_rows`` whose weights are finite, not
    negative and sum to more than 0."""
    weights = get_row_count_weights(value)
    if weights is None:
        return
    if not weights:
        raise ValueError(f"{name} must hold at least one row count")
    if min(weights) < min_rows:
        raise ValueError(f"{name} must be at least {min_rows} rows, got {min(weights)}")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError(f"{name} weights must be finite and not negative")
    if sum(weights.values()) <= 0:
        raise ValueError(f"{name} weights must sum to more than 0")


def _check_column_roles(
    target_column: Optional[str],
    feature_columns: Optional[List[str]],
    task_column: Optional[str],
    time_column: Optional[str],
) -> None:
    """Raise ValueError when the task or time column is the target, a feature or the other one: each only says which
    task a row belongs to or when it was."""
    features = feature_columns or []
    for setting, column in (("task_column", task_column), ("time_column", time_column)):
        if column is not None and (column == target_column or column in features):
            raise ValueError(f"{setting} must be neither target_column nor one of feature_columns")
    if task_column is not None and task_column == time_column:
        raise ValueError("task_column and time_column must be different columns")


def _get_column_list_from_text(value: Any) -> Any:
    """``value``, or the column names its text holds: SageMaker hands over a string, a JSON list or a comma-separated
    list."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"not a JSON list: {exc}") from exc
        else:
            value = [column.strip() for column in text.split(",") if column.strip()]
        if not value:
            raise ValueError("must name at least one column")
    return value


class TaskTable(BaseModel):
    """One table of the training channel and the regression task, or tasks, it holds: one task, or one task per value
    of its task_column."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description=(
            "A Parquet file or a folder of the training channel, relative to the channel; a folder holds every file "
            "under it, and an empty path the whole channel."
        )
    )
    target_column: str = Field(min_length=1, description="The column of this table to predict.")
    feature_columns: Optional[List[str]] = Field(
        None, description="This table's feature columns; default every column but its target, task and time columns."
    )
    task_column: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "A column naming each row's task, such as a store or series code (numbers or text): one task per value. "
            "Not a feature. When the files lack it, key=value folder names supply it, as partition_cols writes them."
        ),
    )
    time_column: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "A column of times (timestamp, date or integer) that makes each task a time series: its rows are sorted "
            "by it, every training step predicts the rows right after its context, and each task is validated and "
            "scored on its latest rows. Not a feature. Needs query_rows."
        ),
    )
    context_rows: RowCounts = Field(
        None,
        description=(
            "How many context rows (rows given with their target) a training step of this table's tasks takes: a "
            'number, or row counts and weights such as {"168": 1, "672": 1}, from which each step draws one; at '
            "least 16. Default: the job's context_rows."
        ),
    )
    query_rows: RowCounts = Field(
        None,
        description=(
            "How many rows a training step of this table's tasks predicts: a number, or row counts and weights. For a "
            "time series these are the rows right after the step's context, so the furthest is this many rows past "
            "it. Default: the job's query_rows."
        ),
    )

    @field_validator("feature_columns", mode="before")
    @classmethod
    def _parse_feature_columns(cls, value):
        """The column names ``value`` holds, parsed from the string SageMaker hands over."""
        return _get_column_list_from_text(value)

    @field_validator("context_rows", "query_rows", mode="before")
    @classmethod
    def _parse_row_counts(cls, value):
        """The row counts ``value`` holds, parsed from the string SageMaker hands over."""
        return _get_row_counts_from_text(value)

    @model_validator(mode="after")
    def _check_columns_and_row_counts(self) -> "TaskTable":
        """The table unchanged; raises ValueError for a task or time column that is the target, a feature or the
        other one, and for row counts out of range."""
        _check_column_roles(self.target_column, self.feature_columns, self.task_column, self.time_column)
        _check_row_counts("context_rows", self.context_rows, MIN_CONTEXT_ROWS)
        _check_row_counts("query_rows", self.query_rows, 1)
        return self


class FineTuneHyperparameters(BaseModel):
    """The Nori fine-tuning Algorithm's hyperparameters; each also parses from the string SageMaker delivers."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    target_column: Optional[str] = Field(
        None,
        min_length=1,
        description="The column of the training table to predict; unset with tasks, whose tables each name theirs.",
    )
    model_name: str = Field(
        description=f"The name the fine-tuned model answers to (the request's model): {FINE_TUNED_MODEL_NAME_RULE}."
    )
    base_model: BaseModelName = Field("nori-1-5", description="The released Nori model to fine-tune.")
    task: Literal["regression"] = Field("regression", description="Only regression is supported.")
    feature_columns: Optional[List[str]] = Field(
        None, description="Comma-separated or JSON list of feature columns; default every other column."
    )
    # Any of the next six makes a job of tasks (module docstring, Tasks). Each defaults to None, never to a value, so a
    # client sends none of them to an Algorithm version that predates them.
    task_column: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "A column naming each row's task, such as a store or series code (numbers or text): one fine-tuned model "
            "learns every task, and a prediction picks its task by sending that task's rows as context. Not a feature. "
            "When the files lack it, key=value folder names supply it, as partition_cols writes them. Each task's "
            "target is standardized, held out and scored on its own. The table is read into memory whole."
        ),
    )
    time_column: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "A column of times (timestamp, date or integer) that makes each task a time series: its rows are sorted "
            "by it, every training step predicts the rows right after its context, and each task is validated and "
            "scored on its latest rows. Not a feature. Needs query_rows. The table is read into memory whole."
        ),
    )
    context_rows: RowCounts = Field(
        None,
        description=(
            "How many context rows (rows given with their target) a training step takes: a number, or row counts and "
            'weights such as {"168": 1, "672": 1}, from which each step draws one; at least 16. Default: as many as '
            "fit in the step. With tasks, the default of every task table."
        ),
    )
    query_rows: RowCounts = Field(
        None,
        description=(
            "How many rows a training step predicts: a number, or row counts and weights. For a time series these are "
            "the rows right after the step's context, so the furthest is this many rows past it; required with "
            "time_column. Default: up to 1,024 rows. With tasks, the default of every task table."
        ),
    )
    tasks: Optional[List[TaskTable]] = Field(
        None,
        description=(
            "A JSON list of task tables, for tasks in different files with different targets or features: each names "
            "a file or folder of the training channel (path) and its target_column, and may set feature_columns, "
            "task_column, time_column, context_rows and query_rows. Every training file belongs to exactly one table. "
            "Set tasks or target_column, not both. The tables are read into memory whole."
        ),
    )
    # The Algorithm's defaults; CI fails if they drift.
    bag: int = Field(
        8,
        ge=1,
        le=8,
        description=(
            "How many models the job trains, 1 to 8; the fine-tuned model averages them. Models train one per GPU "
            "at a time."
        ),
    )
    row_partition_strategy: RowPartitionStrategy = Field(
        "shard",
        description=(
            '"shard": each model trains on its own 1/bag of the training rows; "folds": each model trains on every '
            "training row outside its validation fold. A job of tasks ignores it: each model trains on every task."
        ),
    )
    n_passes_per_row: Optional[float] = Field(
        None,
        gt=0,
        le=100,
        description=(
            "How many times each model reads each training row it trains on (1.0 when unset: every row once; below 1: "
            "part of the rows). Reading a row is not predicting it: at 20 features about 13% of the rows a pass reads "
            "serve as prediction targets (the share depends on the table's width), so more passes make more rows "
            "targets, with diminishing returns: about 1 - 0.87^passes of the rows (1 pass: 13%, 2: 24%, 5: 50%, 10: "
            "75%, 20: 94%). The training plan and report give the figures for your table. Cannot be set together with "
            "n_steps_per_model."
        ),
    )
    n_steps_per_model: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "The exact number of training steps each model takes, in place of the steps n_passes_per_row works out "
            "to; cannot be set together with n_passes_per_row."
        ),
    )
    lr: float = Field(1e-5, gt=0, description="Learning rate.")
    patience: Optional[int] = Field(
        25,
        ge=1,
        description=(
            "How many validation scores in a row without a better one stop a model, a wait of patience x "
            "validation_every_n_steps training steps; none: a model takes every step of its plan."
        ),
    )
    max_training_hours: Optional[float] = Field(
        24.0,
        gt=0,
        le=MAX_TRAINING_HOURS,
        allow_inf_nan=False,
        description=(
            "The cap, in hours from the job's start, on reading the table and training the models: each model stops "
            "at it, and a model not started by then does not train; at most 119.5; none: no cap."
        ),
    )
    validation_every_n_steps: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "How many training steps pass between two validation scores of a model; default: the steps per model "
            "divided by 100, rounded down, at most 100 divided by patience, rounded down (so the wait patience x "
            "this is at most 100 steps), and at least 1. In a job of tasks it counts rounds of one training step per "
            "task."
        ),
    )
    # The Algorithm refuses a negative seed; refusing here names the field.
    seed: int = Field(0, ge=0, description="The job's random seed.")

    @field_validator("patience", "max_training_hours", mode="before")
    @classmethod
    def _parse_none_string_as_null(cls, value: Any) -> Any:
        """``value``, or None when it is the string "none" in any case: how a setting turned off travels as a SageMaker
        hyperparameter."""
        if isinstance(value, str) and value.strip().lower() == _NONE_HYPERPARAMETER:
            return None
        return value

    @model_validator(mode="after")
    def _refuse_steps_with_passes(self) -> "FineTuneHyperparameters":
        """The hyperparameters unchanged; raises ValueError when ``n_steps_per_model`` and ``n_passes_per_row`` are both
        set, since each sets the number of steps."""
        if self.n_steps_per_model is not None and self.n_passes_per_row is not None:
            raise ValueError("n_steps_per_model and n_passes_per_row both set the number of steps; set one of them")
        return self

    @field_validator("context_rows", "query_rows", mode="before")
    @classmethod
    def _parse_row_counts(cls, value):
        """The row counts ``value`` holds, parsed from the string SageMaker hands over."""
        return _get_row_counts_from_text(value)

    @field_validator("tasks", mode="before")
    @classmethod
    def _parse_tasks(cls, value):
        """The task tables ``value`` holds: SageMaker hands over a string, a JSON list of task tables."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"not a JSON list of task tables: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _check_tasks(self) -> "FineTuneHyperparameters":
        """The hyperparameters unchanged; raises ValueError unless exactly one of ``target_column`` and ``tasks`` is
        set; when ``tasks`` is set with table settings at the top level, lists no table, lists a path twice, or is one
        table that is one task (``target_column`` says that); for a task or time column that is the target, a feature
        or the other one; for row counts out of range; and for a time series without query rows."""
        if (self.target_column is None) == (self.tasks is None):
            raise ValueError("set one of target_column (the channel is one table) and tasks (a list of task tables)")
        if self.tasks is not None:
            set_here = [name for name in ("feature_columns", "task_column", "time_column") if getattr(self, name)]
            if set_here:
                raise ValueError(f"with tasks, set {', '.join(set_here)} on each task table instead")
            if not self.tasks:
                raise ValueError("tasks must list at least one task table")
            paths = [table.path.strip("/") for table in self.tasks]
            repeated = sorted({path for path in paths if paths.count(path) > 1})
            if repeated:
                raise ValueError(f"tasks lists the path {repeated[0]!r} more than once")
            only_table = self.tasks[0]
            if len(self.tasks) == 1 and only_table.task_column is None and only_table.time_column is None:
                raise ValueError(
                    "one task table with neither task_column nor time_column is one task: set target_column (and "
                    "feature_columns) instead of tasks"
                )
        _check_column_roles(self.target_column, self.feature_columns, self.task_column, self.time_column)
        _check_row_counts("context_rows", self.context_rows, MIN_CONTEXT_ROWS)
        _check_row_counts("query_rows", self.query_rows, 1)
        for table in self.tasks or [self]:
            if table.time_column is not None and table.query_rows is None and self.query_rows is None:
                raise ValueError(
                    "time_column needs query_rows: how many rows after its context a training step predicts"
                    + ("" if table is self else f" (task table {table.path!r})")
                )
        return self

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        # A released name is refused by name first, the likelier mistake (reusing the base's name).
        if value in RELEASED_MODEL_NAMES:
            raise ValueError(
                f"{value!r} is a released model name; a fine-tuned model needs its own: {FINE_TUNED_MODEL_NAME_RULE}"
            )
        if not FINE_TUNED_MODEL_NAME_RE.fullmatch(value):
            raise ValueError(f"must be {FINE_TUNED_MODEL_NAME_RULE}")
        return value

    @field_validator("feature_columns", mode="before")
    @classmethod
    def _parse_feature_columns(cls, value: Any) -> Any:
        # SageMaker hands over a string: accept a JSON list or a comma-separated list.
        return _get_column_list_from_text(value)


def is_job_of_tasks(hyperparameters: FineTuneHyperparameters) -> bool:
    """Whether the hyperparameters make a job of tasks, which reads its tables into memory and has no plan before it
    runs: ``tasks``, ``task_column``, ``time_column``, ``context_rows`` or ``query_rows`` set."""
    return any(
        getattr(hyperparameters, name) is not None
        for name in ("tasks", "task_column", "time_column", "context_rows", "query_rows")
    )


def get_task_tables(hyperparameters: FineTuneHyperparameters) -> List[TaskTable]:
    """The task tables of a job: each of ``tasks`` with the job's ``context_rows`` and ``query_rows`` where it sets
    none, or the whole channel as one table (path "") with the top-level settings."""
    if hyperparameters.tasks is None:
        return [
            TaskTable(
                path="",
                target_column=hyperparameters.target_column,
                feature_columns=hyperparameters.feature_columns,
                task_column=hyperparameters.task_column,
                time_column=hyperparameters.time_column,
                context_rows=hyperparameters.context_rows,
                query_rows=hyperparameters.query_rows,
            )
        ]
    return [
        table.model_copy(
            update={
                "context_rows": hyperparameters.context_rows if table.context_rows is None else table.context_rows,
                "query_rows": hyperparameters.query_rows if table.query_rows is None else table.query_rows,
            }
        )
        for table in hyperparameters.tasks
    ]


class ChildCurve(BaseModel):
    """One child's training curve: entry ``i`` of each list belongs to validated step ``steps[i]``.

    ``val_mse_best`` is the child's best validation score, reached at ``best_step``, which is 0 when no step
    improved; a non-finite number is null. ``interrupted`` is true when a stop request ended the child early.
    ``joined_average`` is true when the child contributed to the fine-tuned model.
    """

    child: int
    steps: List[int]
    train_loss: List[Optional[float]]
    val_mse: List[Optional[float]]
    best_step: int
    val_mse_best: Optional[float]
    interrupted: bool
    joined_average: bool
    rows_visited: int = Field(
        0, description="Training rows this child's steps used; a row used in two passes counts twice."
    )
    failed: bool = Field(False, description="Whether this child failed and was left out of the fine-tuned model.")
    failure: str = Field("", description="The failure's last lines; empty when the child joined.")
    restored: bool = Field(
        False,
        description="Whether an earlier job trained this child and this job read it back from the checkpoint folder.",
    )
    stop_reason: Optional[StopReason] = Field(
        None,
        description=(
            "What ended this model's training: plan_done (every planned step ran), patience, max_training_hours, "
            "stop_request or non_finite_loss; null in a report from an earlier job."
        ),
    )
    n_skipped_state_saves: int = Field(
        0,
        description=(
            "Saves of this model's progress to the checkpoint folder that were due and did not happen, for lack of "
            "disk space or because the folder already held that step; 0 without a checkpoint folder and for a model "
            "that failed."
        ),
    )
    last_saved_step: Optional[int] = Field(
        None,
        description=(
            "The newest step of this model whose progress this job or an earlier one saved to the checkpoint "
            "folder; null when none was, and for a model that failed."
        ),
    )


class TrainingPlan(BaseModel):
    """What a fine-tuning job reads and what it takes, worked out from the table and the hyperparameters before any
    model trains. ``bag`` and ``row_partition_strategy`` are the values the job uses, which the table's size can
    lower or change; ``warnings`` says when it did."""

    model_config = ConfigDict(protected_namespaces=())

    n_rows_in_table: int = Field(description="Rows of the training table.")
    n_feature_columns: int = Field(description="Feature columns the models read.")
    n_validation_rows: int = Field(
        description=(
            "Rows that score the fine-tuned model in the training report: the validation table's rows when one is "
            "given, else the training table's rows set aside for it."
        )
    )
    n_training_rows: int = Field(description="Rows of the training table the models may train on.")
    bag: int = Field(description="How many models the job trains.")
    row_partition_strategy: RowPartitionStrategy = Field(
        description="How the training rows are divided among the models."
    )
    n_rows_per_model: int = Field(description="Training rows of the largest model's share.")
    rows_per_step: float = Field(description="Rows one training step reads, averaged over the steps.")
    targets_per_step: float = Field(
        description="Rows of a training step used as prediction targets, averaged over the steps."
    )
    rows_per_step_is_estimate: bool = Field(
        description="Whether the two per-step figures are estimated from the number of feature columns rather than "
        "measured."
    )
    steps_per_model: int = Field(description="Training steps each model takes when nothing stops it sooner.")
    n_passes_per_row: float = Field(
        description="How many times each model reads each of its training rows over steps_per_model steps."
    )
    validation_every_n_steps: int = Field(description="Training steps between two validation scores of a model.")
    fraction_of_table_read: float = Field(description="Share of the training rows at least one model reads.")
    fraction_of_rows_as_targets: float = Field(
        description="Estimated share of the training rows used as a prediction target in at least one training step."
    )
    seconds_per_step: Optional[float] = Field(
        description="Seconds one training step takes on this instance type and base model for this table's rows and "
        "columns, with its share of the validation scores; null where not measured."
    )
    instance_type: str = Field(description="The training instance type.")
    gpus: int = Field(description="How many models train at the same time.")
    training_hours: Optional[float] = Field(
        description="Hours the models' training steps take when nothing stops them sooner; null where "
        "seconds_per_step is."
    )
    estimated_cost_usd: Optional[float] = Field(
        description=(
            "training_hours at the instance type's on-demand training price in us-east-1, in US dollars; null when "
            "either is unknown."
        )
    )
    prices_as_of: str = Field(description="The date of the training prices.")
    patience: Optional[int] = Field(
        None,
        description=(
            "Validation scores in a row without a better one that stop a model; null: every model takes every step "
            "of its plan."
        ),
    )
    patience_steps: Optional[int] = Field(
        description=(
            "Training steps a model trains without a better validation score before patience stops it: patience x "
            "validation_every_n_steps; null when patience is null."
        )
    )
    max_training_hours: Optional[float] = Field(description="The job's cap, in hours from its start; null: none.")
    hits_max_training_hours: bool = Field(
        description="Whether max_training_hours ends the training before every planned step runs."
    )
    warnings: List[str] = Field(description="What the plan warns of, one sentence each.")

    def get_summary(self) -> str:
        """The plan as four lines of text, then one line per warning.

        The lines give the models, the row partition and the rows each model may use; each model's steps, rows and
        targets per step, passes per row and validation interval, and the patience when it can stop the model sooner;
        the share of the training rows read and used as targets; and the time and cost at the dated on-demand prices (a
        ceiling when patience can stop a model sooner), with the ``max_training_hours`` cap.
        """
        validation_interval = self.validation_every_n_steps
        validation_text = "every step" if validation_interval == 1 else f"every {validation_interval:,} steps"
        # A wait for a better score shorter than the plan can stop a model before its last step, so the hours and
        # dollars are a ceiling.
        patience_steps = self.patience_steps
        can_stop_sooner = patience_steps is not None and patience_steps < self.steps_per_model
        stop_text = ""
        if can_stop_sooner:
            stop_text = f"; patience ({self.patience:,} scores, {patience_steps:,} steps) can stop it sooner"
        bound = "up to about" if can_stop_sooner else "about"
        lines = [
            f"Plan: {self.bag} model(s), {self.row_partition_strategy}, on {self.n_training_rows:,} of the table's "
            f"{self.n_rows_in_table:,} rows ({self.n_validation_rows:,} validation rows, {self.n_feature_columns} "
            f"feature columns); up to {self.n_rows_per_model:,} rows per model.",
            f"Each model: {self.steps_per_model:,} steps of about {self.rows_per_step:,.0f} rows "
            f"({self.targets_per_step:,.0f} targets), {self.n_passes_per_row:.2f} passes per row, a validation score "
            f"{validation_text}{stop_text}.",
            f"Reads {100 * self.fraction_of_table_read:.1f}% of the training rows; about "
            f"{100 * self.fraction_of_rows_as_targets:.1f}% serve as targets.",
        ]
        if self.training_hours is None:
            time_text = f"Time: not estimated on {self.instance_type}"
        else:
            time_text = (
                f"Time: {bound} {self.training_hours:,.2f} h at {self.seconds_per_step:.3g} s a step on "
                f"{self.instance_type}, {self.gpus} model(s) at a time"
            )
        if self.estimated_cost_usd is not None:
            time_text += (
                f", {bound} ${self.estimated_cost_usd:,.2f} (us-east-1 on-demand prices of {self.prices_as_of})"
            )
        max_hours_text = "none" if self.max_training_hours is None else f"{self.max_training_hours:g} h"
        lines.append(f"{time_text}; cap {max_hours_text}.")
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


class TaskScore(BaseModel):
    """One task of a job of tasks: its table and rows, and the MSE of the base and of the fine-tuned model on its
    held-out rows (or its validation-channel rows), in the task's own standardized target units."""

    table_path: str = Field("", description="The path of the task's table; empty for the whole channel.")
    task: Optional[Union[int, float, str]] = Field(
        None, description="The task's value of its table's task_column; null without one."
    )
    n_rows: int = Field(description="Rows of the training table that belong to this task.")
    n_val_rows: int = Field(
        description="This task's rows scored before and after: its held-out rows, or its rows of the validation channel."
    )
    val_mse_before: Optional[float] = Field(description="The base model's MSE on this task's scored rows.")
    val_mse_after: Optional[float] = Field(
        description="The fine-tuned model's MSE on the same rows; null when it was not scored."
    )


class TrainingReport(BaseModel):
    """``training_report.json``: what a fine-tuning job did and how its result scores.

    ``val_mse_before`` and ``val_mse_after`` are the mean squared errors of the base model and of the fine-tuned
    model on the same rows, in standardized target units. ``child_curves`` has one entry per child that started or
    failed, with that child's own validation scores: a view of progress, not the result. ``n_rows`` counts the rows
    of the training table and ``n_rows_used`` the rows training could draw from: the table minus any rows set aside
    to score the report. With ``bag`` 1 and no validation channel, the child trains on four fifths of these rows and
    is scored on up to 4,096 rows of the other fifth.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    base_model: str
    base_model_version: str
    n_rows: int
    n_rows_used: int
    n_features: int
    children: int
    steps_run: List[int]
    # A plain str, so a report naming a validation mode added later still parses.
    validation_source: str
    n_val_rows: int
    val_mse_before: Optional[float]
    val_mse_after: Optional[float]
    child_curves: List[ChildCurve]
    # True when a stop request ended the job before training finished.
    cut_short: bool
    seconds: float
    instance_type: str
    # Training steps between two validation scores in a child's curve.
    validation_interval: int = 1
    # Seconds the job's first read of the training table took.
    read_seconds: float = 0.0
    # Training rows the children's steps used, summed over the children; a row used in two passes counts twice.
    rows_visited: int = 0
    # rows_visited over the sum of every planned model's share, failed and unstarted models included: 1.0 after one
    # pass each.
    fraction_visited: float = 0.0
    # Training steps the longest of the children's passes over their shares of this table takes.
    steps_per_pass: int = 0
    workers: int = Field(
        1,
        description=(
            "How many worker processes trained the children at the same time; 1 trains them one after the other."
        ),
    )
    gpus: int = Field(1, description="How many GPUs the children trained on; 0 on a CPU job.")
    failed_children: List[int] = Field(
        [],
        description=(
            "The children that failed and were left out of the fine-tuned model; their curves carry the failure."
        ),
    )
    training_plan: Optional[TrainingPlan] = Field(
        None, description="The plan the job worked out before training; null when no model could train."
    )
    stop_reason: Optional[StopReason] = Field(
        None,
        description=(
            "What ended the training: stop_request when a stop request ended any model or the job, else "
            "max_training_hours when the cap ended a model or left one unstarted, else non_finite_loss when a model's "
            "training loss stopped being finite, else patience when every model that trained stopped early, else "
            "plan_done; null in a report from an earlier job."
        ),
    )
    n_passes_per_row_completed: Optional[float] = Field(
        None,
        description=(
            "How many times the models read each row of their shares, on average over every model the plan set, those "
            "that stopped early, failed or did not start included; null in a report from an earlier job."
        ),
    )
    seconds_per_step: Optional[float] = Field(
        None,
        description=(
            "Seconds a training step took, averaged over the models that trained in this job, with its share of the "
            "validation scores, as the plan's seconds_per_step counts it."
        ),
    )
    fraction_of_rows_as_targets: Optional[float] = Field(
        None,
        description=(
            "Share of the training rows used as a prediction target in at least one training step, the steps after a "
            "model's best validation score included; null in a report from an earlier job."
        ),
    )
    unstarted_children: List[int] = Field(
        [], description="The models max_training_hours left unstarted; they are not in the fine-tuned model."
    )
    n_rows_unread: Optional[int] = Field(
        None,
        description=(
            "Rows of the models' shares that no model read: the share of a model that failed or did not start, and "
            'the rest of a share whose model stopped before reading all of it; null under "folds" with 3 or more '
            "models when it cannot be counted, and in a report from an earlier job."
        ),
    )
    bag: Optional[int] = Field(None, description="How many models the job trained, as the plan sets it.")
    row_partition_strategy: Optional[RowPartitionStrategy] = Field(
        None, description="How the training rows were divided among the models, as the plan sets it."
    )
    tasks: List[TaskScore] = Field(
        [],
        description=(
            "One entry per task in a job of tasks, table by table, each table's tasks in ascending task_column order; "
            "val_mse_before and val_mse_after then combine the tasks' MSEs, each task weighted by one over its MSE "
            "before, and rows_visited, fraction_visited and steps_per_pass are 0. Empty otherwise."
        ),
    )

    def get_summary(self) -> str:
        """The outcome as three lines of text.

        The lines give the models that joined the fine-tuned model, the row partition, the job's stop reason and how
        many models trained, did not start or failed; each model's steps, the passes per row, the share of the training
        rows used as targets and the rows unread; and the validation MSE before and after, with the seconds in all and
        per step. A figure the report lacks prints as n/a.
        """
        trained = [curve for curve in self.child_curves if not curve.failed and not curve.restored]
        steps = ", ".join(f"{curve.child}: {self.steps_run[i]:,}" for i, curve in enumerate(self.child_curves))
        before = "n/a" if self.val_mse_before is None else f"{self.val_mse_before:.6g}"
        after = "n/a" if self.val_mse_after is None else f"{self.val_mse_after:.6g}"
        per_step = "n/a" if self.seconds_per_step is None else f"{self.seconds_per_step:.3g} s"
        passes = "n/a" if self.n_passes_per_row_completed is None else f"{self.n_passes_per_row_completed:.2f}"
        unread = "n/a" if self.n_rows_unread is None else f"{self.n_rows_unread:,}"
        targets = (
            "n/a" if self.fraction_of_rows_as_targets is None else f"{100 * self.fraction_of_rows_as_targets:.1f}%"
        )
        lines = [
            f"Report: {self.children} of {self.bag or self.children} model(s) joined the fine-tuned model "
            f"({self.row_partition_strategy or 'folds'}); stop reason {self.stop_reason or 'not reported'}; "
            f"{len(trained)} trained, {len(self.unstarted_children)} unstarted, {len(self.failed_children)} failed.",
            f"Steps by model: {steps or 'none'}; {passes} passes per row; "
            f"{targets} of the training rows served as targets; "
            f"{unread} rows unread.",
            f"Validation MSE {before} before, {after} after; {self.seconds:,.1f} s in all, {per_step} a step.",
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------ fine-tuning requests and results
#: The training instance when the request names none: the cheapest one the 24 GB bases fit.
_DEFAULT_TRAINING_INSTANCE_TYPE = "ml.g5.xlarge"
#: The job's disk when the request names none, on an instance type without local storage: the image's working files
#: and the artifact fit with room to spare. The Algorithm's instance types have local storage, which holds the tables.
_DEFAULT_VOLUME_SIZE_GB = 30
#: Bases that need a 48 GB GPU (LARGE_GPU_INSTANCE_TYPES); the job would fail on a 24 GB one after billing it.
_LARGE_GPU_BASES = frozenset({"nori-100m"})
#: SageMaker's limit on one hyperparameter value, in characters.
_MAX_HYPERPARAMETER_CHARS = 2500
#: The extensions, in any case, of a channel given as one object: the job reads Parquet files.
_CHANNEL_FILE_SUFFIXES = (".parquet", ".pq")
#: The extension, in any case, of a CSV object, which a channel cannot hold.
_CSV_SUFFIX = ".csv"
#: The job skips a file whose name starts with one of these, so a one-object channel so named holds no table.
_CHANNEL_SKIPPED_NAME_PREFIXES = ("_", ".")


def serialize_hyperparameters(hyperparameters: FineTuneHyperparameters) -> Dict[str, str]:
    """The job's HyperParameters: every set field as the string SageMaker delivers to the job.

    Every non-None field is sent, defaults included, so the console and ``DescribeTrainingJob`` show every value
    the job ran with. A field whose default is not None and whose value is None (``patience=None``,
    ``max_training_hours=None``) goes as the string ``"none"``, which turns the setting off. ``feature_columns``,
    ``tasks`` and row-count mappings go as JSON, so a column name may hold a comma; a task table sends only the
    settings it sets. The job parses them back with the same schema (``FineTuneHyperparameters``).

    It lives here, beside :class:`FineTuneJobRequest`, because the request's length check measures exactly these
    strings when it is built, and this module cannot import the fine-tuner that sends them.
    """
    strings: Dict[str, str] = {}
    for name, value in hyperparameters.model_dump().items():
        if value is None:
            if FineTuneHyperparameters.model_fields[name].default is not None:
                strings[name] = _NONE_HYPERPARAMETER
            continue
        if name == "tasks":
            # Only what each table sets, so more tables fit in one hyperparameter value.
            value = [{key: item for key, item in table.items() if item is not None} for table in value]
        strings[name] = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
    return strings


def _check_channel_uri(value: str, argument: str) -> None:
    """Refuse a channel URI that is not an S3 folder ending in ``/`` or one ``.parquet`` or ``.pq`` object.

    SageMaker's S3Prefix channel reads every key that starts with the given text, so a folder named without its
    ``/`` also reads look-alike siblings (``data`` reads ``data-old/``), and the job reads only Parquet files
    (``.parquet`` or ``.pq``). A ``.csv`` object is refused with the way to write the table as Parquet. The job
    skips a file whose name starts with ``_`` or ``.``, so one object so named is also refused.
    """
    parse_s3_uri(value, name=argument)
    if value.lower().endswith(_CSV_SUFFIX):
        raise ValueError(
            f"{argument} names a CSV object; got {value!r}. The job reads Parquet only: write the table with "
            "pandas.DataFrame.to_parquet(path, index=False) and upload that .parquet object or a folder of them"
        )
    if not (value.endswith("/") or value.lower().endswith(_CHANNEL_FILE_SUFFIXES)):
        raise ValueError(
            f"{argument} must be an S3 folder ending in '/' or one .parquet or .pq object; got {value!r}. SageMaker "
            "reads every key that starts with the given text, so a folder without its '/' also reads look-alike "
            "siblings, and the job reads only .parquet and .pq files"
        )
    if not value.endswith("/") and value.rsplit("/", 1)[-1].startswith(_CHANNEL_SKIPPED_NAME_PREFIXES):
        raise ValueError(
            f"{argument} names one object whose name starts with '_' or '.'; got {value!r}. The job skips such files, "
            "like the markers and checksums Spark and Hadoop write next to a table, so it would find no rows: rename "
            "the object"
        )


class FineTuneJobRequest(BaseModel):
    """One fine-tuning job: the Parquet tables, what to train, and where.

    Everything SageMaker or the job would refuse only after an instance started billing is checked when the request
    is built: the S3 URIs (each channel a folder of Parquet files or one Parquet object, never a CSV object), the job
    name, the instance a base needs, and SageMaker's limit on a hyperparameter
    value's length. ``hyperparameters`` is a copy of what the caller passed, so changing the caller's object
    afterwards changes nothing here;
    :meth:`~synthefy.SynthefyNoriFineTuner.fine_tune` runs every check again before it submits the job.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    algorithm: str = Field(
        min_length=1,
        description=(
            "The fine-tuning Algorithm: its ARN, which a Marketplace subscription gives, or its name for an "
            "Algorithm in the caller's own account."
        ),
    )
    training_data: str = Field(
        description="The train channel: an S3 folder ending in '/' of Parquet files, or one .parquet or .pq object."
    )
    validation_data: Optional[str] = Field(
        None,
        description=(
            "The optional validation channel, in the same form: rows the job validates on instead of rows it sets "
            "aside from the training table."
        ),
    )
    output_path: str = Field(description="The S3 folder SageMaker writes <job name>/output/model.tar.gz under.")
    hyperparameters: FineTuneHyperparameters
    instance_type: Optional[str] = Field(
        None,
        validate_default=True,
        description=(
            "The training instance; by default the first 48 GB type for nori-100m and ml.g5.xlarge otherwise."
        ),
    )
    role_arn: str = Field(min_length=1, description="The execution role SageMaker assumes to run the job.")
    job_name: Optional[str] = Field(
        None,
        description=(
            "By default nori-<model_name[:35]>-<UTC timestamp>-<6 random characters>, chosen when fine_tune() runs."
        ),
    )
    max_runtime_s: Optional[int] = Field(
        None,
        ge=1,
        validate_default=True,
        description=(
            "Seconds after which SageMaker stops the job: by default max_training_hours x 3,600 + "
            f"{_RUNTIME_MARGIN_S:,}, and {SAGEMAKER_MAX_RUNTIME_S:,} when max_training_hours is none; a value you set "
            "is sent as it is, and fine_tune() and get_training_plan() refuse one below that sum unless the "
            "Algorithm's version trains each model for a fixed number of steps. On that version the default is "
            f"ceil(bag / N) x 1,350 + {_RUNTIME_MARGIN_S:,} on an instance type with N GPUs, at most "
            f"{SAGEMAKER_MAX_RUNTIME_S:,}."
        ),
    )
    volume_size_gb: int = Field(
        _DEFAULT_VOLUME_SIZE_GB,
        ge=1,
        description=(
            "The job's disk in GB on an instance type without local storage. The instance types the Algorithm offers have "
            "at least 250 GB of local storage, which SageMaker uses instead and which holds the tables and the artifact."
        ),
    )
    kms_key_id: Optional[str] = Field(None, description="The KMS key SageMaker encrypts the artifact with.")
    checkpoint_uri: Optional[str] = Field(
        None,
        description=(
            "An S3 folder ending in '/' where the job saves each model it finishes and each model's training state "
            "as it goes. fine_tune() needs it empty, or deletes its files with overwrite=True; resume() continues a "
            "job from the files it saved there."
        ),
    )

    @field_validator("training_data", "validation_data")
    @classmethod
    def _check_channel(cls, value: Optional[str], info: ValidationInfo) -> Optional[str]:
        if value is not None:
            _check_channel_uri(value, info.field_name)
        return value

    @field_validator("output_path")
    @classmethod
    def _check_output_path(cls, value: str) -> str:
        parse_s3_uri(value, name="output_path")
        return value

    @field_validator("checkpoint_uri")
    @classmethod
    def _check_checkpoint_uri(cls, value: Optional[str]) -> Optional[str]:
        # A folder of its own: SageMaker copies everything under it into the job, which refuses any file it did not
        # write, so a bucket's root would cost a refused job.
        if value is not None:
            parse_s3_uri(value, name="checkpoint_uri", require_prefix=True)
            if not value.endswith("/"):
                raise ValueError(f"checkpoint_uri must be an S3 folder ending in '/'; got {value!r}")
        return value

    @field_validator("hyperparameters")
    @classmethod
    def _check_hyperparameters(cls, value: FineTuneHyperparameters) -> FineTuneHyperparameters:
        if value.feature_columns is not None and not value.feature_columns:
            raise ValueError(
                "feature_columns must name at least one column, or be left unset to use every column but the target"
            )
        if value.feature_columns is not None and value.target_column in value.feature_columns:
            # The job refuses this too, but only once the instance is running.
            raise ValueError(
                f"feature_columns includes the target column {value.target_column!r}; drop it, or leave "
                "feature_columns unset to use every column but the target"
            )
        for name, text in serialize_hyperparameters(value).items():
            if len(text) > _MAX_HYPERPARAMETER_CHARS:
                remedy = {
                    "feature_columns": "name fewer columns, or leave feature_columns unset to use every column but "
                    "the target",
                    "tasks": "list fewer task tables, or stack tables of one target into one table with a task_column",
                }.get(name, "shorten it")
                raise ValueError(
                    f"{name} is {len(text):,} characters as sent to SageMaker, which allows at most "
                    f"{_MAX_HYPERPARAMETER_CHARS:,} per hyperparameter value; {remedy}"
                )
        # A snapshot, so the instance type and runtime derived below stay true of what the job is sent.
        return value.model_copy(deep=True)

    @field_validator("instance_type")
    @classmethod
    def _resolve_instance_type(cls, value: Optional[str], info: ValidationInfo) -> Optional[str]:
        hyperparameters = info.data.get("hyperparameters")
        if hyperparameters is None:
            return value
        needs_large_gpu = hyperparameters.base_model in _LARGE_GPU_BASES
        if value is None:
            return LARGE_GPU_INSTANCE_TYPES[0] if needs_large_gpu else _DEFAULT_TRAINING_INSTANCE_TYPE
        if needs_large_gpu and value not in LARGE_GPU_INSTANCE_TYPES:
            raise ValueError(
                f"base_model={hyperparameters.base_model!r} needs a 48 GB GPU, which {value!r} does not have; "
                f"use instance_type {' or '.join(repr(name) for name in LARGE_GPU_INSTANCE_TYPES)}"
            )
        return value

    @field_validator("job_name")
    @classmethod
    def _check_job_name(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            check_sagemaker_resource_name(value, "job_name")
        return value

    # Whether max_runtime_s came from the hyperparameters, so a check run again derives it again from them.
    _max_runtime_derived: bool = PrivateAttr(False)

    @model_validator(mode="wrap")
    @classmethod
    def _record_whether_max_runtime_is_derived(cls, data: Any, handler: Any) -> "FineTuneJobRequest":
        """The validated request; its ``_max_runtime_derived`` is True when the fields it was built from leave
        ``max_runtime_s`` unset, so the request derives it."""
        derived = isinstance(data, dict) and data.get("max_runtime_s") is None
        request = handler(data)
        if isinstance(data, dict):
            request._max_runtime_derived = derived
        return request

    @model_validator(mode="after")
    def _refuse_checkpoints_for_a_job_of_tasks(self) -> "FineTuneJobRequest":
        """The request unchanged; raises ValueError when ``checkpoint_uri`` is set for a job of tasks
        (:func:`is_job_of_tasks`), which saves no model to continue from (the job refuses it too, once it runs)."""
        if self.checkpoint_uri is not None and is_job_of_tasks(self.hyperparameters):
            raise ValueError(
                "checkpoint_uri cannot be set for a job of tasks (tasks, task_column, time_column, context_rows or "
                "query_rows): it saves no model to continue from; leave checkpoint_uri unset"
            )
        return self

    @field_validator("max_runtime_s")
    @classmethod
    def _resolve_max_runtime(cls, value: Optional[int], info: ValidationInfo) -> Optional[int]:
        hyperparameters = info.data.get("hyperparameters")
        if value is not None or hyperparameters is None:
            return value
        if hyperparameters.max_training_hours is None:
            return SAGEMAKER_MAX_RUNTIME_S
        return get_runtime_for_training_cap(hyperparameters.max_training_hours)


class FineTuneJob(BaseModel):
    """One training job as SageMaker describes it at the moment of the call; every method returns a fresh one."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    job_name: str
    job_arn: str
    status: str = Field(
        description=(
            "InProgress, Completed, Failed, Stopping or Stopped; a plain string so a status AWS adds later "
            "still parses."
        )
    )
    secondary_status: Optional[str] = Field(None, description="SageMaker's finer step, e.g. Starting or Training.")
    status_message: Optional[str] = Field(
        None, description="What SageMaker says about the latest secondary status, e.g. that a job waits for capacity."
    )
    created_at: datetime
    ended_at: Optional[datetime] = None
    artifact_uri: Optional[str] = Field(None, description="The job's model.tar.gz, once SageMaker uploaded it.")
    failure_reason: Optional[str] = None
    latest_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "The last value SageMaker read for each metric: one child's latest validated step, from whichever child "
            "finished last once the job ends (the child metric names which), progress only, not a score of the "
            "fine-tuned model (get_training_report() has that)."
        ),
    )
    algorithm: str = Field(description="The Algorithm ARN or name the job ran, which deploy() packages with.")
    model_name: Optional[str] = Field(None, description="The job's model_name hyperparameter: the name predict takes.")
    billable_seconds: Optional[int] = None
    training_plan: Optional[TrainingPlan] = Field(
        None,
        description=(
            "What the job reads and what it takes, when fine_tune() or resume() worked it out before submitting. For "
            "a job resume() started it is the plan of the whole request, the most that job takes: it trains only "
            "what its checkpoint folder does not hold."
        ),
    )


class FineTunedEndpoint(BaseModel):
    """An asynchronous SageMaker endpoint serving one fine-tuned model, and every resource deploy() used for it.

    :meth:`~synthefy.SynthefyNoriFineTuner.create_client` builds a SageMaker-mode client bound to it.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    endpoint_name: str
    model_name: str = Field(description="The name predict() takes: the training job's model_name.")
    instance_type: str
    status: str = Field(description="InService after deploy(wait=True); Creating after deploy(wait=False).")
    model_package_arn: str
    sagemaker_model_name: str
    endpoint_config_name: str
    async_output_uri: str = Field(description="The S3 folder the endpoint writes outputs/ and failures/ under.")
    cut_short: bool = Field(
        description=(
            "The job was stopped before its children finished, so the model trained less than asked "
            "(the report's cut_short)."
        )
    )
    region_name: str
