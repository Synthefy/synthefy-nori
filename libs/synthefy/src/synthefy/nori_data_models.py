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

The fine-tuning models at the end of this file (``FineTuneHyperparameters``, ``ChildCurve`` and
``TrainingReport``) and the constants beside them describe the Nori fine-tuning Algorithm: the
hyperparameters a job takes and the ``training_report.json`` it writes. CI checks them against the
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
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from synthefy._aws import check_sagemaker_resource_name, format_seconds, parse_s3_uri
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
RELEASED_MODEL_NAMES = ("nori-6m", "nori-30m", "nori-30m-thinking-medium", "nori-100m")

#: The released models a fine-tuning job accepts as ``base_model``.
BaseModelName = Literal["nori-6m", "nori-100m"]

#: The fine-tuning Algorithm's 48 GB training instance types; ``nori-100m`` needs one of them.
LARGE_GPU_INSTANCE_TYPES = ("ml.g6e.xlarge", "ml.g6e.2xlarge")

#: The job's input channels: the training table and the optional validation table.
TRAIN_CHANNEL = "train"
VALIDATION_CHANNEL = "validation"

#: The report inside a fine-tuning job's ``model.tar.gz``.
REPORT_FILENAME = "training_report.json"


class FineTuneHyperparameters(BaseModel):
    """The Nori fine-tuning Algorithm's hyperparameters; each also parses from the string SageMaker delivers."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    target_column: str = Field(min_length=1, description="The column of the training table to predict.")
    model_name: str = Field(
        description=f"The name the fine-tuned model answers to (the request's model): {FINE_TUNED_MODEL_NAME_RULE}."
    )
    base_model: BaseModelName = Field("nori-6m", description="The released Nori model to fine-tune.")
    task: Literal["regression"] = Field("regression", description="Only regression is supported.")
    feature_columns: Optional[List[str]] = Field(
        None, description="Comma-separated or JSON list of feature columns; default every other column."
    )
    bag: int = Field(
        8,
        ge=1,
        le=8,
        description=(
            "The number of children the job trains, 1 to 8. The fine-tuned model is built from them, and each "
            "child has its own steps and time budget."
        ),
    )
    # The Algorithm's defaults; CI fails if they drift.
    steps: int = Field(100, ge=1, description="Maximum training steps per child.")
    lr: float = Field(1e-5, gt=0, description="Learning rate.")
    patience: int = Field(25, ge=1, description="A child stops after this many steps without improvement.")
    budget_s: float = Field(900.0, gt=0, description="Wall-clock cap per child, in seconds.")
    # The Algorithm refuses a negative seed; refusing here names the field.
    seed: int = Field(0, ge=0, description="The job's random seed.")

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


class TrainingReport(BaseModel):
    """``training_report.json``: what a fine-tuning job did and how its result scores.

    ``val_mse_before`` and ``val_mse_after`` are the mean squared errors of the base model and of the fine-tuned
    model on the same rows, in standardized target units. ``child_curves`` has one entry per child that started,
    with that child's own validation scores: a view of progress, not the result.
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


# ------------------------------------------------------------------ fine-tuning requests and results
#: The training instance when the request names none: the cheapest one the 24 GB bases fit.
_DEFAULT_TRAINING_INSTANCE_TYPE = "ml.g5.xlarge"
#: The job's disk when the request names none: the channels and the image's working files fit with room to spare.
_DEFAULT_VOLUME_SIZE_GB = 30
#: Bases that need a 48 GB GPU (LARGE_GPU_INSTANCE_TYPES); the job would fail on a 24 GB one after billing it.
_LARGE_GPU_BASES = frozenset({"nori-100m"})
#: SageMaker's limit on one hyperparameter value, in characters.
_MAX_HYPERPARAMETER_CHARS = 2500
#: The derived MaxRuntimeInSeconds: every child's budget, plus half again and 30 minutes for what the job does
#: besides training.
_RUNTIME_MARGIN_FACTOR = 1.5
_RUNTIME_MARGIN_S = 1800


def serialize_hyperparameters(hyperparameters: FineTuneHyperparameters) -> Dict[str, str]:
    """The job's HyperParameters: every set field as the string SageMaker delivers to the job.

    Every non-None field is sent, defaults included, so the console and ``DescribeTrainingJob`` show every value
    the job ran with. ``feature_columns`` goes as a JSON list, so a column name may hold a comma. The job parses
    them back with the same schema (``FineTuneHyperparameters``).

    It lives here, beside :class:`FineTuneJobRequest`, because the request's length check measures exactly these
    strings when it is built, and this module cannot import the fine-tuner that sends them.
    """
    strings: Dict[str, str] = {}
    for name, value in hyperparameters.model_dump(exclude_none=True).items():
        strings[name] = json.dumps(value) if name == "feature_columns" else str(value)
    return strings


def _check_channel_uri(value: str, argument: str) -> None:
    """Refuse a channel URI that is not an S3 folder ending in ``/`` or one ``.csv`` object.

    SageMaker's S3Prefix channel reads every key that starts with the given text, so a folder named without its
    ``/`` also reads look-alike siblings (``data`` reads ``data-old/``), and the job reads only ``.csv`` files.
    """
    parse_s3_uri(value, name=argument)
    if not (value.endswith("/") or value.lower().endswith(".csv")):
        raise ValueError(
            f"{argument} must be an S3 folder ending in '/' or one .csv object; got {value!r}. SageMaker reads "
            "every key that starts with the given text, so a folder without its '/' also reads look-alike "
            "siblings, and the job reads only .csv files"
        )


class FineTuneJobRequest(BaseModel):
    """One fine-tuning job: the tables, what to train, and where.

    Everything SageMaker or the job would refuse only after an instance started billing is checked when the request
    is built: the S3 URIs, the job name, the instance a base needs, a runtime long enough for the bag, and
    SageMaker's limit on a hyperparameter value's length. ``hyperparameters`` is a
    copy of what the caller passed, so changing the caller's object afterwards changes nothing here;
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
    training_data: str = Field(description="The train channel: an S3 folder ending in '/' or one .csv object.")
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
        None, description="By default nori-<model_name[:42]>-<UTC timestamp>, chosen when fine_tune() runs."
    )
    max_runtime_s: Optional[int] = Field(
        None,
        ge=1,
        validate_default=True,
        description=(
            f"SageMaker stops the job after this; by default ceil({_RUNTIME_MARGIN_FACTOR} x bag x budget_s) + "
            f"{_RUNTIME_MARGIN_S}."
        ),
    )
    volume_size_gb: int = Field(
        _DEFAULT_VOLUME_SIZE_GB, ge=1, description="The job's disk; SageMaker copies the channels onto it."
    )
    kms_key_id: Optional[str] = Field(None, description="The KMS key SageMaker encrypts the artifact with.")

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
                remedy = (
                    "name fewer columns, or leave feature_columns unset to use every column but the target"
                    if name == "feature_columns"
                    else "shorten it"
                )
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

    @field_validator("max_runtime_s")
    @classmethod
    def _resolve_max_runtime(cls, value: Optional[int], info: ValidationInfo) -> Optional[int]:
        hyperparameters = info.data.get("hyperparameters")
        if hyperparameters is None:
            return value
        floor = hyperparameters.bag * hyperparameters.budget_s
        if value is None:
            return math.ceil(_RUNTIME_MARGIN_FACTOR * floor) + _RUNTIME_MARGIN_S
        if value < floor:
            shown = format_seconds(floor)
            raise ValueError(
                f"max_runtime_s={value} is below bag x budget_s={shown}, so SageMaker would stop the job before "
                "every child has had its budget. Lower budget_s or bag, or raise max_runtime_s. The job also needs "
                f"time besides training; the derived default adds {round((_RUNTIME_MARGIN_FACTOR - 1) * 100)} % plus "
                f"{_RUNTIME_MARGIN_S // 60} min for that."
            )
        return value


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
            "The last value SageMaker read for each metric, which is the last child's last validated step: "
            "progress only, not a score of the fine-tuned model (get_training_report() has that)."
        ),
    )
    algorithm: str = Field(description="The Algorithm ARN or name the job ran, which deploy() packages with.")
    model_name: Optional[str] = Field(None, description="The job's model_name hyperparameter: the name predict takes.")
    billable_seconds: Optional[int] = None


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
