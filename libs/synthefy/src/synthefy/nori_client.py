"""Standalone client for Synthefy Nori in-context regression.

Synthefy Nori is an in-context learning regressor: each call supplies labeled
context rows (``X_train``, ``y_train``) and query rows (``X_test``), and the model
returns one predicted value per query row in a single forward pass -- there is no
training step. The same forward pass also carries a full predictive distribution,
so ``predict(output_type="quantiles", quantiles=[...])`` returns calibrated
prediction intervals at no extra cost.

This module uses the package-wide exception types and HTTP error handling from
:mod:`synthefy.errors` so errors behave consistently across transports.

A single :class:`SynthefyNoriClient` runs predictions in one of three modes,
selected with the ``mode`` constructor argument:

- ``"remote"`` (default) -- calls the hosted Baseten endpoint over HTTPS.
- ``"sagemaker"`` -- invokes a named Amazon SageMaker endpoint with AWS Signature V4,
  using boto3's standard credential chain.
- ``"local"`` -- runs the same prediction in-process via the optional
  ``synthefy-nori`` package (``pip install synthefy-nori``), no network and
  no API key.
"""

import importlib.util
import json
import os
import threading
import time
import warnings
from numbers import Integral, Real
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import httpx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from synthefy.featurize import (
    CATEGORICAL_AUTO as _CATEGORICAL_AUTO,
    DEFAULT_CATEGORICAL_ENCODING as _DEFAULT_CATEGORICAL_ENCODING,
    DEFAULT_MAX_CARDINALITY as _DEFAULT_MAX_CARDINALITY,
    align_and_featurize as _align_and_featurize,
)
from synthefy.data_models import NoriPredictRequest, NoriPredictResponse
from synthefy.nori_data_models import (
    DEFAULT_LARGE_CONTEXT_SEED,
    DEFAULT_LARGE_CONTEXT_THRESHOLD,
    LargeContextPolicy,
    LargeContextReport,
    MAX_LARGE_CONTEXT_SEED,
    MAX_LARGE_CONTEXT_THRESHOLD,
    MemoryPolicyInput,
)
from synthefy.errors import (
    APIConnectionError,
    APITimeoutError,
    _raise_for_status,
)

# Gateway endpoint (default): routes to the model by name, body carries "model".
GATEWAY_BASE_URL = "https://inference.baseten.co"
GATEWAY_ENDPOINT = "/predict"

# Environment variable holding the hosted-Nori API key.
NORI_API_KEY_ENV = "SYNTHEFY_NORI_API_KEY"

# Sentinel for a required ``model=`` (there is no default -- every caller names a size).
# Explicit ``None`` is rejected too: model identity is part of every transport contract.
_MODEL_REQUIRED: Any = object()

# Model registry. Maps a ``model=`` selector -> ``(remote_gateway_slug, local_variant)``:
#   key                 = what the caller passes as ``model=`` (a friendly name or a raw gateway slug)
#   remote_gateway_slug = the "model" string sent in the gateway request body (remote mode)
#   local_variant       = the name forwarded to synthefy-nori's ``model=`` selector (local mode)
# Every selector names its size -- there is no bare "nori"/"synthefy/nori", so a slug never silently
# changes which model it serves (``model=`` is required; see the constructor). "nori-6m" is the ~6M
# base; "nori-30m" is the ~29.2M variant; "nori-100m" is the ~98.3M variant. The raw gateway slugs
# are listed too so they load the right checkpoint locally instead of being treated as a raw HF repo.
# The "nori-30m-thinking-medium" entries are the test-time-compute variant: hosted-API only,
# so they map a remote gateway slug but have NO local variant -- the thinking guard in __init__
# refuses it in mode="local" (its ``local_variant`` below is therefore never consulted).
# A friendly name/slug absent here that reaches local mode has no local checkpoint and is refused
# rather than silently running a different model (see ``_resolve_local_variant``).
NORI_VARIANTS = {
    "nori-6m": ("synthefy/nori-6m", "nori-6m"),
    "nori-30m": ("synthefy/nori-30m", "nori-30m"),
    "nori-100m": ("synthefy/nori-100m", "nori-100m"),
    "synthefy/nori-6m": ("synthefy/nori-6m", "nori-6m"),
    "synthefy/nori-30m": ("synthefy/nori-30m", "nori-30m"),
    "synthefy/nori-100m": ("synthefy/nori-100m", "nori-100m"),
    # Thinking (test-time compute) -- hosted deployments only; local is refused by the
    # thinking guard. Medium is the only released Thinking budget.
    "nori-30m-thinking-medium": ("synthefy/nori-30m-thinking-medium", None),
    "synthefy/nori-30m-thinking-medium": ("synthefy/nori-30m-thinking-medium", None),
}

# Selectable model names for error messages, derived from NORI_VARIANTS so they stay current as
# variants are added. Friendly names only (the "synthefy/..." raw slugs are aliases, not listed).
_MODEL_NAMES = tuple(name for name in NORI_VARIANTS if "/" not in name)

# SageMaker Marketplace publishes these exact named inference specifications. A SageMaker
# endpoint is created from one specification; the client sends the same canonical name so the
# container can fail closed if an endpoint and requested model do not match.
SAGEMAKER_VARIANTS = tuple(_MODEL_NAMES)


def _is_thinking_model(model: Optional[str]) -> bool:
    """Return ``True`` if ``model`` names a Nori Thinking (test-time-compute) variant.

    The Thinking variant (gateway slug ``"synthefy/nori-30m-thinking-medium"``) spends extra
    inference to lift accuracy and run **only** on the hosted API -- there is no local checkpoint
    for it. Matching on the ``"thinking"`` token covers both the friendly and slug spellings.
    """
    return model is not None and "thinking" in model.lower()


def _resolve_variant(model: Optional[str]) -> tuple:
    """Map a model selector to ``(gateway_model, local_variant)``.

    A known name or slug resolves via :data:`NORI_VARIANTS`; anything else (a custom gateway
    slug) passes through as the gateway model. The ``local_variant`` is what local mode would
    forward to synthefy-nori's ``model=``
    selector, but whether a selector is actually runnable locally is enforced by
    :func:`_resolve_local_variant`, not here -- this function never raises.
    """
    if model is not None and model in NORI_VARIANTS:
        return NORI_VARIANTS[model]
    return model, None


def _canonical_model_name(model: Optional[str]) -> Optional[str]:
    """Return the canonical deployment name without the gateway namespace."""
    gateway_model, _ = _resolve_variant(model)
    if gateway_model is None:
        return None
    return gateway_model.removeprefix("synthefy/")


def _resolve_local_variant(model: Optional[str]) -> Optional[str]:
    """Resolve the synthefy-nori ``model=`` value for local inference, or raise if impossible.

    ``"nori-6m"``/``"synthefy/nori-6m"`` run the ~6M base checkpoint; ``"nori-30m"``/
    ``"synthefy/nori-30m"`` run the 29.2M checkpoint; ``"nori-100m"``/``"synthefy/nori-100m"``
    run the 98.3M checkpoint. ``None`` forwards no ``model=`` (so
    synthefy-nori, which itself requires an explicit model, would raise). Any other selector has
    no local checkpoint, so this
    raises :class:`ValueError` instead of silently falling back to the base model -- a Nori
    Thinking variant gets a message pointing at the hosted API, everything else a message listing
    the locally runnable options.
    """
    if _is_thinking_model(model):
        raise ValueError(
            f"model={model!r} is a Nori Thinking (test-time-compute) variant, which runs only "
            "on the hosted Synthefy API and has no local checkpoint. Use mode='remote' with a "
            "Baseten API key to run Thinking, or select 'nori-6m'/'nori-30m'/'nori-100m' for "
            "local inference."
        )
    if model is None or model in NORI_VARIANTS:
        return _resolve_variant(model)[1]
    raise ValueError(
        f"model={model!r} has no local checkpoint and cannot run in mode='local'. Local "
        "inference supports 'nori-6m', 'nori-30m' and 'nori-100m'. For hosted-only "
        "variants (e.g. Nori Thinking) or a custom deployment slug, use mode='remote' with a "
        "Baseten API key."
    )


DEFAULT_TASK = "regression"

# What ``predict`` returns from the model's predictive distribution. Shared names
# have the same meanings as ``synthefy_nori.NoriRegressor.predict``'s
# ``output_type``, so a selector behaves consistently locally and remotely.
#   point         -> one value per query row (a summary of the distribution)
#   distribution  -> the quantile function itself (prediction intervals, CRPS, ...)
_POINT_OUTPUT_TYPES = ("mean", "median")
_DISTRIBUTION_OUTPUT_TYPES = ("quantiles", "full")
_OUTPUT_TYPES = _POINT_OUTPUT_TYPES + _DISTRIBUTION_OUTPUT_TYPES
DEFAULT_OUTPUT_TYPE = "mean"

Mode = Literal["remote", "local", "sagemaker"]
_VALID_MODES = ("remote", "local", "sagemaker")

# AWS Marketplace's SageMaker endpoint request-body limit. The live runtime accepts exactly
# 25,000,000 bytes and returns HTTP 413 at 25,000,001 for response-stream invocations; the
# operation-specific API reference still shows the older 6 MiB value. Check the current service
# limit locally so a caller gets a deterministic error before signing or sending a paid request.
SAGEMAKER_MAX_BODY_BYTES = 25_000_000

# Authorization header scheme for remote requests. The Baseten inference *gateway* accepts only
# ``Bearer``, which is why it is the default. ``Api-Key`` exists for a caller who points
# ``base_url`` at some other Baseten host that expects that scheme.
AuthScheme = Literal["Bearer", "Api-Key"]
_VALID_AUTH_SCHEMES = ("Bearer", "Api-Key")
DEFAULT_AUTH_SCHEME: AuthScheme = "Bearer"

# Array-like inputs accepted by ``predict`` -- nested Python sequences, numpy
# arrays, or pandas DataFrames/Series (all coerced to plain numeric arrays).
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray, pd.DataFrame]
VectorLike = Union[Sequence[float], np.ndarray, pd.Series, pd.DataFrame]


def _load_aws_sdk() -> Tuple[Any, Any]:
    """Load the optional AWS SDK only when a SageMaker deployment is selected."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError('A SageMaker deployment needs the AWS extra: install `pip install "synthefy[aws]"`.') from exc
    return boto3, Config


def _create_sagemaker_runtime_client(
    *,
    region_name: Optional[str],
    timeout: float,
    max_retries: int,
    user_agent_extra: str,
) -> Any:
    """Create a SageMaker Runtime client through boto3's credential chain.

    Deliberately constructs an argument-free ``Session``: environment variables,
    shared config/credentials, web identity (including GitHub OIDC), ECS/EC2 role
    credentials, and SSO profiles retain boto3's normal precedence. The public
    client never accepts raw AWS access keys.
    """
    boto3, Config = _load_aws_sdk()
    config = Config(
        connect_timeout=timeout,
        read_timeout=timeout,
        retries={"mode": "standard", "total_max_attempts": max_retries + 1},
        user_agent_extra=user_agent_extra,
    )
    return boto3.Session().client(
        "sagemaker-runtime",
        region_name=region_name,
        config=config,
    )


def _target_name(y_train: Any) -> Any:
    """Name for the ``as_pandas`` output Series, taken from ``y_train``.

    Uses the ``Series.name`` or the single-column ``DataFrame``'s column label;
    falls back to ``"prediction"`` when ``y_train`` carries no name (lists/arrays).
    """
    if isinstance(y_train, pd.Series):
        return y_train.name if y_train.name is not None else "prediction"
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        return y_train.columns[0]
    return "prediction"


def _result_index(X_test: Any) -> Optional[Any]:
    """Index for the ``as_pandas`` output, copied from ``X_test`` when it is a
    pandas object so predictions join straight back; ``None`` (default RangeIndex)
    otherwise."""
    if isinstance(X_test, (pd.DataFrame, pd.Series)):
        return X_test.index
    return None


def _reject_non_numeric_columns(frame: pd.DataFrame, name: str) -> None:
    """Raise ``ValueError`` if any column is not numeric.

    This helper runs after public DataFrame preprocessing, so any remaining
    categorical/text/temporal column means the schema was not resolved.
    """
    non_numeric = [str(col) for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col])]
    if non_numeric:
        raise ValueError(
            f"{name} has unresolved non-numeric column(s) {non_numeric}. Pass raw "
            "features as DataFrames and choose categorical_columns=[...] or "
            "text_columns=[...], or pre-encode them numerically."
        )


def _coerce_matrix(arr: MatrixLike, name: str) -> np.ndarray:
    """Coerce an array-like into a 2D float ``np.ndarray`` or raise ``ValueError``.

    Accepts nested Python sequences, numpy arrays, and pandas DataFrames. A
    pandas DataFrame is checked for non-numeric columns first (so the caller
    gets a clear message rather than a cryptic float-cast error). NaN/missing
    values are preserved and forwarded for server-side imputation.
    """
    if isinstance(arr, pd.DataFrame):
        _reject_non_numeric_columns(arr, name)
        matrix = arr.to_numpy(dtype=float)
    else:
        try:
            matrix = np.asarray(arr, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{name} must be a numeric 2D array/list with equal-length rows; "
                f"got error: {exc}. If it has categorical/string columns, pass a "
                "pandas DataFrame (with both X_train and X_test as DataFrames) so "
                "they can be one-hot encoded."
            ) from exc
    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must be 2D with shape (n_rows, n_features); got {matrix.ndim}D with shape {matrix.shape}"
        )
    return matrix


def _coerce_vector(arr: VectorLike, name: str) -> np.ndarray:
    """Coerce an array-like into a 1D float ``np.ndarray`` or raise ``ValueError``.

    Accepts nested Python sequences, numpy arrays, a pandas Series, or a
    single-column pandas DataFrame. NaN/missing values are preserved and
    forwarded for server-side imputation.
    """
    if isinstance(arr, pd.DataFrame):
        if arr.shape[1] != 1:
            raise ValueError(
                f"{name} must be 1D; got a DataFrame with {arr.shape[1]} "
                "columns. Pass a single column (a Series) for the targets."
            )
        _reject_non_numeric_columns(arr, name)
        vector = arr.to_numpy(dtype=float).reshape(-1)
    elif isinstance(arr, pd.Series):
        if not pd.api.types.is_numeric_dtype(arr):
            raise ValueError(f"{name} must be numeric; got a non-numeric Series (dtype {arr.dtype}).")
        vector = arr.to_numpy(dtype=float)
    else:
        try:
            vector = np.asarray(arr, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} must be a numeric 1D array/list; got error: {exc}") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be 1D with shape (n_rows,); got {vector.ndim}D with shape {vector.shape}")
    return vector


def _build_nori_request(
    X_train: MatrixLike,
    y_train: VectorLike,
    X_test: MatrixLike,
    task: str = DEFAULT_TASK,
    categorical_columns: Any = _CATEGORICAL_AUTO,
    max_categorical_cardinality: int = _DEFAULT_MAX_CARDINALITY,
    categorical_encoding: str = _DEFAULT_CATEGORICAL_ENCODING,
    output_type: str = DEFAULT_OUTPUT_TYPE,
    quantile_levels: Optional[List[float]] = None,
    memory_policy: Optional[MemoryPolicyInput] = None,
    large_context_policy: Optional[LargeContextPolicy] = None,
    large_context_threshold: Optional[int] = None,
    large_context_seed: Optional[int] = None,
) -> NoriPredictRequest:
    """Validate shapes and build a :class:`NoriPredictRequest`.

    Accepts Python lists, numpy arrays, or pandas DataFrames/Series. When both
    ``X_train`` and ``X_test`` are DataFrames, ``X_test`` is aligned to
    ``X_train``'s columns *by name* (so column order is irrelevant), and a
    mismatch in the column sets raises ``ValueError``; then any non-numeric
    columns are encoded (fit on ``X_train``, applied to ``X_test`` —
    ``categorical_encoding`` picks ordinal codes or one-hot indicators, see
    :func:`synthefy.featurize.align_and_featurize`) so the request carries a
    fully numeric matrix.
    Otherwise columns are matched positionally, as before. Raises ``ValueError``
    on any shape mismatch before a request leaves the process. NaN/missing
    values are preserved and imputed server-side.

    ``output_type``/``quantile_levels`` are expected to have been validated
    already (by :func:`_validate_output_type`); a default ``output_type`` leaves
    both fields unset so the request body is unchanged from earlier versions.
    """
    X_train, X_test = _align_and_featurize(
        X_train,
        X_test,
        max_categorical_cardinality,
        categorical_columns=categorical_columns,
        categorical_encoding=categorical_encoding,
        _warning_stacklevel=5,
    )

    X_train_arr = _coerce_matrix(X_train, "X_train")
    X_test_arr = _coerce_matrix(X_test, "X_test")
    y_train_arr = _coerce_vector(y_train, "y_train")

    n_context, n_features = X_train_arr.shape
    if n_context == 0:
        raise ValueError("X_train must contain at least one context row")
    if n_features == 0:
        raise ValueError("X_train must contain at least one feature column")
    if y_train_arr.shape[0] != n_context:
        raise ValueError(f"X_train has {n_context} rows but y_train has {y_train_arr.shape[0]}; they must match")
    if X_test_arr.shape[0] == 0:
        raise ValueError("X_test must contain at least one query row")
    if X_test_arr.shape[1] != n_features:
        raise ValueError(f"X_test has {X_test_arr.shape[1]} features but X_train has {n_features}; they must match")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    # All fields, including the large-context trio, are passed to the constructor
    # together rather than assigned onto an already-built request afterward: pydantic
    # validates a constructor call's fields as one complete set, so
    # NoriPredictRequest's "threshold/seed require a policy" model_validator sees a
    # coherent object on its first and only run here. Sequential post-construction
    # attribute assignment (with validate_assignment=True) instead validates after
    # EVERY individual assignment, so it only works if the caller assigns policy
    # before threshold/seed -- an ordering invariant that would otherwise be enforced
    # by nothing but a comment.
    return NoriPredictRequest(
        X_train=_nullable_matrix(X_train_arr),
        y_train=y_train_arr.tolist(),
        X_test=_nullable_matrix(X_test_arr),
        task=task,
        # Left as None for the default so the serialized body carries neither
        # field (see _predict_remote's exclude_none dump).
        output_type=None if output_type == DEFAULT_OUTPUT_TYPE else output_type,
        quantiles=quantile_levels,
        memory_policy=memory_policy,
        large_context_policy=large_context_policy,
        large_context_threshold=large_context_threshold,
        large_context_seed=large_context_seed,
    )


def _as_float_list(values: Any) -> List[float]:
    """Coerce a prediction result (list or numpy array) into a flat ``list[float]``."""
    return np.asarray(values, dtype=float).reshape(-1).tolist()


def _nullable_matrix(
    values: NDArray[np.floating],
) -> List[List[Optional[float]]]:
    """Encode non-finite feature cells as JSON-compatible nulls."""
    return np.where(np.isfinite(values), values, None).tolist()


def _load_local_predict() -> Any:
    """Lazily import ``synthefy_nori.predict`` with a helpful error if absent."""
    try:
        from synthefy_nori import predict as local_predict
    except ModuleNotFoundError as exc:
        if exc.name != "synthefy_nori":
            raise
        raise ImportError(
            "Local nori inference requires the optional 'synthefy-nori' "
            "package. Install it with: pip install synthefy-nori."
        ) from exc
    return local_predict


def _validate_output_type(
    output_type: str,
    quantiles: Optional[VectorLike],
    *,
    discretizing: bool,
) -> Optional[List[float]]:
    """Check ``output_type``/``quantiles`` and normalize the tau levels.

    The rules are ``NoriRegressor.predict``'s, enforced here so a bad argument
    raises before any expensive work (text embedding, a network round-trip, or a
    checkpoint load) rather than after it. Returns the tau levels as a plain
    ``list[float]`` in the **caller's order** — order is meaningful, since it is
    the order of the returned rows (``lo, mid, hi = ...``) — or ``None`` when no
    levels were given.
    """
    if output_type not in _OUTPUT_TYPES:
        raise ValueError(f"output_type must be one of {_OUTPUT_TYPES}; got {output_type!r}.")
    if discretizing and (output_type != DEFAULT_OUTPUT_TYPE or quantiles is not None):
        raise ValueError(
            "categorical output (discretize=/categorical_levels=) returns "
            "discrete labels, so it combines only with the default "
            f"output_type={DEFAULT_OUTPUT_TYPE!r} (the discretize= strategy "
            "chooses the summary), not with output_type/quantiles."
        )
    if quantiles is not None and output_type != "quantiles":
        raise ValueError(f"quantiles= is only valid with output_type='quantiles'; got output_type={output_type!r}.")
    if output_type != "quantiles":
        return None
    if quantiles is None:
        raise ValueError(
            "output_type='quantiles' requires quantiles=[...] with at least one "
            "tau level in (0, 1), e.g. quantiles=[0.1, 0.5, 0.9] for an 80% "
            "interval around the median."
        )
    levels = np.asarray(quantiles, dtype=float).reshape(-1)
    if levels.size == 0:
        raise ValueError(
            "output_type='quantiles' requires quantiles=[...] with at least one "
            "tau level in (0, 1); got an empty sequence."
        )
    if not np.all(np.isfinite(levels)) or np.any((levels <= 0.0) | (levels >= 1.0)):
        raise ValueError(f"quantiles must lie strictly in (0, 1); got {quantiles!r}.")
    return [float(level) for level in levels]


def _load_local_regressor() -> Any:
    """Lazily import ``synthefy_nori.NoriRegressor`` with a helpful error if absent.

    Needed for every ``output_type`` other than ``"mean"``: the functional
    ``synthefy_nori.predict`` cannot reach them, because it forwards ``**kwargs``
    to the ``NoriRegressor`` *constructor* and only passes ``discretize`` /
    ``categorical_levels`` through to ``predict``.
    """
    try:
        from synthefy_nori import NoriRegressor
    except ModuleNotFoundError as exc:
        if exc.name != "synthefy_nori":
            raise
        raise ImportError(
            "Local nori inference requires the optional 'synthefy-nori' "
            "package. Install it with: pip install synthefy-nori."
        ) from exc
    return NoriRegressor


def _local_discretize_available() -> bool:
    """Return ``True`` if the installed ``synthefy-nori`` supports discretization.

    The ``discretize=`` / ``categorical_levels=`` arguments need a build that
    ships the ``synthefy_nori.discretize`` module; probing with ``find_spec``
    avoids importing (and thus loading) the package.
    """
    return importlib.util.find_spec("synthefy_nori.discretize") is not None


def _local_memory_policy_available() -> bool:
    """Return ``True`` if the installed ``synthefy-nori`` accepts ``memory_policy=``.

    The policy landed in synthefy-nori 0.13.0 as ``synthefy_nori.inference.memory_policy``,
    so the module's presence is the capability.

    A signature probe would NOT work here, unlike the ``model=`` check further down:
    ``synthefy_nori.predict`` forwards ``**kwargs`` to ``NoriRegressor``, so ``memory_policy``
    never appears in its parameters on any version, and gating on that would reject
    every build including new ones.
    """
    try:
        return importlib.util.find_spec("synthefy_nori.inference.memory_policy") is not None
    except ModuleNotFoundError:
        # find_spec on a dotted path imports the PARENT package to read its __path__, so
        # when synthefy-nori is not installed at all (the base synthefy install, the
        # common case this check exists for) it raises here instead of returning None.
        return False


def _local_large_context_available() -> bool:
    """Return ``True`` if the installed ``synthefy-nori`` accepts ``large_context_policy=``.

    The feature landed as ``synthefy_nori.inference.large_context``, so the module's
    presence is the capability -- a signature probe on ``NoriRegressor.__init__`` would
    NOT reliably work here, for the same reason ``_local_memory_policy_available`` doesn't
    use one: a future ``NoriRegressor`` could forward ``large_context_policy`` via
    ``**kwargs`` instead of naming it in its signature, which a probe would falsely reject.
    """
    try:
        return importlib.util.find_spec("synthefy_nori.inference.large_context") is not None
    except ModuleNotFoundError:
        # Same reason as _local_memory_policy_available: find_spec on a dotted path
        # imports the parent package first, which raises instead of returning None
        # when synthefy-nori is not installed at all.
        return False


def _validate_large_context_controls(
    *,
    policy: Optional[LargeContextPolicy],
    threshold: Optional[int],
    seed: Optional[int],
    output_type: str,
    model: Optional[str],
    mode: str,
) -> None:
    """Fail before preprocessing, checkpoint load, or a paid hosted request.

    ``threshold``/``seed`` are ``None`` when the caller did not pass them (mirroring
    ``NoriPredictRequest``'s own ``Optional[int] = None`` wire fields), not a
    resolved default -- so an explicit value equal to the default is still
    distinguishable from "omitted" here, unlike comparing against the default value.
    """
    if policy is None:
        if threshold is not None or seed is not None:
            # Both are otherwise silently dropped from the wire request: the caller
            # almost certainly meant to also pass large_context_policy=. Same rule,
            # same wording, as NoriPredictRequest's own model_validator and the
            # server's _parse_large_context, so the client fails exactly where the
            # request would have anyway -- before a paid hosted round-trip.
            raise ValueError(
                "large_context_threshold/large_context_seed require "
                f"large_context_policy; got threshold={threshold!r}, seed={seed!r} "
                "with no policy selected. Pass large_context_policy=..., or omit "
                "both large_context_threshold and large_context_seed."
            )
        return
    if not isinstance(policy, str) and (mode != "local" or not callable(policy)):
        raise ValueError(
            "large_context_policy must be a policy name string; custom callables are supported only in local mode."
        )
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, Integral)
        or not 1 <= int(threshold) <= MAX_LARGE_CONTEXT_THRESHOLD
    ):
        raise ValueError(
            "large_context_threshold must be an integer between 1 and "
            f"{MAX_LARGE_CONTEXT_THRESHOLD:,}; got {threshold!r}."
        )
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= int(seed) <= MAX_LARGE_CONTEXT_SEED
    ):
        raise ValueError(
            f"large_context_seed must be an integer between 0 and {MAX_LARGE_CONTEXT_SEED:,}; got {seed!r}."
        )
    if output_type in _DISTRIBUTION_OUTPUT_TYPES:
        raise ValueError(
            f"large_context_policy={policy!r} cannot serve "
            f"output_type={output_type!r}: routed policies combine point "
            "predictions from several Nori calls and have no combined "
            "predictive distribution. Use output_type='mean' or 'median', or "
            "omit large_context_policy."
        )
    if _is_thinking_model(model):
        raise ValueError(
            "large_context_policy is not available on Nori Thinking "
            "(test-time-compute) variants. Use a base nori-6m or nori-30m "
            "model, or omit the policy."
        )


def _normalized_large_context_report(request: NoriPredictRequest, report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Give local mode the same typed per-call report as hosted modes."""
    policy = request.large_context_policy
    if policy is None:
        raise RuntimeError("large-context report requested without a policy")
    threshold = (
        request.large_context_threshold
        if request.large_context_threshold is not None
        else DEFAULT_LARGE_CONTEXT_THRESHOLD
    )
    seed = request.large_context_seed if request.large_context_seed is not None else DEFAULT_LARGE_CONTEXT_SEED
    if report is None:
        return LargeContextReport(
            applied=False,
            policy=(policy if isinstance(policy, str) else getattr(policy, "__name__", repr(policy))),
            threshold=threshold,
            seed=seed,
            reason="below_threshold",
            window=None,
            n_train=len(request.X_train),
            n_test=len(request.X_test),
            shards_available=None,
            nori_calls=0,
            full_context=None,
            reused_train_state=False,
        ).model_dump()
    return LargeContextReport.model_validate(
        {
            **report,
            "applied": True,
            "threshold": threshold,
            "seed": seed,
            "reason": None,
        }
    ).model_dump()


# The one discretization strategy computable from the hosted endpoint's
# response: it returns only point predictions (the distribution mean), and
# "snap-mean" is by definition the nearest level to that mean — so snapping
# client-side gives the same labels local mode would. Every other strategy
# reads the full predictive distribution and needs local mode.
_REMOTE_DISCRETIZE_METHOD = "snap-mean"


def _resolve_remote_levels(
    y_train: List[float],
    discretize: Optional[str],
    categorical_levels: Optional[VectorLike],
) -> "np.ndarray":
    """Validate remote discretization arguments and resolve the level lattice.

    Called before the network request so an unsupported strategy or bad level
    set fails fast instead of after a paid inference round-trip.
    """
    if discretize is None:
        raise ValueError(
            "categorical_levels without discretize= implies the package "
            'default strategy ("map-cell"), which needs the full predictive '
            "distribution — the hosted endpoint returns only point "
            'predictions. Pass discretize="snap-mean" explicitly (nearest '
            "level to the point prediction), or use local mode "
            "(pip install synthefy-nori) for the full strategy set."
        )
    if discretize != _REMOTE_DISCRETIZE_METHOD:
        raise ValueError(
            f"discretize={discretize!r} needs the full predictive "
            "distribution, which the hosted endpoint does not return; "
            'remote mode supports discretize="snap-mean" (nearest level to '
            "the returned point prediction). For the full strategy set, use "
            "local mode (pip install synthefy-nori)."
        )
    if categorical_levels is None:
        levels = np.unique(np.asarray(y_train, dtype=float))
        levels = levels[np.isfinite(levels)]
        if levels.size == 0:
            raise ValueError(
                "y_train has no finite values to derive categorical levels from; pass categorical_levels explicitly."
            )
    else:
        levels = np.unique(np.asarray(categorical_levels, dtype=float).reshape(-1))
        if levels.size == 0 or not np.all(np.isfinite(levels)):
            raise ValueError(
                f"categorical_levels must be a non-empty sequence of finite numbers; got {categorical_levels!r}"
            )
    return levels


def _snap_to_levels(predictions: List[float], levels: "np.ndarray") -> List[float]:
    """Snap point predictions onto the level lattice."""
    preds = np.asarray(predictions, dtype=float)
    snapped = preds.copy()
    finite = np.isfinite(preds)
    # A NaN prediction stays NaN rather than becoming a confident label.
    nearest = np.abs(preds[finite, None] - levels[None, :]).argmin(axis=1)
    snapped[finite] = levels[nearest]
    return snapped.tolist()


# --------------------------------------------------------------------------- #
# Distribution output shaping (output_type="quantiles" / "full")
# --------------------------------------------------------------------------- #


def _quantile_frame(
    values: "np.ndarray",
    levels: Sequence[float],
    *,
    X_test: Any,
    y_train: Any,
) -> pd.DataFrame:
    """Build the ``as_pandas`` quantile frame: one row per query row, one column
    per tau level.

    Columns are named ``"<target>[<level>]"`` — the same convention the
    forecasting client uses for its quantile columns, so both halves of the
    package label quantiles the same way. Indexed by ``X_test`` when it is a
    pandas object, so the bands join straight back onto the query rows.
    """
    name = _target_name(y_train)
    return pd.DataFrame(
        values,
        index=_result_index(X_test),
        columns=[f"{name}[{level}]" for level in levels],
        dtype=float,
    )


def _nullable_rows_to_array(rows: Sequence[Sequence[Optional[float]]]) -> "np.ndarray":
    """Coerce the wire's nullable quantile rows to a 2D float array.

    JSON cannot carry ``NaN``, so the server sends ``null`` for a non-finite
    quantile; ``np.asarray(..., dtype=float)`` maps those back to ``NaN`` (the
    value the local package would have returned) rather than leaving ``None``
    objects in a numeric result.
    """
    try:
        arr = np.asarray(rows, dtype=float)
    except (ValueError, TypeError) as exc:
        # Ragged rows: numpy >= 2 rejects an inhomogeneous shape. Report it as a
        # malformed response rather than letting the raw numpy message surface.
        raise ValueError(f"The server returned a malformed quantile block (rows of unequal length): {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(
            "The server returned a malformed quantile block: expected a 2D "
            f"(n_query, n_levels) array, got shape {arr.shape}."
        )
    return arr


def _shape_quantiles(
    q_by_row: "np.ndarray",
    levels: Sequence[float],
    *,
    as_pandas: bool,
    X_test: Any,
    y_train: Any,
) -> Union[List[List[float]], pd.DataFrame]:
    """Shape an ``(n_query, n_levels)`` block into the ``output_type="quantiles"`` result.

    The default result is transposed to ``(n_levels, n_query)`` — level-major —
    which is ``NoriRegressor.predict``'s shape and what makes
    ``lo, mid, hi = client.predict(..., quantiles=[0.1, 0.5, 0.9])`` work. The
    pandas result stays row-major, because a DataFrame indexed by the query rows
    is what joins back onto the data.
    """
    if as_pandas:
        return _quantile_frame(q_by_row, levels, X_test=X_test, y_train=y_train)
    return q_by_row.T.tolist()


def _shape_full(
    q_by_row: "np.ndarray",
    taus: "np.ndarray",
    mean: "np.ndarray",
    *,
    as_pandas: bool,
    X_test: Any,
    y_train: Any,
) -> Dict[str, Any]:
    """Shape the whole quantile bank into the ``output_type="full"`` result.

    Mirrors ``NoriRegressor.predict(output_type="full")``'s dict — ``"quantiles"``
    row-major ``(n_query, K)``, ``"taus"`` ``(K,)``, ``"mean"`` ``(n_query,)`` —
    with lists (or pandas objects under ``as_pandas``) in place of numpy arrays,
    matching how the rest of this client returns data.
    """
    if as_pandas:
        return {
            "quantiles": _quantile_frame(q_by_row, taus.tolist(), X_test=X_test, y_train=y_train),
            "taus": taus.tolist(),
            "mean": pd.Series(
                mean,
                index=_result_index(X_test),
                name=_target_name(y_train),
                dtype=float,
            ),
        }
    return {
        "quantiles": q_by_row.tolist(),
        "taus": taus.tolist(),
        "mean": mean.tolist(),
    }


def _resolve_text_device(device: Optional[str]) -> str:
    """Resolve the device for a named sentence encoder.

    ``None`` and ``"auto"`` prefer a CUDA device (including PyTorch ROCm
    builds), then Apple MPS, and otherwise use CPU. An explicit PyTorch device
    string such as ``"cpu"`` or ``"cuda:1"`` is returned unchanged.
    """
    if device not in (None, "auto"):
        if not isinstance(device, str) or not device.strip():
            raise ValueError(
                "text_device must be 'auto', None, or a non-empty PyTorch "
                "device string such as 'cpu', 'cuda', 'cuda:1', or 'mps'."
            )
        return device

    try:
        import torch
    except (ImportError, OSError):
        return "cpu"

    cuda = getattr(torch, "cuda", None)
    try:
        if cuda is not None and cuda.is_available():
            return "cuda"
    except (AttributeError, RuntimeError):
        pass

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    try:
        if mps is not None and mps.is_available():
            return "mps"
    except (AttributeError, RuntimeError):
        pass

    return "cpu"


def _widen_text_columns(
    X_train,
    X_test,
    text_columns,
    svd_dim,
    embedder,
    max_cardinality,
    categorical_encoding,
    text_device,
    categorical_columns=_CATEGORICAL_AUTO,
):
    """Prepare named text columns client-side, returning numeric frames.

    The shared DataFrame preprocessor fits text SVD and categorical mappings on
    ``X_train``. Both inputs must be DataFrames; their indexes are preserved for
    ``as_pandas`` output.
    """
    resolved_text_device = _resolve_text_device(text_device) if isinstance(embedder, str) else None
    return _align_and_featurize(
        X_train,
        X_test,
        max_cardinality,
        categorical_columns=categorical_columns,
        categorical_encoding=categorical_encoding,
        text_columns=text_columns,
        svd_dim=svd_dim,
        embedder=embedder,
        text_device=resolved_text_device,
    )


def _has_declared_text_columns(text_columns: Any) -> bool:
    """Return whether the public declaration names at least one text column."""
    if text_columns is None:
        return False
    if isinstance(text_columns, str):
        return True
    try:
        return len(text_columns) > 0
    except TypeError:
        return True


class SynthefyNoriClient:
    """Client for Synthefy Nori in-context regression.

    Each :meth:`predict` call performs in-context regression: the labeled context
    rows are supplied alongside the query rows and one value per query row is
    returned in a single forward pass.

    The ``mode`` argument selects how predictions run:

    - ``"remote"`` (default): call the hosted Baseten endpoint over HTTPS.
      Requires an API key (``api_key`` argument or the
      ``SYNTHEFY_NORI_API_KEY`` environment variable), sent as
      ``Authorization: <auth_scheme> <key>`` (``Bearer`` by default).
    - ``"local"``: run in-process via the optional ``synthefy-nori`` package
      (``pip install synthefy-nori``). No network and no API key.
    - ``"sagemaker"``: invoke a named Amazon SageMaker endpoint
      through boto3. Requests are SigV4-signed using boto3's standard credential
      chain; install the optional dependency with ``pip install "synthefy[aws]"``.

    For remote mode, the client targets the Baseten inference *gateway*
    (``https://inference.baseten.co/predict``) and includes the chosen size slug (e.g.
    ``"model": "synthefy/nori-30m"``) in the request body. The gateway resolves that slug
    to a deployment and authenticates with the ``Bearer`` scheme (the default
    ``auth_scheme``). ``base_url``, ``endpoint``, a custom ``model`` slug, and ``auth_scheme`` are
    available for pointing the client at some other host, but the gateway is the supported
    path and the only one Synthefy meters and rate-limits.

    Parameters
    ----------
    api_key : str or None, optional
        API key for hosted Nori (remote mode only). If ``None``, falls back to
        the ``SYNTHEFY_NORI_API_KEY`` environment variable. A
        :class:`ValueError` is raised if neither is set when remote mode is in
        effect.
    mode : {"remote", "local", "sagemaker"}, default "remote"
        How predictions run. See above.
    timeout : float, default 300.0
        Per-request timeout in seconds for remote HTTP. SageMaker uses it as botocore's
        per-read inactivity timeout; 15-second server heartbeat chunks allow an active stream
        to continue up to SageMaker's eight-minute service limit. Per-call overrides are not
        supported for SageMaker.
    max_retries : int, default 2
        Number of retries for transient errors (timeouts, connection errors,
        429 and 5xx responses). Remote mode uses exponential backoff; SageMaker
        configures botocore's standard retry policy with the same retry count.
    base_url : str, default GATEWAY_BASE_URL
        Base URL of the inference host (remote mode).
    endpoint : str, default GATEWAY_ENDPOINT
        Path appended to ``base_url`` for predictions (remote mode).
    model : str, REQUIRED
        Which Nori to run — there is no default; every request names a size. Pass a friendly
        size selector — ``"nori-6m"`` (the ~6M base) or ``"nori-30m"`` (the ~29.2M variant) —
        which selects both the remote gateway deployment and, in local mode, the checkpoint. A
        raw gateway slug (e.g. ``"synthefy/nori-30m"``) is also accepted verbatim. Omitting
        the argument or passing ``None`` raises :class:`ValueError`; every transport requires
        explicit model identity. Selecting a variant in local mode requires a
        synthefy-nori build with the ``model=`` selector.
        Nori Thinking Medium — ``"nori-30m-thinking-medium"`` — runs only on hosted
        deployments: passing it with ``mode="local"`` raises
        :class:`ValueError` rather
        than silently running the base model — use ``mode="remote"``. Likewise a selector with no
        local checkpoint (an unknown/custom slug) raises in local mode instead of falling back to
        the base model. SageMaker also requires the model name and verifies it
        against the model specification used to create the endpoint.
    auth_scheme : {"Bearer", "Api-Key"}, default "Bearer"
        HTTP ``Authorization`` scheme prefixed to the API key (remote mode). The
        inference gateway requires ``"Bearer"``; ``"Api-Key"`` is only for a
        caller-supplied ``base_url`` pointing at a host that expects that scheme.
    user_agent : str or None, optional
        Custom ``User-Agent`` value for HTTP remote mode. For SageMaker it is
        appended to botocore's normal user agent for request attribution.
    endpoint_name : str or None, optional
        SageMaker endpoint name. Required with ``mode="sagemaker"`` and
        invalid otherwise.
    region_name : str or None, optional
        AWS region for SageMaker. If omitted, boto3 resolves it from
        the standard environment/shared-config chain.

    Attributes
    ----------
    mode : str
        The explicitly selected execution mode.

    Examples
    --------
    >>> from synthefy import SynthefyNoriClient
    >>> client = SynthefyNoriClient(api_key="...", model="nori-30m")  # or SYNTHEFY_NORI_API_KEY
    >>> preds = client.predict(
    ...     X_train=[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    ...     y_train=[1.0, 1.0, 2.0],
    ...     X_test=[[2.0, 2.0]],
    ... )
    >>> len(preds)
    1

    Prediction intervals come from the same forward pass -- no conformal or
    quantile-regression add-ons:

    >>> lo, mid, hi = client.predict(          # doctest: +SKIP
    ...     X_train, y_train, X_test,
    ...     output_type="quantiles", quantiles=[0.1, 0.5, 0.9],
    ... )

    Run the same prediction locally (no API key, needs ``synthefy-nori``):

    >>> client = SynthefyNoriClient(mode="local", model="nori-30m")  # doctest: +SKIP

    Invoke a named SageMaker endpoint with ambient AWS credentials:

    >>> client = SynthefyNoriClient(  # doctest: +SKIP
    ...     mode="sagemaker", model="nori-30m",
    ...     endpoint_name="nori-30m-prod", region_name="us-east-1"
    ... )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        mode: Mode = "remote",
        timeout: float = 300.0,
        max_retries: int = 2,
        base_url: str = GATEWAY_BASE_URL,
        endpoint: str = GATEWAY_ENDPOINT,
        model: Any = _MODEL_REQUIRED,
        auth_scheme: AuthScheme = DEFAULT_AUTH_SCHEME,
        user_agent: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        if mode == "auto":
            raise ValueError(
                "mode='auto' has been removed; choose one explicit execution mode: 'remote', 'sagemaker', or 'local'"
            )
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}; got {mode!r}")
        if auth_scheme not in _VALID_AUTH_SCHEMES:
            raise ValueError(f"auth_scheme must be one of {_VALID_AUTH_SCHEMES}; got {auth_scheme!r}")
        if isinstance(timeout, bool) or not isinstance(timeout, Real) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite number greater than zero")
        if isinstance(max_retries, bool) or not isinstance(max_retries, Integral) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if mode == "sagemaker":
            if api_key is not None:
                raise ValueError(
                    "api_key is not used with mode='sagemaker'; boto3 resolves and "
                    "SigV4-signs with the standard AWS credential chain"
                )
            if not endpoint_name or not endpoint_name.strip():
                raise ValueError("endpoint_name is required with mode='sagemaker'")
        elif endpoint_name is not None or region_name is not None:
            raise ValueError("endpoint_name and region_name are only valid with mode='sagemaker'")
        if model is _MODEL_REQUIRED or model is None:
            raise ValueError(
                "model is required -- there is no default; every request names a size. "
                f"Choose one of: {', '.join(_MODEL_NAMES)} (a raw gateway slug is also accepted)."
            )
        self.mode: str = mode

        canonical_model = _canonical_model_name(model)
        if mode == "sagemaker" and canonical_model not in SAGEMAKER_VARIANTS:
            raise ValueError(
                "SageMaker model must name a published Nori inference specification; "
                f"choose one of: {', '.join(SAGEMAKER_VARIANTS)}"
            )
        self._sagemaker_model = canonical_model

        # Nori Thinking runs only on hosted backends; never silently substitute a local model.
        if _is_thinking_model(model) and mode == "local":
            raise ValueError(
                f"model={model!r} is a Nori Thinking (test-time-compute) variant, which runs only "
                f"on the hosted Synthefy API. Set mode='remote' with a Baseten API key to use it "
                "(mode='local' has no Thinking checkpoint)."
            )

        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.base_url = base_url
        self.endpoint = endpoint
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        # self.model is the gateway model id sent in the remote body: a friendly name maps to its
        # slug, and a raw slug passes through. In local mode we additionally resolve -- and
        # validate -- the local checkpoint selector, which raises for a selector that has no local
        # checkpoint rather than silently substituting the base model.
        self.model, self._local_variant = _resolve_variant(model)
        if mode == "local":
            self._local_variant = _resolve_local_variant(model)
        self.auth_scheme = auth_scheme
        self.user_agent = user_agent or f"synthefy-python httpx/{httpx.__version__}"
        self._aws_user_agent_extra = user_agent or "synthefy-python"
        self._aws_client: Optional[Any] = None
        self._local_regressor: Optional[Any] = None
        # Guards mutation of the cached local regressor's memory_policy/large_context_*
        # attributes and its fit/predict/report sequence, the same shared-mutable-state
        # hazard the server engine's own lock exists to prevent for its estimator.
        self._local_regressor_lock = threading.Lock()

        if mode == "sagemaker":
            self.api_key = None
            self.client = None
            self._aws_client = _create_sagemaker_runtime_client(
                region_name=self.region_name,
                timeout=self.timeout,
                max_retries=self.max_retries,
                user_agent_extra=self._aws_user_agent_extra,
            )
        elif mode == "remote":
            if api_key is None:
                api_key = os.getenv(NORI_API_KEY_ENV)
            if not api_key:
                raise ValueError(
                    "A Synthefy Nori API key must be provided either as the "
                    f"`api_key` argument or through the {NORI_API_KEY_ENV} "
                    "environment variable when mode='remote'"
                )
            self.api_key: Optional[str] = api_key
            self.client: Optional[httpx.Client] = httpx.Client(base_url=self.base_url)
        else:  # local
            self.api_key = api_key  # unused in local mode; may be None
            self.client = None

        #: What the server did about ``memory_policy=`` on the most recent :meth:`predict`, or
        #: ``None`` if that call did not set one. Mirrors the local package's
        #: ``NoriRegressor.memory_report_``: which fallback rung ran, the estimated and
        #: resident cache sizes, the query chunk, any dropped context rows, plus which fields
        #: the server clamped and any coherence notes about the policy sent.
        #:
        #: Worth reading, because the rung is decided by the replica's free VRAM rather than
        #: by the request -- it is not knowable from the client side.
        #:
        #: **Hosted modes only (remote/AWS).** In local mode the policy is honoured by the
        #: estimator owned by this client, but no report is copied here. Use ``NoriRegressor``
        #: directly and read ``memory_report_`` if you need the local report.
        self.last_memory_report: Optional[Dict[str, Any]] = None
        # Typed capability handshake for the most recent call that set a
        # large-context policy, in every mode. Cleared before every call so an
        # error or ordinary prediction cannot expose stale provenance.
        self.last_large_context_report: Optional[Dict[str, Any]] = None

    # Context manager support (sync) and utilities
    def __enter__(self) -> "SynthefyNoriClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        if self._aws_client is not None:
            try:
                self._aws_client.close()
            except Exception:
                pass
        with self._local_regressor_lock:
            self._local_regressor = None

    def predict(
        self,
        X_train: MatrixLike,
        y_train: VectorLike,
        X_test: MatrixLike,
        task: str = DEFAULT_TASK,
        *,
        as_pandas: bool = False,
        output_type: str = DEFAULT_OUTPUT_TYPE,
        quantiles: Optional[VectorLike] = None,
        categorical_columns: Any = _CATEGORICAL_AUTO,
        max_categorical_cardinality: int = _DEFAULT_MAX_CARDINALITY,
        categorical_encoding: str = _DEFAULT_CATEGORICAL_ENCODING,
        text_columns: Optional[Sequence[str]] = None,
        svd_dim: Optional[int] = 128,
        embedder: Any = "minilm",
        text_device: Optional[str] = "auto",
        discretize: Optional[str] = None,
        categorical_levels: Optional[VectorLike] = None,
        memory_policy: Optional[MemoryPolicyInput] = None,
        large_context_policy: Optional[LargeContextPolicy] = None,
        large_context_threshold: Optional[int] = None,
        large_context_seed: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Union[List[float], pd.Series, List[List[float]], pd.DataFrame, Dict[str, Any]]:
        """Predict a value for each query row via in-context regression.

        Parameters
        ----------
        X_train : array-like of shape (n_context, n_features)
            Labeled context rows. Python lists, numpy arrays, or a pandas
            DataFrame are accepted. In the DataFrame/DataFrame case, feature
            roles follow ``categorical_columns`` and ``text_columns``;
            otherwise all columns must be numeric. Missing values are allowed:
            NaN in a numeric column is imputed server-side; NaN in a categorical
            column stays NaN under ordinal encoding (imputed server-side) or
            becomes its own indicator under one-hot.
        y_train : array-like of shape (n_context,)
            Target value for each context row. A Python list, numpy array, or a
            pandas Series / single-column DataFrame is accepted.
        X_test : array-like of shape (n_query, n_features)
            Query rows to predict. Must have the same number of features as
            ``X_train``. When both ``X_train`` and ``X_test`` are DataFrames,
            ``X_test`` is aligned to ``X_train``'s columns *by name* (column
            order is irrelevant; a mismatch in the column sets raises), and any
            categorical columns are encoded using mappings fitted only on
            ``X_train``. Ordinal missing values remain NaN; rare and unseen
            values use one bounded ``other`` code. Unsupported temporal values
            and ambiguous high-cardinality strings raise with guidance instead
            of being silently dropped or embedded.
        task : str, default "regression"
            The prediction task. Currently only ``"regression"`` is supported.
        as_pandas : bool, default False
            If ``True``, return pandas objects instead of lists: a ``Series``
            for a point ``output_type`` (one value per ``X_test`` row) or a
            ``DataFrame`` for ``output_type="quantiles"``/``"full"`` (one row per
            ``X_test`` row, one column per tau level, named
            ``"<target>[<level>]"``). Either way the result is named after
            ``y_train`` (its ``Series`` name or single-column ``DataFrame``
            label, else ``"prediction"``) and indexed by ``X_test``'s index when
            ``X_test`` is a pandas object, so predictions join straight back.
            Default is plain ``list``/``dict``.
        output_type : {"mean", "median", "quantiles", "full"}, default "mean"
            What to return from the model's predictive distribution. Shared
            selectors have the same meanings as
            ``synthefy_nori.NoriRegressor.predict``:

            - ``"mean"`` (default) — the distribution mean, one value per query
              row. Optimal for squared error / R², and byte-for-byte the
              behavior of earlier client versions.
            - ``"median"`` — the distribution median (MAE-optimal), one value
              per query row.
            - ``"quantiles"`` — **prediction intervals**: the predictive
              quantiles at the levels in ``quantiles=``. Returns
              ``(n_levels, n_query)`` — level-major, so
              ``lo, mid, hi = client.predict(..., quantiles=[0.1, 0.5, 0.9])``
              unpacks directly.
            - ``"full"`` — the whole quantile bank as a dict with keys
              ``"quantiles"`` (``(n_query, K)`` ascending values), ``"taus"``
              (``(K,)`` levels) and ``"mean"`` (``(n_query,)``) — for CRPS /
              interval scoring and calibration work. Its ``"mean"`` is the bank's
              own mean, which can differ slightly from ``output_type="mean"``:
              the point path runs the model's augmentation (Yeo-Johnson)
              ensemble and the quantile bank deliberately does not. Inherited
              from ``NoriRegressor``, so it is the same in every mode.

            Quantiles come back in original-``y`` units, sorted to a valid
            (monotone) quantile function per row. Everything other than
            ``"mean"`` needs the model's predictive distribution, so it requires
            local mode or a hosted deployment that serves distribution output —
            see *Raises*.
        quantiles : array-like of float or None, optional
            Tau levels for ``output_type="quantiles"``, strictly inside
            ``(0, 1)`` — e.g. ``[0.1, 0.5, 0.9]`` for an 80% interval around the
            median. Required by (and valid only with) ``output_type="quantiles"``;
            the returned rows follow the order given here, so the order is
            yours to choose.
        categorical_columns : {"auto", None} or sequence of column names, default "auto"
            DataFrame categorical-feature policy. ``"auto"`` encodes every
            remaining non-numeric, non-text column. A sequence encodes exactly
            those columns and raises for any other non-numeric column. ``None``
            disables categorical inference. Explicit text and categorical
            declarations must not overlap.
        max_categorical_cardinality : int, default 100
            Maximum retained training levels per categorical. An automatically
            inferred column above the cap raises because it may be an identifier
            or free text. An explicitly named categorical keeps its top-K levels;
            rarer and unseen values share the bounded ``other`` value.
        categorical_encoding : {"ordinal", "onehot"}, default "ordinal"
            How non-numeric columns are encoded (DataFrame inputs only).
            ``"ordinal"`` maps each categorical column to one column of integer
            codes, with missing values kept as NaN and rare/unseen values mapped
            to one bounded ``other`` code;
            it benchmarked at least as well as one-hot across 35 categorical
            datasets and never widens the feature matrix. ``"onehot"``
            reproduces the previous client behavior (indicator columns per
            category, missing values get their own indicator).
        text_columns : sequence of str or None, optional
            Free-text columns to embed. When set (``X_train``/``X_test`` must be
            DataFrames), those columns are embedded by a frozen sentence encoder,
            reduced to ``svd_dim`` columns with a TruncatedSVD fit on ``X_train``,
            and appended as numeric features — the request still carries a fully
            numeric matrix, so every backend works unchanged. Needs the ``text``
            extra (``pip install "synthefy[text]"``). ``None`` (default) leaves
            text embedding disabled. It may not overlap ``categorical_columns``.
        svd_dim : int or None, default 128
            Number of SVD text columns appended (``None`` = full raw embedding).
            Ignored when ``text_columns`` is None.
        embedder : str, default "minilm"
            Sentence-encoder short name (e.g. ``"minilm"``, ``"qwen4b"``) for
            ``text_columns``. Ignored when ``text_columns`` is None.
        text_device : str or None, default "auto"
            Device for a named sentence encoder used by ``text_columns``.
            ``"auto"`` (and ``None``) prefers CUDA/ROCm, then Apple MPS, and
            falls back to CPU. Pass an explicit PyTorch device string such as
            ``"cpu"`` or ``"cuda:1"`` to override selection. Ignored when
            ``text_columns`` is None or ``embedder`` is a callable/preloaded
            encoder object, which controls its own placement.
        discretize : str or None, optional
            Declare a categorical/ordinal **target** (a 1–5 rating, a count, a
            quality score) and pick the strategy that maps each prediction
            onto the target's level lattice, so every returned value is one
            the target can actually take. Strictly opt-in: nothing is snapped
            unless ``discretize=`` or ``categorical_levels=`` is passed. In
            local mode the full strategy set of the installed ``synthefy-nori``
            is forwarded (``"map-cell"`` — accuracy-optimal, ``"median-cell"``
            — MAE-optimal, ``"snap-mean"``, ``"snap-median"``,
            ``"expected-level"``, ``"prior-match"``; see
            ``synthefy_nori.discretize``). In remote mode the hosted endpoint
            returns only point predictions, so ``"snap-mean"`` (nearest level
            to the point prediction — identical to local ``"snap-mean"``) is
            the one supported strategy; anything else raises ``ValueError``
            with guidance. A ``NaN`` prediction stays ``NaN`` after snapping.
        categorical_levels : array-like of float or None, optional
            The complete set of values the target can take — its label set,
            in classification terms. Values must be numeric; order and
            duplicates are irrelevant (the set is normalized to sorted
            distinct values). Defaults to the distinct values of
            ``y_train``, which is leak-safe; pass it explicitly when the
            context may under-cover the true scale (e.g. a 1–5 rating whose
            context has no 1s). Passing it alone activates discretization
            with the package default strategy in local mode (``"map-cell"``);
            remote mode requires ``discretize="snap-mean"`` explicitly.
        memory_policy : str, dict, or MemoryPolicy, optional
            Serving-memory policy, at parity with the local package's
            ``NoriRegressor(memory_policy=...)``. A preset name -- ``"exact"`` (never
            trade accuracy for memory), ``"max_context"`` (fit the largest table you
            can), ``"off"`` (no cache) -- a dict of individual fields, e.g.
            ``{"cache_dtype": "int8"}``, or a
            ``synthefy_nori.inference.memory_policy.MemoryPolicy`` instance if you have
            ``synthefy-nori`` installed. The model is the schema: this client does not
            redeclare the fields, so there is nothing here to drift from it.

            Nori does in-context regression, so your table is *input*: one prediction
            keeps a per-layer key/value cache over every context row, and that cache --
            not the model -- is what exhausts GPU memory on a big table. This decides what
            to do about it. Omit it for defaults that suit almost every request.

            Works in every mode. Remote/SageMaker, the policy is validated server-side and what it
            actually did comes back in :attr:`last_memory_report`; an incoherent policy is
            rejected before any inference is paid for. Local, it needs
            ``synthefy-nori >= 0.13.0`` and raises :class:`ImportError` with an upgrade
            hint on older builds.

            One field behaves differently over the network: ``elements_budget``. The cache
            is only built when the query set spans more than one chunk, which at default
            settings needs far more query rows than the hosted request-body limit allows
            (~64 MiB) -- so lowering ``elements_budget`` is what lets a hosted caller reach
            the cached path at all.
        large_context_policy : str, callable, or None
            Select context rows explicitly when `len(X_train)` exceeds
            `large_context_threshold`. Off by default. Remote and SageMaker
            modes forward a policy-name string unchanged; the server accepts
            every built-in in its installed Nori registry, including `random`,
            `cluster_route`, `cluster_route_g4`, `safeboost`, and `boost`.
            Parameter strings such as `safeboost[nu=0.25]` are supported.
            Local mode additionally accepts a custom callable.

            Point output only: `mean` and `median` are supported;
            `quantiles`/`full` and Nori Thinking variants fail before
            inference. After a call, `last_large_context_report` records
            whether the policy engaged, its resolved window and internal call
            count. Hosted modes require that report as a capability handshake.
        large_context_threshold : int or None, optional
            Context-row count strictly above which the selected policy engages.
            Valid range 1 through 10000000. `None` (the default) resolves to 50000
            once `large_context_policy` is set; passing it without a policy raises,
            rather than being silently dropped.
        large_context_seed : int or None, optional
            Deterministic policy seed in the range 0 through 2**32 - 1. `None` (the
            default) resolves to 0 once `large_context_policy` is set; passing it
            without a policy raises, rather than being silently dropped.

            Client calls are one-shot: every call supplies and fits `X_train`
            again. No hidden local or hosted cross-request context cache is
            created, and `large_context_cache_entries` is intentionally not a
            client option. Use `NoriRegressor` directly for explicit
            fit-once/predict-many local state.
        timeout : float or None, optional
            Override the client timeout for this request (remote mode only).
            It is ignored with a warning for SageMaker, where timeout is fixed
            on the boto3 client at construction.
        extra_headers : dict of str to str, optional
            Additional HTTP headers to send with the request (remote mode only;
            ignored in local mode and rejected for SageMaker deployments).

        Returns
        -------
        list of float, or pandas.Series if ``as_pandas=True``
            For a point ``output_type`` (``"mean"``/``"median"``):
            one predicted value per row of ``X_test``.
        list of list of float, or pandas.DataFrame if ``as_pandas=True``
            For ``output_type="quantiles"``: the quantiles as
            ``(n_levels, n_query)`` — level-major, matching
            ``NoriRegressor.predict`` — or, as pandas, a ``(n_query, n_levels)``
            frame with one column per level.
        dict
            For ``output_type="full"``: ``{"quantiles", "taus", "mean"}``, with
            lists (or a ``DataFrame`` and a ``Series`` under ``as_pandas=True``).

        Raises
        ------
        ValueError
            If the input shapes are inconsistent (e.g. ``X_train`` and
            ``y_train`` row counts differ, or ``X_test`` has a different number
            of features than ``X_train``); if DataFrame ``X_train``/``X_test``
            have mismatched column sets or duplicate column names; if declared
            categorical/text columns are missing or overlap; if an undeclared
            non-numeric column is ambiguous under an explicit/disabled policy;
            if an automatically inferred categorical exceeds the cardinality cap;
            if a column is
            numeric in one of ``X_train``/``X_test`` but not the other; if a
            column has unsupported ``timedelta`` dtype; if a non-DataFrame input
            contains non-numeric values; if ``categorical_encoding`` is not one
            of ``"ordinal"``/``"onehot"``; if featurization leaves no usable
            columns; if ``output_type`` is not one of the four names, or
            ``quantiles`` is missing/empty/outside ``(0, 1)`` for
            ``output_type="quantiles"``, or is passed with a different
            ``output_type``, or is combined with
            ``discretize=``/``categorical_levels=``; or, in remote mode, if
            ``discretize`` is a strategy other than ``"snap-mean"`` (or
            ``categorical_levels`` is passed without ``discretize=``, or is
            empty/non-finite), or if the hosted deployment does not serve the
            requested ``output_type`` (it echoes back the type it honored, so an
            ignored ``output_type`` raises here instead of silently returning
            means — use local mode, or a deployment with distribution support);
            or if HTTP-only ``extra_headers``/per-call ``timeout`` are passed in
            a SageMaker deployment. Per-call ``timeout`` is instead ignored
            with a warning for SageMaker.
        ImportError
            In local mode, if the optional ``synthefy-nori`` package is not
            installed (with guidance to ``pip install synthefy-nori``), or
            if it is too old for ``discretize=``/``categorical_levels=`` or for
            a non-default ``output_type`` (with an upgrade hint).
        NotImplementedError
            If the checkpoint cannot produce a predictive distribution — in
            local mode, ``output_type="quantiles"``/``"full"`` on a
            ``bar_distribution`` checkpoint (the default pinball checkpoint is
            required); raised by ``synthefy-nori``.
        BadRequestError
            In remote mode, if the server rejects the request (HTTP 400),
            carrying the server's ``error`` string as the message.
        AuthenticationError
            In remote mode, if the API key is missing or invalid (HTTP 401).
        ValueError
            In remote mode, if ``memory_policy=`` was sent and the deployment did not report back
            on it -- meaning it was silently ignored, so the policy had no effect.
        ImportError
            In local mode, if ``memory_policy=`` was given but the installed ``synthefy-nori``
            predates it.
        APITimeoutError
            In remote mode, if the request times out.
        APIConnectionError
            In remote mode, if a network/connection error occurs.
        """
        # These reports belong to one prediction attempt. Clear them before
        # validation so even a rejected call cannot expose an earlier call's
        # provenance as its own.
        self.last_memory_report = None
        self.last_large_context_report = None

        # Validate the output contract first: a bad output_type/quantiles pair is
        # caught before the expensive steps below (loading a sentence encoder for
        # text_columns, a checkpoint, or a paid network round-trip).
        if self.mode == "sagemaker" and extra_headers is not None:
            raise ValueError(
                "extra_headers is only valid for HTTP remote mode; SageMaker requests are SigV4-signed by boto3"
            )
        if self.mode == "sagemaker" and timeout is not None:
            warnings.warn(
                "Per-prediction timeout is ignored for SageMaker; boto3 applies "
                "the timeout configured on SynthefyNoriClient.",
                UserWarning,
                stacklevel=2,
            )
        quantile_levels = _validate_output_type(
            output_type,
            quantiles,
            discretizing=discretize is not None or categorical_levels is not None,
        )
        _validate_large_context_controls(
            policy=large_context_policy,
            threshold=large_context_threshold,
            seed=large_context_seed,
            output_type=output_type,
            model=self.model,
            mode=self.mode,
        )
        request_categorical_columns = categorical_columns
        if _has_declared_text_columns(text_columns):
            # Embed free-text columns client-side into numeric SVD features, then
            # send the widened numeric matrix through the normal request path
            # (works identically for local / remote / AWS backends).
            X_train, X_test = _widen_text_columns(
                X_train,
                X_test,
                text_columns,
                svd_dim,
                embedder,
                max_categorical_cardinality,
                categorical_encoding,
                text_device,
                categorical_columns,
            )
            # The widened frames are already numeric; do not resolve the original
            # named declarations a second time in _build_nori_request.
            request_categorical_columns = None
        # Optional controls stay absent (None) from an ordinary request, preserving its
        # historical wire bytes; _build_nori_request passes all three together into the
        # constructor so there is no assignment-order dependency on the request model's
        # "threshold/seed require a policy" validator. _validate_large_context_controls
        # above already guarantees threshold/seed are None here whenever policy is None.
        #
        # When a policy IS set, an omitted threshold/seed resolves to the documented
        # default and is still sent explicitly -- the wire request always carries the
        # values that actually ran, never leaving the caller to infer them.
        if large_context_policy is not None:
            resolved_large_context_threshold = (
                DEFAULT_LARGE_CONTEXT_THRESHOLD if large_context_threshold is None else int(large_context_threshold)
            )
            resolved_large_context_seed = (
                DEFAULT_LARGE_CONTEXT_SEED if large_context_seed is None else int(large_context_seed)
            )
        else:
            resolved_large_context_threshold = None
            resolved_large_context_seed = None
        request = _build_nori_request(
            X_train,
            y_train,
            X_test,
            task,
            categorical_columns=request_categorical_columns,
            max_categorical_cardinality=max_categorical_cardinality,
            categorical_encoding=categorical_encoding,
            output_type=output_type,
            quantile_levels=quantile_levels,
            memory_policy=memory_policy,
            large_context_policy=large_context_policy,
            large_context_threshold=resolved_large_context_threshold,
            large_context_seed=resolved_large_context_seed,
        )
        # Distribution output is shaped separately: it is not one value per query
        # row, so it does not flow through the point-prediction path below.
        if output_type in _DISTRIBUTION_OUTPUT_TYPES:
            if self.mode == "local":
                q_by_row, taus, mean = self._predict_local_distribution(
                    request, output_type=output_type, quantile_levels=quantile_levels
                )
            elif self.mode == "sagemaker":
                q_by_row, taus, mean = self._predict_aws_distribution(
                    request,
                    output_type=output_type,
                )
            else:
                q_by_row, taus, mean = self._predict_remote_distribution(
                    request,
                    output_type=output_type,
                    timeout=timeout,
                    extra_headers=extra_headers,
                )
            if output_type == "quantiles":
                return _shape_quantiles(
                    q_by_row,
                    quantile_levels or [],
                    as_pandas=as_pandas,
                    X_test=X_test,
                    y_train=y_train,
                )
            return _shape_full(
                q_by_row,
                taus,
                mean,
                as_pandas=as_pandas,
                X_test=X_test,
                y_train=y_train,
            )

        if self.mode == "local":
            predictions = self._predict_local(
                request,
                output_type=output_type,
                discretize=discretize,
                categorical_levels=categorical_levels,
            )
        else:
            remote_levels = None
            if discretize is not None or categorical_levels is not None:
                remote_levels = _resolve_remote_levels(request.y_train, discretize, categorical_levels)
            if self.mode == "sagemaker":
                predictions = self._predict_aws(
                    request,
                    output_type=output_type,
                )
            else:
                predictions = self._predict_remote(
                    request,
                    output_type=output_type,
                    timeout=timeout,
                    extra_headers=extra_headers,
                )
            if remote_levels is not None:
                predictions = _snap_to_levels(predictions, remote_levels)
        if as_pandas:
            return pd.Series(
                predictions,
                index=_result_index(X_test),
                name=_target_name(y_train),
                dtype=float,
            )
        return predictions

    # ------------------------------------------------------------------ #
    # Local mode
    # ------------------------------------------------------------------ #

    def _local_regressor_predict(
        self,
        request: NoriPredictRequest,
        *,
        output_type: str,
        quantile_levels: Optional[List[float]],
        discretize: Optional[str] = None,
        categorical_levels: Optional[VectorLike] = None,
    ) -> Any:
        """Run local inference through :class:`NoriRegressor` when its stateful API is needed.

        The functional ``synthefy_nori.predict`` cannot express ``output_type``
        (it forwards ``**kwargs`` to the constructor) or expose a per-call
        large-context report. Those calls fit/predict the estimator directly. Returns
        whatever ``NoriRegressor.predict`` returns for that ``output_type``: a
        point array, a ``(n_levels, n_query)`` quantile array, or the ``"full"``
        dict.
        """
        regressor_cls = _load_local_regressor()
        import inspect

        if "output_type" not in inspect.signature(regressor_cls.predict).parameters:
            raise ImportError(
                f"output_type={output_type!r} requires a newer synthefy-nori (with "
                "output_type= on NoriRegressor.predict, added in 0.6.0). Upgrade "
                "with: pip install -U synthefy-nori."
            )
        init_kwargs: Dict[str, Any] = {}
        if self._local_variant is not None:
            # Same upgrade guard as the functional path: the variant selector is a
            # constructor argument here, so check it before an opaque TypeError.
            if "model" not in inspect.signature(regressor_cls).parameters:
                raise ImportError(
                    f"Local Nori variant {self._local_variant!r} requires a newer "
                    "synthefy-nori (with the model= selector). Upgrade with: "
                    "pip install -U synthefy-nori."
                )
            init_kwargs["model"] = self._local_variant
        if request.memory_policy is not None:
            if not _local_memory_policy_available():
                raise ImportError(
                    "memory_policy= requires synthefy-nori >= 0.13.0 (the serving-memory policy). "
                    "Upgrade with: pip install -U synthefy-nori."
                )
            # NoriRegressor belongs to another package and expects its own MemoryPolicy
            # class (or a plain input), not this client's equivalent pydantic class.
            init_kwargs["memory_policy"] = (
                request.memory_policy
                if isinstance(request.memory_policy, str)
                else request.memory_policy.model_dump(exclude_unset=True)
            )
        if request.large_context_policy is not None:
            if not _local_large_context_available():
                raise ImportError(
                    "large_context_policy requires a newer synthefy-nori "
                    "with the large-context estimator controls. Upgrade with: "
                    "pip install -U synthefy-nori."
                )
            init_kwargs.update(
                {
                    "large_context_policy": request.large_context_policy,
                    "large_context_threshold": (
                        request.large_context_threshold
                        if request.large_context_threshold is not None
                        else DEFAULT_LARGE_CONTEXT_THRESHOLD
                    ),
                    "large_context_seed": (
                        request.large_context_seed
                        if request.large_context_seed is not None
                        else DEFAULT_LARGE_CONTEXT_SEED
                    ),
                    # The client API is one-shot: every call fits again. A
                    # caller that needs fit-once/predict-many cache control
                    # should use NoriRegressor directly.
                    "large_context_cache_entries": 1,
                }
            )
        # Everything below mutates or reads the cached, shared regressor's
        # memory_policy/large_context_* attributes and its fit/predict/report
        # sequence -- one lock per client instance, so a concurrent call on the
        # same SynthefyNoriClient cannot interleave with this one's fit/predict.
        with self._local_regressor_lock:
            if self._local_regressor is None:
                self._local_regressor = regressor_cls(**init_kwargs)
            regressor = self._local_regressor
            if hasattr(regressor, "memory_policy"):
                regressor.memory_policy = init_kwargs.get("memory_policy")
            if hasattr(regressor, "large_context_policy"):
                # Re-declare the full mutable setting set on every estimator call.
                # This clears a previous policy before a later median/default call.
                # hasattr only proves the attribute is readable, not settable (a future
                # read-only property would pass it and then raise on assignment), so
                # translate that into the same upgrade-hint ImportError as the signature
                # probe above rather than letting an AttributeError crash the call.
                try:
                    regressor.large_context_policy = request.large_context_policy
                    regressor.large_context_threshold = (
                        request.large_context_threshold
                        if request.large_context_threshold is not None
                        else DEFAULT_LARGE_CONTEXT_THRESHOLD
                    )
                    regressor.large_context_seed = (
                        request.large_context_seed
                        if request.large_context_seed is not None
                        else DEFAULT_LARGE_CONTEXT_SEED
                    )
                    regressor.large_context_cache_entries = 1
                except AttributeError as exc:
                    raise ImportError(
                        "large_context_policy requires a newer synthefy-nori "
                        "with the large-context estimator controls. Upgrade with: "
                        "pip install -U synthefy-nori."
                    ) from exc
            regressor.fit(request.X_train, request.y_train)
            predict_kwargs: Dict[str, Any] = {
                "output_type": output_type,
                "quantiles": quantile_levels,
            }
            if discretize is not None:
                predict_kwargs["discretize"] = discretize
            if categorical_levels is not None:
                predict_kwargs["categorical_levels"] = categorical_levels
            result = regressor.predict(request.X_test, **predict_kwargs)
            if request.large_context_policy is not None:
                self.last_large_context_report = _normalized_large_context_report(
                    request, getattr(regressor, "large_context_report_", None)
                )
            return result

    def _predict_local_distribution(
        self,
        request: NoriPredictRequest,
        *,
        output_type: str,
        quantile_levels: Optional[List[float]],
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Local distribution output, normalized to ``(quantiles_by_row, taus, mean)``.

        ``NoriRegressor`` returns ``output_type="quantiles"`` level-major
        (``(n_levels, n_query)``) but ``"full"`` row-major; both are normalized to
        row-major here so one shaping path serves local and remote alike.
        """
        result = self._local_regressor_predict(request, output_type=output_type, quantile_levels=quantile_levels)
        n_query = len(request.X_test)
        if output_type == "quantiles":
            levels = quantile_levels or []
            by_level = np.asarray(result, dtype=float)  # (n_levels, n_query)
            if by_level.shape != (len(levels), n_query):
                raise ValueError(
                    "synthefy-nori returned a quantile array of shape "
                    f"{by_level.shape}, expected {(len(levels), n_query)}."
                )
            # taus/mean belong to output_type="full" only; an empty array keeps the
            # tuple shape without a second forward pass.
            empty = np.asarray([], dtype=float)
            return by_level.T, empty, empty
        q_by_row = np.asarray(result["quantiles"], dtype=float)
        taus = np.asarray(result["taus"], dtype=float)
        if q_by_row.shape != (n_query, taus.shape[0]):
            raise ValueError(
                "synthefy-nori returned a quantile bank of shape "
                f"{q_by_row.shape}, expected {(n_query, taus.shape[0])}."
            )
        return q_by_row, taus, np.asarray(result["mean"], dtype=float)

    def _predict_local(
        self,
        request: NoriPredictRequest,
        *,
        output_type: str = DEFAULT_OUTPUT_TYPE,
        discretize: Optional[str] = None,
        categorical_levels: Optional[VectorLike] = None,
    ) -> List[float]:
        if output_type != DEFAULT_OUTPUT_TYPE or request.large_context_policy is not None:
            # "median" is a point output but still needs the estimator API
            # (see _local_regressor_predict). Discretization cannot reach here:
            # _validate_output_type rejects that combination up front.
            if (discretize is not None or categorical_levels is not None) and not _local_discretize_available():
                raise ImportError(
                    "Categorical-target discretization (discretize=/"
                    "categorical_levels=) requires a newer synthefy-nori. "
                    "Upgrade with: pip install -U synthefy-nori."
                )
            return _as_float_list(
                self._local_regressor_predict(
                    request,
                    output_type=output_type,
                    quantile_levels=None,
                    discretize=discretize,
                    categorical_levels=categorical_levels,
                )
            )
        local_predict = _load_local_predict()
        extra: Dict[str, Any] = {}
        if discretize is not None or categorical_levels is not None:
            if not _local_discretize_available():
                raise ImportError(
                    "Categorical-target discretization (discretize=/"
                    "categorical_levels=) requires a newer synthefy-nori. "
                    "Upgrade with: pip install -U synthefy-nori."
                )
            if discretize is not None:
                extra["discretize"] = discretize
            if categorical_levels is not None:
                extra["categorical_levels"] = categorical_levels
        if request.memory_policy is not None:
            if not _local_memory_policy_available():
                raise ImportError(
                    "memory_policy= requires synthefy-nori >= 0.13.0 (the serving-memory policy). "
                    "Upgrade with: pip install -U synthefy-nori."
                )
            # A dict, not our MemoryPolicy instance: the library's coerce() accepts its OWN
            # class, a dict, a preset name or None -- a same-named class from this package is
            # none of those and would raise. exclude_unset for the same reason as the wire
            # path: let the library apply its own defaults to whatever was not set.
            extra["memory_policy"] = (
                request.memory_policy
                if isinstance(request.memory_policy, str)
                else request.memory_policy.model_dump(exclude_unset=True)
            )
        if self._local_variant is not None:
            # Selecting a non-base local variant needs a synthefy-nori that exposes the model=
            # selector; fail with a clear upgrade hint instead of an opaque TypeError on old builds.
            import inspect

            if "model" not in inspect.signature(local_predict).parameters:
                raise ImportError(
                    f"Local Nori variant {self._local_variant!r} requires a newer synthefy-nori "
                    "(with the model= selector). Upgrade with: pip install -U synthefy-nori."
                )
            extra["model"] = self._local_variant
        result = local_predict(
            request.X_train,
            request.y_train,
            request.X_test,
            task=request.task,
            **extra,
        )
        return _as_float_list(result)

    # ------------------------------------------------------------------ #
    # Hosted transports (Baseten HTTP and AWS SageMaker)
    # ------------------------------------------------------------------ #

    def _parse_predict_response(
        self,
        request: NoriPredictRequest,
        *,
        output_type: str,
        response_data: Dict[str, Any],
    ) -> NoriPredictResponse:
        """Validate a response shared by both hosted transports."""
        parsed = NoriPredictResponse(**response_data)
        if request.memory_policy is not None:
            # The capability handshake. A deployment that predates `memory_policy` ignores the field
            # and answers with default-memory predictions that are numerically valid, so
            # nothing in `predictions` reveals that the policy was dropped. The server echoes
            # `memory_report` precisely so this is detectable -- refuse to let a caller believe
            # a policy took effect when it did not.
            if parsed.memory_report is None:
                raise ValueError(
                    "memory_policy= was sent but the deployment did not report back on it, which "
                    "means it was ignored: the predictions are valid but the policy had no "
                    "effect. The endpoint is most likely running a build from before "
                    "memory_policy= was supported. Omit memory_policy= to use the deployment's defaults."
                )
            # Validated through MemoryReport, exposed as a dict: the library's own
            # memory_report_ is a dict, and `report["rung"]` is how it is read.
            self.last_memory_report = parsed.memory_report.model_dump()
        if request.large_context_policy is not None:
            report = parsed.large_context_report
            if report is None:
                raise ValueError(
                    "large_context_policy was sent but the deployment omitted "
                    "large_context_report. The endpoint predates this capability "
                    "or ignored it, so the returned ordinary prediction is refused "
                    "instead of being mislabeled as policy output. Upgrade/deploy "
                    "a server with large-context support, or omit the policy."
                )
            expected_threshold = (
                request.large_context_threshold
                if request.large_context_threshold is not None
                else DEFAULT_LARGE_CONTEXT_THRESHOLD
            )
            expected_seed = (
                request.large_context_seed if request.large_context_seed is not None else DEFAULT_LARGE_CONTEXT_SEED
            )
            mismatches = []
            expected_policy = request.large_context_policy.strip()
            if report.policy != expected_policy:
                mismatches.append(f"policy={report.policy!r}, expected {expected_policy!r}")

            if report.threshold != expected_threshold:
                mismatches.append(f"threshold={report.threshold}, expected {expected_threshold}")
            if report.seed != expected_seed:
                mismatches.append(f"seed={report.seed}, expected {expected_seed}")
            if mismatches:
                raise ValueError(
                    "The deployment returned a mismatched "
                    "large_context_report (" + "; ".join(mismatches) + "). "
                    "The client cannot prove the requested policy was honored."
                )
            self.last_large_context_report = report.model_dump()
        if output_type != DEFAULT_OUTPUT_TYPE and parsed.output_type != output_type:
            honored = (
                "omitted the output_type field entirely, so it predates distribution output"
                if parsed.output_type is None
                else f"honored output_type={parsed.output_type!r} instead"
            )
            raise ValueError(
                f"The hosted deployment did not serve output_type={output_type!r}: "
                f"it {honored}. Such a deployment answers with the distribution "
                f"mean, which is indistinguishable from a real {output_type!r} "
                "result, so this is raised rather than returning means as if they "
                "were what you asked for. Use local mode (pip install "
                'synthefy-nori, then mode="local"), or point at a deployment '
                "that serves distribution output."
            )
        return parsed

    @staticmethod
    def _raise_sagemaker_model_error(exc: Exception) -> None:
        """Translate a SageMaker ModelError and re-raise every other SDK error."""
        aws_response = getattr(exc, "response", None)
        error = aws_response.get("Error", {}) if isinstance(aws_response, dict) else {}
        if error.get("Code") != "ModelError":
            raise exc
        original_status = aws_response.get("OriginalStatusCode", 500)
        try:
            status = int(original_status)
        except (TypeError, ValueError):
            status = 500
        if status < 400 or status > 599:
            status = 500
        message = (
            aws_response.get("OriginalMessage") or error.get("Message") or "SageMaker endpoint returned a model error"
        )
        details = {
            "error": {
                "message": message,
                "code": "ModelError",
                "log_stream_arn": aws_response.get("LogStreamArn"),
            }
        }
        headers: Dict[str, str] = {}
        metadata = aws_response.get("ResponseMetadata", {})
        if metadata.get("RequestId"):
            headers["x-request-id"] = metadata["RequestId"]
        _raise_for_status(httpx.Response(status, json=details, headers=headers))

    @staticmethod
    def _read_sagemaker_stream(event_stream: Any) -> bytes:
        """Collect SageMaker event-stream payload parts and close the stream.

        This deliberately stays synchronous: the public ``predict`` API and
        boto3's event-stream iterator are both synchronous. An async wrapper
        here would still block while boto3 reads each event.
        """
        chunks: List[bytes] = []
        try:
            for event in event_stream:
                if "PayloadPart" in event:
                    chunks.append(event["PayloadPart"].get("Bytes", b""))
                    continue
                if "ModelStreamError" in event:
                    details = event["ModelStreamError"]
                    raise RuntimeError(
                        "SageMaker model stream failed: "
                        f"{details.get('ErrorCode', 'unknown')}: "
                        f"{details.get('Message', 'no message')}"
                    )
                if "InternalStreamFailure" in event:
                    raise RuntimeError(
                        "SageMaker response stream failed internally: "
                        f"{event['InternalStreamFailure'].get('Message', 'no message')}"
                    )
        finally:
            close = getattr(event_stream, "close", None)
            if close is not None:
                close()
        return b"".join(chunks)

    def _invoke_sagemaker_predict(
        self,
        request: NoriPredictRequest,
        *,
        output_type: str,
    ) -> NoriPredictResponse:
        """Invoke the configured SageMaker endpoint with a SigV4-signed request."""
        if self._aws_client is None or self.endpoint_name is None:
            raise RuntimeError(
                "SageMaker transport is not initialized; construct the client with mode='sagemaker' and endpoint_name"
            )
        if self._sagemaker_model is None:
            raise RuntimeError("SageMaker transport has no resolved model identity")
        payload = request.to_wire()
        payload["model"] = self._sagemaker_model
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(body) > SAGEMAKER_MAX_BODY_BYTES:
            raise ValueError(
                "The SageMaker request body is "
                f"{len(body):,} bytes, exceeding InvokeEndpointWithResponseStream's "
                f"{SAGEMAKER_MAX_BODY_BYTES:,}-byte limit. Reduce the context/query rows."
            )
        try:
            # All AWS variants stream. Even base 30M can exceed SageMaker's 60-second regular
            # response ceiling for large tables; heartbeat payload parts keep the connection
            # active while the public predict() API continues to return one final typed value.
            response = self._aws_client.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                CustomAttributes="synthefy-response-stream=v1",
                Body=body,
            )
        except Exception as exc:
            # Credential, region, signing, and throttling exceptions stay as botocore errors.
            self._raise_sagemaker_model_error(exc)
            raise RuntimeError("unreachable")  # pragma: no cover

        response_body = response["Body"]
        raw = self._read_sagemaker_stream(response_body)
        response_data = json.loads(raw)
        if not isinstance(response_data, dict):
            raise ValueError("SageMaker endpoint returned JSON that was not an object")
        stream_error = response_data.get("error")
        if isinstance(stream_error, dict) and "status_code" in stream_error:
            status = int(stream_error.get("status_code", 500))
            if status < 400 or status > 599:
                status = 500
            _raise_for_status(
                httpx.Response(
                    status,
                    json={"error": {"message": stream_error.get("message", "SageMaker streaming inference failed")}},
                )
            )
        actual_model = response_data.get("model")
        if actual_model != self._sagemaker_model:
            raise ValueError(
                "SageMaker endpoint model identity mismatch: requested "
                f"{self._sagemaker_model!r}, received {actual_model!r}"
            )
        return self._parse_predict_response(
            request,
            output_type=output_type,
            response_data=response_data,
        )

    def _predict_aws(
        self,
        request: NoriPredictRequest,
        *,
        output_type: str = DEFAULT_OUTPUT_TYPE,
    ) -> List[float]:
        parsed = self._invoke_sagemaker_predict(request, output_type=output_type)
        return _as_float_list(parsed.predictions)

    def _post_predict(
        self,
        request: NoriPredictRequest,
        *,
        output_type: str,
        timeout: Optional[float],
        extra_headers: Optional[Dict[str, str]],
    ) -> NoriPredictResponse:
        """POST one prediction request and parse the response.

        Enforces the capability handshakes for both explicitly requested
        distribution output and a serving-memory policy.
        """
        # Serialise the policy with exclude_unset so only the fields the caller actually set
        # go on the wire. This is what lets the field be a typed MemoryPolicy without the
        # CLIENT pinning the SERVER's defaults: a full dump would send all twelve fields, and a
        # later change to a default server-side would then be silently overridden by every
        # older client. A request that sets no policy omits the key entirely rather than
        # sending null -- the hosted schema declares a preset name or an object, never null.
        # The distribution fields are likewise omitted when unset so the default request stays
        # byte-for-byte compatible with deployments that predate them.
        payload = request.to_wire()
        if self.model is not None:
            payload["model"] = self.model

        response = self._post_with_retries(
            self.endpoint,
            payload=payload,
            headers=self._headers(extra_headers=extra_headers),
            timeout=timeout,
        )
        return self._parse_predict_response(
            request,
            output_type=output_type,
            response_data=response.json(),
        )

    def _predict_remote(
        self,
        request: NoriPredictRequest,
        *,
        output_type: str = DEFAULT_OUTPUT_TYPE,
        timeout: Optional[float],
        extra_headers: Optional[Dict[str, str]],
    ) -> List[float]:
        parsed = self._post_predict(
            request,
            output_type=output_type,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        return _as_float_list(parsed.predictions)

    def _predict_remote_distribution(
        self,
        request: NoriPredictRequest,
        *,
        output_type: str,
        timeout: Optional[float],
        extra_headers: Optional[Dict[str, str]],
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Remote distribution output, normalized to ``(quantiles_by_row, taus, mean)``.

        The wire carries the quantile block row-major — ``(n_query, K)``, the
        natural JSON shape and the one ``output_type="full"`` uses — so no
        transpose is needed here; ``_shape_quantiles`` handles the level-major
        flip for ``output_type="quantiles"``.
        """
        parsed = self._post_predict(
            request,
            output_type=output_type,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        return self._parse_hosted_distribution(request, parsed=parsed, output_type=output_type)

    def _predict_aws_distribution(
        self,
        request: NoriPredictRequest,
        *,
        output_type: str,
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        parsed = self._invoke_sagemaker_predict(request, output_type=output_type)
        return self._parse_hosted_distribution(request, parsed=parsed, output_type=output_type)

    def _parse_hosted_distribution(
        self,
        request: NoriPredictRequest,
        *,
        parsed: NoriPredictResponse,
        output_type: str,
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Normalize either hosted transport's quantile response."""
        if parsed.quantiles is None or parsed.taus is None:
            raise ValueError(
                f"The hosted deployment echoed output_type={output_type!r} but "
                "returned no quantile block; it cannot serve distribution output. "
                'Use local mode (pip install synthefy-nori, then mode="local").'
            )
        q_by_row = _nullable_rows_to_array(parsed.quantiles)
        taus = np.asarray(parsed.taus, dtype=float)
        n_query = len(request.X_test)
        expected = (n_query, taus.shape[0])
        if q_by_row.shape != expected:
            raise ValueError(
                "The server returned a quantile block of shape "
                f"{q_by_row.shape}, expected {expected} "
                "(one row per X_test row, one column per tau level)."
            )
        if request.quantiles is not None:
            requested = np.asarray(request.quantiles, dtype=float)
            if taus.shape[0] != requested.shape[0]:
                raise ValueError(
                    f"Requested {requested.shape[0]} quantile level(s) but the server returned {taus.shape[0]}."
                )
            # The returned levels must be the requested ones, in order: the
            # columns are labeled from the request, so levels that drifted would
            # mean data at one tau labeled with another.
            if not np.allclose(taus, requested, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"Requested quantile levels {requested.tolist()} but the server returned {taus.tolist()}."
                )
        return q_by_row, taus, np.asarray(parsed.predictions, dtype=float)

    def _headers(self, *, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
            "Authorization": f"{self.auth_scheme} {self.api_key}",
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _should_retry(self, response: Optional[httpx.Response], exc: Optional[Exception]) -> bool:
        if exc is not None:
            # Connection errors/timeouts are retryable
            return True
        if response is None:
            return False
        if response.status_code in (408, 409, 425, 429) or 500 <= response.status_code <= 599:
            return True
        return False

    def _compute_backoff(self, attempt: int, response: Optional[httpx.Response]) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    parsed_retry_after = float(retry_after)
                    if np.isfinite(parsed_retry_after):
                        return max(0.0, parsed_retry_after)
                except (TypeError, ValueError):
                    pass
        # Exponential backoff with jitter
        base = min(2**attempt, 30)
        return base * (0.5 + 0.5 * (os.urandom(1)[0] / 255))

    def _post_with_retries(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        *,
        headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        assert self.client is not None  # remote mode always has a client
        # Serialize once, outside the retry loop: the payload is identical on every
        # attempt, and a non-finite value here is invalid client input, not a
        # transport failure a retry could ever fix. Explicit so upgrading httpx
        # cannot change the public wire bytes -- the imported 6.3 client used
        # stdlib JSON spacing, which the frozen compatibility trace pins.
        try:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
        except ValueError as exc:
            raise ValueError(
                "Request payload contains a non-finite value (NaN/Infinity), which "
                "is not valid JSON. y_train must contain only finite numbers for "
                f"remote inference. Underlying error: {exc}"
            ) from exc
        last_exc: Optional[Exception] = None
        response: Optional[httpx.Response] = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            # Reset per attempt so the "no more retries" block below reflects the
            # final attempt only. Otherwise an exception from an earlier attempt
            # (e.g. a transient connection error) would be re-raised in place of
            # the true final error (e.g. a 5xx that should map to a server error).
            last_exc = None
            response = None
            try:
                response = self.client.post(
                    endpoint,
                    content=body,
                    headers=headers or self._headers(),
                    timeout=self.timeout if timeout is None else timeout,
                )
                if not self._should_retry(response, None):
                    _raise_for_status(response)
                    return response
            except httpx.TimeoutException as exc:
                last_exc = APITimeoutError(str(exc))
            except httpx.HTTPError as exc:
                last_exc = APIConnectionError(str(exc))

            # Decide to retry
            if attempt < attempts - 1 and self._should_retry(response, last_exc):
                delay = self._compute_backoff(attempt, response)
                time.sleep(delay)
                continue

            # No more retries
            if last_exc is not None:
                raise last_exc
            if response is not None:
                _raise_for_status(response)
                return response

        # Should not reach here
        raise APIConnectionError("Request failed after retries")
