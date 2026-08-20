"""Small public inference API."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from synthefy.featurize import CATEGORICAL_AUTO, DataFramePreprocessor

# Re-exported: `synthefy_nori.config_path` and `from synthefy_nori.api import config_path`
# are both long-standing entry points. The implementation lives in one place.
from synthefy_nori.configs import DEFAULT_INFERENCE_CONFIG, config_path
from synthefy_nori.inference.large_context import (
    DEFAULT_LARGE_CONTEXT_THRESHOLD,
    LargeContextUnsupportedOutputError,
    build_problem,
    large_context_applies,
    predictor_call_fn,
    resolve_large_context_policy,
    run_policy,
)
from synthefy_nori.inference.memory_policy import MemoryPolicy
from synthefy_nori.discretize import (
    DEFAULT_DISCRETIZE_METHOD,
    DISCRETIZE_METHODS,
    SNAP_METHODS,
    discretize_predictions,
    target_levels,
)
from synthefy_nori.featurize import (
    DEFAULT_CATEGORICAL_ENCODING,
    DEFAULT_MAX_CARDINALITY,
)


Task = Literal["regression", "reg"]


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    try:
        return bool(mps is not None and mps.is_available())
    except (AttributeError, RuntimeError):
        return False


def _default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _as_device(device):
    if device is None:
        return _default_device()
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={str(resolved)!r} was requested, but CUDA is not "
                "available to PyTorch."
            )
        if resolved.index is not None:
            device_count = torch.cuda.device_count()
            if resolved.index >= device_count:
                raise RuntimeError(
                    f"device={str(resolved)!r} was requested, but only "
                    f"{device_count} CUDA device(s) are visible."
                )
    if resolved.type == "mps" and not _mps_available():
        mps = getattr(torch.backends, "mps", None)
        built = bool(mps is not None and getattr(mps, "is_built", lambda: False)())
        reason = (
            "this PyTorch build does not include MPS support"
            if not built
            else "MPS is not usable on this macOS version or hardware"
        )
        raise RuntimeError(f"device='mps' was requested, but {reason}.")
    return resolved


def _resolve_model_path(model_path: str | None, token: str | bool | None = None,
                        model: str | None = None) -> str:
    if model_path is not None:
        return model_path
    from synthefy_nori.hf import download_checkpoint

    return download_checkpoint(model=model, token=token)


def _coerce_numeric_feature_matrix(X, *, name: str) -> np.ndarray:
    """Coerce a positional feature matrix with actionable non-numeric errors."""
    try:
        matrix = np.asarray(X, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        try:
            raw = np.asarray(X, dtype=object)
        except ValueError:
            raw = None
        offending: list[int] = []
        if raw is not None and raw.ndim == 2:
            for index in range(raw.shape[1]):
                try:
                    np.asarray(raw[:, index], dtype=np.float32)
                except (TypeError, ValueError):
                    offending.append(index)
        detail = f"; non-numeric column indices={offending}" if offending else ""
        raise ValueError(
            f"{name} must be a numeric 2D array/list{detail}. For named categorical or text "
            "features, pass a pandas DataFrame and use categorical_columns=[...] or text_columns=[...]."
        ) from exc
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape (n_rows, n_features); got shape {matrix.shape}.")
    return matrix


def _has_explicit_columns(value) -> bool:
    if value is None or (isinstance(value, str) and value == CATEGORICAL_AUTO):
        return False
    try:
        return bool(len(value))
    except TypeError:
        return True


class NoriRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn regression estimator wrapping the Synthefy checkpoint.

    Subclasses ``BaseEstimator``/``RegressorMixin`` so it works directly with the
    scikit-learn ecosystem (``clone``, ``get_params``/``set_params``, ``score``,
    partial dependence, sequential feature selection) and with shapiq — see
    ``synthefy_nori.interpretability``. The ``__init__`` arguments are stored
    verbatim (the only normalizations applied are idempotent), so ``clone`` round
    trips correctly.

    Memory on large tables (``memory_policy=``)
        Nori does in-context regression, so your table is *input*: one ``predict``
        keeps a per-layer key/value cache over every context row, and that cache --
        not the model -- is what exhausts GPU memory on a big table. ``memory_policy=``
        decides what to do about it. Omit it for defaults that handle the common
        cases, or pass:

        * a preset name -- ``"exact"`` (never quantize; offload instead),
          ``"max_context"`` (fit the largest table you can), ``"off"`` (no cache)
        * a dict of individual fields, e.g. ``{"gpu_budget_frac": 0.25}``, which is
          the shape a YAML/JSON config lands in
        * a :class:`~synthefy_nori.inference.memory_policy.MemoryPolicy`

        Budgets are fractions of your hardware, so one setting travels from a laptop
        GPU to an H200. Settings that cannot take effect raise rather than being
        ignored, and redundant ones warn. Validated in :meth:`fit`; afterwards
        :attr:`memory_report_` says which fallback rung ran, at what precision, and
        whether any context rows had to be dropped. Full field table and the rung
        ladder: README, "Serving memory on large tables".
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        model: str | None = None,
        device=None,
        inference_config: str | None = None,
        token: str | bool | None = None,
        augmentations: tuple[str, ...] | list[str] | None = ("yj",),
        yj_skew_threshold: float = 10.0,
        quantile_collapse: str = "mean",
        bar_temperature: float = 1.0,
        bar_point_estimator: str = "mean",
        discrete_y_snap_max_unique: int = 0,
        discretize: str | None = None,
        categorical_levels=None,
        categorical_columns=CATEGORICAL_AUTO,
        categorical_encoding: str = DEFAULT_CATEGORICAL_ENCODING,
        max_categorical_cardinality: int = DEFAULT_MAX_CARDINALITY,
        text_columns=None,
        memory_policy: "MemoryPolicy | dict | str | None" = None,
        large_context_policy=None,
        large_context_threshold: int = DEFAULT_LARGE_CONTEXT_THRESHOLD,
        large_context_seed: int = 0,
        large_context_cache_entries: int = 1,
        svd_dim: int | None = 128,
        embedder="minilm",
        text_max_cardinality: int | None = None,
        text_normalize: bool | None = None,
    ) -> None:
        """Configure the estimator (arguments are stored verbatim; see class docs).

        Args:
            model_path: path to a local ``.pt`` checkpoint. When ``None``, ``model``
                is required and its checkpoint is downloaded/cached from Hugging Face.
            model: variant selector -- REQUIRED when ``model_path`` is None. Choose
                ``"nori-6m"`` (the base), ``"nori-30m"`` or ``"nori-100m"`` (the
                largest); there is no default and omitting both raises. Ignored when
                ``model_path`` is given.
            device: torch device for inference (``"cuda:0"``, ``"cpu"``, ...).
                ``None`` automatically picks CUDA, then Apple MPS when available,
                and otherwise CPU. The fitted ``device_`` attribute records the
                device actually selected.
            inference_config: path to an inference-config JSON. ``None`` uses
                the bundled default config.
            token: Hugging Face token (higher rate limits / private repos);
                never required for the public checkpoint.
            augmentations: preprocessing ensemble members, e.g. ``("yj",)`` for
                the Yeo-Johnson pipeline (default). Empty/None disables.
            yj_skew_threshold: absolute target-skew above which the YJ member
                is used (only relevant when ``"yj"`` is in ``augmentations``).
            quantile_collapse: how the pinball head's quantile bank collapses
                to a point estimate for ``output_type="mean"`` (``"mean"``,
                ``"median"``, ``"trimmed_mean"``, ...).
            bar_temperature: softmax temperature for bar-distribution heads.
            bar_point_estimator: point decode for bar-distribution heads
                (``"mean"``/``"median"``/``"mode"``).
            discrete_y_snap_max_unique: legacy auto-snap threshold. ``0``
                (default) = off. If ``> 0`` and the training target has at
                most that many distinct values, point predictions are snapped
                to the nearest training-y value. Prefer ``discretize=``.
            discretize: declare a categorical/ordinal target and pick the
                discretization strategy — ``"map-cell"``, ``"median-cell"``,
                ``"snap-mean"``, ``"snap-median"``, ``"expected-level"``, or
                ``"prior-match"`` (see ``synthefy_nori.discretize``).
                ``None`` (default) = ordinary continuous regression.
            categorical_levels: the complete set of values the target can
                take (numeric, order matters — NOT arbitrary class labels),
                e.g. ``[1, 2, 3, 4, 5]`` for a 1–5 rating. Setting it alone
                declares a categorical target with the default strategy
                (``DEFAULT_DISCRETIZE_METHOD``); ``None`` (default) uses the
                distinct values of the fitted ``y`` when a strategy is set.
            categorical_columns: DataFrame feature-column policy. ``"auto"``
                (default) encodes remaining non-numeric columns; a sequence
                encodes exactly those columns and rejects undeclared strings;
                ``None`` disables categorical inference.
            categorical_encoding: ``"ordinal"`` (default) or ``"onehot"``.
            max_categorical_cardinality: maximum retained levels per explicitly
                declared categorical. Automatically inferred columns above this
                limit raise an ambiguity error instead of being dropped.
            text_columns: enables the zero-shot text path when set (requires ``X``
                to be a :class:`pandas.DataFrame`). A list of column names embeds
                those columns. ``None`` and ``[]`` both mean no text columns;
                neither imports or loads the text dependency.
            svd_dim: width of the TruncatedSVD text block appended to the numeric
                features (fit on train only). Default 128; ``None`` appends the full
                raw embedding without reduction. Ignored when ``text_columns`` names
                no text columns.
            embedder: the sentence encoder for ``text_columns`` — a short name / HF
                id string (e.g. ``"minilm"``; needs the optional
                ``sentence-transformers`` extra), a preloaded encoder object, or a
                callable ``texts -> ndarray``.
            text_max_cardinality: deprecated alias for
                ``max_categorical_cardinality``. Prefer the latter.
            text_normalize: cosine-normalize the text embeddings. ``None`` (default)
                auto-enables it for known LLM encoders and disables it otherwise;
                set ``True``/``False`` to override (needed for a preloaded encoder
                object, whose model id can't be inspected).
            large_context_policy: how to choose the context when the table exceeds
                ``large_context_threshold`` rows. ``None`` (default) keeps the existing
                behavior — full context, trimmed by ``memory_policy`` if it does not
                fit. Otherwise a policy name (``"cluster_route"``,
                ``"cluster_route_g4"``, ``"safeboost"``, ``"boost"``, ``"random"``),
                ``True`` for the default (``"cluster_route"``), a
                ``"pkg.mod:fn"``/``"file.py:fn"`` path, a callable, or a LIST of any of
                those (scored on a train holdout, per-table winner deployed). Append
                ``"[k=v]"`` to pass parameters, e.g. ``"cluster_route[groups=16]"``.
                See :mod:`synthefy_nori.inference.large_context` — and note that ``"boost"``
                measured −0.229 worst-case, so prefer ``"safeboost"`` or a list.
            large_context_threshold: row count above which ``large_context_policy`` engages.
                Default 50,000. Inert when ``large_context_policy`` is ``None``.
            large_context_seed: seeds the policies' row draws, so two identical predicts
                agree. Default 0.
            large_context_cache_entries: how many distinct encoded contexts to retain across
                ``predict`` calls. Default 1 (the historical behavior). A policy that
                rotates between pools — ``cluster_route`` builds ``groups`` of them —
                gets no cache hits at 1, since each pool evicts the previous one; raise
                it to the pool count to keep the rotation. **Costs one full K/V cache
                per entry**, which is why it is not raised automatically.

        The discretize/categorical_levels pair are estimator-level defaults; the
        same-named ``predict`` kwargs override them per call. The text_* params take
        effect at ``fit``.
        """
        self.model_path = model_path
        # Variant selector (required when model_path is None): "nori-6m" / "nori-30m" /
        # "nori-100m", resolved to a Hugging Face repo via synthefy_nori.hf. Ignored when
        # model_path is given.
        self.model = model
        self.device = device
        self.token = token
        self.inference_config = inference_config or config_path(DEFAULT_INFERENCE_CONFIG)
        self.augmentations = tuple(augmentations) if augmentations else ()
        self.yj_skew_threshold = float(yj_skew_threshold)
        self.quantile_collapse = quantile_collapse
        self.bar_temperature = float(bar_temperature)
        self.bar_point_estimator = bar_point_estimator
        self.discrete_y_snap_max_unique = int(discrete_y_snap_max_unique)
        # estimator-level defaults for the categorical-target path, so the
        # feature is reachable through the sklearn ecosystem (clone/get_params/
        # GridSearchCV/cross_val_score); predict() kwargs override per call.
        self.discretize = discretize
        self.categorical_levels = categorical_levels
        self.categorical_columns = categorical_columns
        self.categorical_encoding = categorical_encoding
        self.max_categorical_cardinality = max_categorical_cardinality
        # Zero-shot text config — constructor params (not fit kwargs) so the
        # feature round-trips through clone/get_params/GridSearchCV/cross_val_score
        # and pickle, exactly like discretize/categorical_levels above.
        #   text_columns: None / [] -> no text columns; list of names -> embed
        #     those columns. DataFrames always use the unified numeric and
        #     categorical preprocessing contract.
        #   svd_dim: SVD width of the appended text block (None = raw embedding).
        #   embedder: short name / preloaded encoder / callable.
        self.text_columns = text_columns
        self.svd_dim = svd_dim
        self.embedder = embedder
        self.text_max_cardinality = text_max_cardinality
        self.text_normalize = text_normalize
        # Serving-memory policy: a preset name ("exact", "max_context", "off"), a
        # dict, or a MemoryPolicy. Stored VERBATIM and coerced lazily in
        # _get_predictor(); transforming it here would break sklearn clone(), whose
        # identity check requires the stored attribute to be the object handed in.
        self.memory_policy = memory_policy
        # Large-context policy. Stored verbatim for the same clone() reason as
        # memory_policy above — large_context_policy may be a callable or a list, and
        # resolving it here would replace the object sklearn compares identity on.
        self.large_context_policy = large_context_policy
        self.large_context_threshold = int(large_context_threshold)
        self.large_context_seed = int(large_context_seed)
        self.large_context_cache_entries = int(large_context_cache_entries)
        self._predictor = None
        self._feature_preprocessor = None
        # Legacy fitted-text slot retained for old pickles; new fits use the
        # unified _feature_preprocessor above.
        # The fitted half of a large-context prediction (train-derived state that survives
        # across predict calls), rebuilt by fit(). See _large_context_predict.
        self._large_context_problem = None
        # The table-derived half of the large-context window (see _large_context_predict), cached
        # for the life of a fit because deriving it scans the whole table.
        self._large_context_budget_features = None
        self.large_context_report_ = None
        self._text_preprocessor = None

    @property
    def memory_report_(self) -> dict | None:
        """What the last ``predict`` call did about memory, or None before one.

        A ``MemoryPolicy.model_dump()``: the ladder rung taken, the cache precision
        and placement chosen, the budgets used, and how many context rows (if any)
        had to be dropped to fit. Reconstruct the object with
        ``MemoryPolicy(**estimator.memory_report_)`` for derived facts such as
        ``is_bit_exact``.

        Forwards to the underlying predictor so callers never have to reach through
        ``._predictor``, which is private.
        """
        if self._predictor is None:
            return None
        return self._predictor.memory_report_

    def fit(self, X, y):
        """Fit the in-context regressor on ``(X, y)``.

        DataFrames are resolved into numeric, categorical, and explicitly named
        text columns by a fitted schema. Positional arrays/lists must already be
        numeric.

        Set ``text_columns`` in the constructor to embed named DataFrame columns by a
        frozen sentence encoder, reduced to ``svd_dim`` columns via TruncatedSVD
        (fit on this training split only), and appended to the numeric/categorical
        block; Nori then consumes the widened matrix like any other features. No
        gradient training happens — the encoder and Nori stay frozen. ``predict``
        replays the same transform, so its ``X`` must be a DataFrame with these
        columns. The text config lives in ``__init__`` so it round-trips through
        ``clone``/``get_params``/``GridSearchCV``/``cross_val_score`` and pickle.
        """
        # Validate memory_policy= HERE rather than in __init__ (sklearn requires __init__ to
        # store params verbatim, and clone() depends on it) and rather than lazily at
        # predict time, where an incoherent config would only surface minutes into a
        # job. The coerced policy is discarded: this call exists for its errors.
        MemoryPolicy.coerce(self.memory_policy)
        # Same reason, for large_context_policy=: an unknown policy name or a bad parameter
        # should fail at fit(), not minutes into a job on a million-row table.
        if self.large_context_policy is not None:
            resolve_large_context_policy(self.large_context_policy)
        # Every large-context cache is keyed to the table being replaced, so drop it here.
        # Keeping it would let a boosting chain built on the PREVIOUS fit's rows serve
        # this one -- a wrong answer, and the reason this is invalidated in fit rather
        # than checked lazily. Long-lived estimators may be re-fitted repeatedly, so
        # keep this invalidation on the ordinary fit path.
        self._large_context_problem = None
        self._large_context_budget_features = None
        self.large_context_report_ = None
        # Resolve automatic placement once per fit so text preprocessing and model
        # inference cannot independently choose different devices. Keep the raw
        # constructor parameter unchanged for sklearn clone/get_params semantics.
        self.device_ = _as_device(self.device)
        max_cardinality = self.max_categorical_cardinality
        if self.text_max_cardinality is not None:
            if (
                self.max_categorical_cardinality != DEFAULT_MAX_CARDINALITY
                and self.max_categorical_cardinality != self.text_max_cardinality
            ):
                raise ValueError(
                    "text_max_cardinality and max_categorical_cardinality disagree; "
                    "remove the deprecated text_max_cardinality argument."
                )
            warnings.warn(
                "text_max_cardinality is deprecated; use max_categorical_cardinality instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            max_cardinality = self.text_max_cardinality

        # Validate feature-policy knobs even for positional numeric inputs so a
        # misspelled encoding or invalid cap never becomes a silently ignored
        # constructor argument.
        feature_parameters = DataFramePreprocessor(
            categorical_encoding=self.categorical_encoding,
            max_categorical_cardinality=max_cardinality,
        )
        feature_parameters._validate_parameters()

        if isinstance(X, pd.DataFrame):
            self._feature_preprocessor = DataFramePreprocessor(
                categorical_columns=self.categorical_columns,
                categorical_encoding=self.categorical_encoding,
                max_categorical_cardinality=max_cardinality,
                text_columns=self.text_columns,
                svd_dim=self.svd_dim,
                embedder=self.embedder,
                text_device=self.device_,
                text_normalize=self.text_normalize,
            )
            X_frame = self._feature_preprocessor.fit_transform(X)
            X_mat = X_frame.to_numpy(dtype=np.float32)
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            if _has_explicit_columns(self.categorical_columns):
                raise ValueError(
                    "Explicit categorical_columns requires a pandas DataFrame with named columns; "
                    "positional arrays/lists must already be numeric."
                )
            if _has_explicit_columns(self.text_columns):
                raise ValueError(
                    "text_columns requires a pandas DataFrame with named columns; "
                    "positional arrays/lists must already be numeric."
                )
            self._feature_preprocessor = None
            X_mat = _coerce_numeric_feature_matrix(X, name="X")
            self.n_features_in_ = X_mat.shape[1]
            if hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        self._text_preprocessor = None
        self.X_train_ = X_mat.astype(np.float32)
        self.y_train_ = np.asarray(y, dtype=np.float64)
        self.y_mean_ = float(self.y_train_.mean())
        y_std = float(self.y_train_.std())
        self.y_std_ = y_std if y_std >= 1e-12 else 1.0
        return self

    def _prepare_query_features(self, X) -> np.ndarray:
        """Coerce query rows to the model's float32 matrix.

        Applies the fitted DataFrame schema when configured; otherwise validates
        a positional numeric matrix.
        """
        if getattr(self, "_feature_preprocessor", None) is not None:
            return self._feature_preprocessor.transform(X).to_numpy(dtype=np.float32)
        # Pickles fitted before the unified DataFrame preprocessor stored the old
        # multimodal object here. Keep those artifacts usable.
        if getattr(self, "_text_preprocessor", None) is not None:
            return self._text_preprocessor.transform(X).astype(np.float32)
        return _coerce_numeric_feature_matrix(X, name="X")

    def _get_predictor(self):
        """The cached predictor, with ``memory_policy=`` re-read from this estimator.

        The predictor is built once -- it owns the loaded checkpoint -- and reused, so
        every *other* constructor argument here is effectively frozen at first use.
        That is right for those: they choose the weights and the inference config,
        neither of which can change without a reload. It is wrong for ``memory_policy``,
        which is a per-call resource decision that says nothing about the model.

        So re-declare it on every call. ``NoriPredictor.__init__`` stores ``memory_policy``
        verbatim and ``_memory_policy()`` coerces it afresh inside each ``predict``,
        which is what makes this single assignment sufficient -- nothing downstream
        caches the resolved policy. Without it the FIRST predict's policy would stick
        for the estimator's lifetime, so ``est.memory_policy = "off"; est.predict(X)`` would
        be silently ignored, and a long-lived server that re-declares the policy per
        request would serve every caller the first caller's setting.
        """
        if self._predictor is None:
            from synthefy_nori.inference.predictor import NoriPredictor

            resolved_device = (
                self.device_ if hasattr(self, "device_") else _as_device(self.device)
            )
            self._predictor = NoriPredictor(
                device=resolved_device,
                model_path=_resolve_model_path(self.model_path, self.token, self.model),
                inference_config=self.inference_config,
                augmentations=self.augmentations,
                yj_skew_threshold=self.yj_skew_threshold,
                quantile_collapse=self.quantile_collapse,
                bar_temperature=self.bar_temperature,
                bar_point_estimator=self.bar_point_estimator,
                discrete_y_snap_max_unique=self.discrete_y_snap_max_unique,
                memory_policy=self.memory_policy,
            )
        else:
            self._predictor.memory_policy = self.memory_policy
        # Re-declared per call for the same reason as memory_policy: a resource decision
        # that says nothing about the weights, so it must not freeze at first use.
        self._predictor.context_cache_entries = self.large_context_cache_entries
        return self._predictor

    def predict(
        self,
        X,
        *,
        output_type: str = "mean",
        quantiles: list[float] | None = None,
        discretize: str | None = None,
        categorical_levels=None,
    ):
        """Predict targets for the query rows.

        ``output_type`` selects what is returned from the model's predictive
        distribution:

        - ``"mean"``   — distribution mean (default; identical to prior behavior)
        - ``"median"`` — distribution median (the ``tau=0.5`` quantile)
        - ``"mode"``   — distribution mode
        - ``"quantiles"`` — quantiles at the levels given in ``quantiles=`` (a
          list of taus in (0, 1)); returns an array of shape
          ``(len(quantiles), n_samples)``
        - ``"full"`` — the full predictive distribution as a dict with keys
          ``"quantiles"`` (``(n_samples, K)`` ascending quantile values),
          ``"taus"`` (``(K,)`` quantile levels), and ``"mean"`` (``(n_samples,)``)

        ``"main"`` is a recognized output_type name but is not supported here.
        ``"quantiles"`` / ``"full"`` are only available for the pinball
        (quantile-head) checkpoint shipped by default; a ``bar_distribution``
        checkpoint raises ``NotImplementedError``.

        Categorical/ordinal targets — passing ``discretize=`` (a strategy) or
        ``categorical_levels=`` (a known lattice) declares the target discrete
        and returns labels on its level lattice:

        - ``discretize`` — the strategy that picks WHICH lattice point:
          ``"map-cell"`` (most-probable level — accuracy-optimal; the default
          when only ``categorical_levels`` is given), ``"median-cell"``
          (discrete median — MAE-optimal), ``"snap-mean"`` (nearest level to
          the mean — QWK), ``"snap-median"`` (nearest level to the
          distribution median), ``"expected-level"`` (lattice-informed
          CONTINUOUS expectation — analysis tool, not on-lattice), and
          ``"prior-match"`` (label frequencies match training priors —
          benchmarked worse; calibration experiments only). Full guidance in
          ``synthefy_nori.discretize`` and docs/inference.md.
        - ``categorical_levels`` — the complete set of values the target can
          take, when you know it: numeric, order-significant (cells are built
          between adjacent values), NOT arbitrary class labels. Example: a
          1–5 rating whose small context happens to contain no 1s →
          ``categorical_levels=[1, 2, 3, 4, 5]`` makes 1 predictable anyway.
          ``None`` (default) uses the distinct values of the fitted ``y``.

        Discretization is opt-in (for R²-scored tasks keep the continuous
        default). Both are also constructor params (so
        ``clone``/``GridSearchCV``/``cross_val_score`` see them); ``predict``
        kwargs override the estimator values per call.
        """
        if discretize is None:
            discretize = self.discretize
        if categorical_levels is None:
            categorical_levels = self.categorical_levels
        if discretize is not None or categorical_levels is not None:
            if output_type != "mean" or quantiles is not None:
                raise ValueError(
                    "categorical output (discretize=/categorical_levels=) "
                    "returns discrete labels; combine it only with the default "
                    "output_type='mean' (the discretize= strategy chooses the "
                    "summary), not output_type/quantiles."
                )
            return self._predict_categorical(
                X,
                method=DEFAULT_DISCRETIZE_METHOD if discretize is None else discretize,
                levels=categorical_levels,
            )
        if output_type in ("quantiles", "full"):
            return self._predict_distribution(X, output_type=output_type, quantiles=quantiles)
        if output_type == "main":
            raise NotImplementedError(
                "output_type='main' is not supported. Use 'mean', 'median', "
                "'mode', 'quantiles', or 'full'."
            )
        if output_type not in ("mean", "median", "mode"):
            raise ValueError(
                f"Unknown output_type={output_type!r}; expected one of "
                "'mean', 'median', 'mode', 'quantiles', 'full'."
            )
        if quantiles is not None:
            raise ValueError(
                "quantiles= is only valid with output_type='quantiles'."
            )

        # Drive the predictor's distribution-collapse from output_type. "mean"
        # uses the regressor's configured collapse so the default path is
        # byte-for-byte the prior behavior; "median"/"mode" override it for this
        # call. A quantile head has no native mode, so "mode" falls back to the
        # median there, while bar-distribution heads decode a true mode.
        if output_type == "mean":
            return self._predict_point(
                X,
                quantile_collapse=self.quantile_collapse,
                bar_point_estimator=self.bar_point_estimator,
            )
        return self._predict_point(
            X, quantile_collapse="median", bar_point_estimator=output_type)

    def _predict_point(self, X, *, quantile_collapse: str, bar_point_estimator: str):
        """One point-prediction pass with an explicit collapse, predictor state
        restored afterwards (so no call leaks its collapse into the next)."""
        if not hasattr(self, "X_train_"):
            raise ValueError("Call fit(X, y) before predict(X).")

        X_test = self._prepare_query_features(X)
        y_norm = ((self.y_train_ - self.y_mean_) / self.y_std_).astype(np.float32)

        # This report belongs to one prediction attempt. Clear an earlier policy's
        # report before dispatch so a later full-context call cannot claim the old
        # policy ran (and a failed attempt cannot leave stale provenance behind).
        self.large_context_report_ = None
        predictor = self._get_predictor()
        saved = (predictor.quantile_collapse, predictor.bar_point_estimator)
        try:
            predictor.quantile_collapse = quantile_collapse
            predictor.bar_point_estimator = bar_point_estimator
            if large_context_applies(len(self.X_train_), self.large_context_policy,
                               self.large_context_threshold):
                pred = self._large_context_predict(
                    predictor, X_test, y_norm,
                    decoder=(quantile_collapse, bar_point_estimator))
            else:
                pred = predictor.predict(self.X_train_, y_norm, X_test)
        finally:
            predictor.quantile_collapse, predictor.bar_point_estimator = saved
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float64).squeeze()
        return pred * self.y_std_ + self.y_mean_

    def _large_context_predict(self, predictor, X_test: np.ndarray, y_norm: np.ndarray,
                         *, decoder: tuple):
        """Predict through ``large_context_policy`` instead of one full-context call.

        The :class:`~synthefy_nori.inference.policies.Problem` is built once per ``fit``
        and reused, so train-derived work — the imputed train view, the train routing
        space, a boosting chain's residuals — is paid once rather than per ``predict``.
        Only the query-derived half is recomputed here.

        Two things are re-derived per call rather than frozen at ``fit``, because both
        are per-call inputs the caller is entitled to change:

        * **the window**, from ``memory_policy``. That policy is re-declared on the
          predictor on every call (a server sets it per request), so a later, smaller
          ``elements_budget`` must shrink the context a policy emits. Frozen, the policy
          would keep emitting the old size and the predictor would subsample it back
          down at random or raise ``ContextTooLargeError`` — while the report still
          advertised the old window. A changed window invalidates the Problem outright:
          a boosting chain's shards are window-sized, so it is a different chain.
        * **everything that changes the model's answer**, as the cache scope. That is
          the decoder plus ``memory_policy``: INT8 cache precision is deliberately
          lossy, so its residual labels and gate winner are not interchangeable with
          BF16's even when both policies produce the same context window. See
          :attr:`~synthefy_nori.inference.policies.Problem.train_cache`.

        Runs in NORMALIZED y (the caller denormalizes), which every policy tolerates:
        row selection is scale-free and residual boosting works on differences.
        """
        # The budget that would otherwise have trimmed the context sizes it instead, so
        # the per-call subsample never fires under a policy. Re-resolved per call, but
        # its table-derived half is not: that one scans the whole table for binary
        # columns (~5s on 1M x 130), which is precisely the kind of per-predict
        # repetition the rest of this path exists to remove.
        if self._large_context_budget_features is None:
            self._large_context_budget_features = predictor.budget_n_features(self.X_train_)
        window = predictor.max_context_rows(
            self.X_train_, budget_n_features=self._large_context_budget_features)
        if self._large_context_problem is None or self._large_context_problem.window != window:
            previous = self._large_context_problem
            self._large_context_problem = build_problem(
                predictor_call_fn(predictor),
                self.X_train_,
                y_norm,
                window=window,
                seed=self.large_context_seed,
            )
            if previous is not None:
                # The window changed, so the cached DECISIONS are stale (a chain's
                # shards are window-sized) -- but the imputed train block is not, and
                # re-deriving it per memory_policy change is the cost this path exists
                # to avoid.
                self._large_context_problem.adopt_train_state(previous)
        # A train-cache decision depends on every outside input that can change
        # `predict_fn`, not just the decoder. MemoryPolicy's JSON is stable, hashable,
        # includes defaults, and follows in-place dict changes between calls.
        memory_scope = MemoryPolicy.coerce(self.memory_policy).model_dump_json()
        pred, self.large_context_report_ = run_policy(
            self._large_context_problem, X_test,
            policy_spec=self.large_context_policy, seed=self.large_context_seed,
            cache_scope=("decoder", decoder, "memory_policy", memory_scope),
        )
        return pred

    def get_embeddings(self, X=None, *, data_source: str = "test") -> np.ndarray:
        """Return the model's learned representation of rows.

        Embeds ``X`` against the context stored by ``fit`` and returns the
        final-layer target-token representation per row.

        - ``data_source="test"`` (default): embed the query rows ``X`` (required).
        - ``data_source="train"``: embed the stored context rows. ``X`` is
          genuinely ignored here and may be omitted — the context embeddings
          depend only on the data passed to ``fit`` — so it is neither validated
          against the fitted feature count nor preprocessed.

        Returns an array of shape ``(n_estimators, n_samples, embed_dim)``,
        where ``n_estimators`` is the number of preprocessing pipelines in the
        inference config. Pick a member (``embeds[0]``) or average across
        ``axis=0`` for a 2D feature matrix.
        """
        if not hasattr(self, "X_train_"):
            raise ValueError("Call fit(X, y) before get_embeddings(X).")
        y_norm = ((self.y_train_ - self.y_mean_) / self.y_std_).astype(np.float32)
        predictor = self._get_predictor()
        if data_source == "train":
            # X is ignored for context embeddings; do not touch it so a
            # missing/mismatched X cannot raise. The predictor synthesizes a
            # dummy query from the context.
            return predictor.get_embeddings(
                self.X_train_, y_norm, None, data_source=data_source)
        if X is None:
            raise ValueError(
                "get_embeddings requires X for data_source='test'.")
        X_test = self._prepare_query_features(X)
        return predictor.get_embeddings(
            self.X_train_, y_norm, X_test, data_source=data_source)

    def _predict_distribution(self, X, *, output_type: str, quantiles: list[float] | None):
        """Return the model's predictive distribution as quantiles.

        Backs ``output_type in {"quantiles", "full"}``. Reads the raw per-row
        quantile bank from the predictor (no point collapse, no Yeo-Johnson
        ensemble), denormalizes it back to original-y units, and enforces
        monotonicity by sorting each row's quantiles ascending.
        """
        if not hasattr(self, "X_train_"):
            raise ValueError("Call fit(X, y) before predict(X).")
        # The one chokepoint for every distribution-based path, including the
        # discretize strategies that decode from a quantile bank -- so refusing here
        # covers all of them. A large-context policy returns POINT predictions: it averages
        # or sums the results of several Nori calls, and there is no meaningful way to
        # combine their quantile banks. Silently ignoring the policy and answering from
        # a memory-trimmed full context would hand back numbers the caller believes
        # came from their policy.
        if large_context_applies(len(self.X_train_), self.large_context_policy,
                           self.large_context_threshold):
            raise LargeContextUnsupportedOutputError(
                f"large_context_policy={self.large_context_policy!r} cannot serve "
                f"output_type={output_type!r} (nor a discretize strategy that decodes "
                f"from the predictive distribution): a shared-pool policy chains several "
                f"Nori calls into a POINT prediction and has no combined distribution to "
                f"report. Use output_type='mean'/'median'/'mode', raise "
                f"large_context_threshold above {len(self.X_train_)}, or set "
                f"large_context_policy=None for this call."
            )
        if output_type == "quantiles":
            if not quantiles:
                raise ValueError(
                    "output_type='quantiles' requires quantiles=[...] with at "
                    "least one tau level in (0, 1)."
                )
            q_levels = np.asarray(quantiles, dtype=np.float64)
            if np.any((q_levels <= 0.0) | (q_levels >= 1.0)):
                raise ValueError("quantiles must lie strictly in (0, 1).")

        predictor = self._get_predictor()
        if predictor.regression_head == "bar_distribution":
            raise NotImplementedError(
                "output_type='quantiles'/'full' is not supported for "
                "bar_distribution checkpoints yet; the default pinball "
                "(quantile-head) checkpoint is required."
            )

        X_test = self._prepare_query_features(X)
        y_norm = ((self.y_train_ - self.y_mean_) / self.y_std_).astype(np.float32)

        bank = predictor.predict(self.X_train_, y_norm, X_test, return_distribution=True)
        if isinstance(bank, torch.Tensor):
            bank = bank.detach().cpu().numpy()
        bank = np.asarray(bank, dtype=np.float64)
        if bank.ndim == 1:  # single query row -> [1, K]
            bank = bank[None, :]

        # Denormalize (affine, monotone) then sort each row to a valid quantile
        # function. K quantiles sit at evenly spaced taus = i/(K+1).
        bank = bank * self.y_std_ + self.y_mean_
        Q = np.sort(bank, axis=1)
        K = Q.shape[1]
        taus = (np.arange(K, dtype=np.float64) + 1.0) / (K + 1.0)

        if output_type == "full":
            return {"quantiles": Q, "taus": taus, "mean": Q.mean(axis=1)}

        # output_type == "quantiles": interpolate the inverse-CDF at each level.
        out = np.empty((q_levels.shape[0], Q.shape[0]), dtype=np.float64)
        for i, level in enumerate(q_levels):
            # np.interp is 1-D in the x-grid (shared taus) but loops rows; do a
            # vectorized linear interpolation across all rows for this level.
            pos = np.interp(level, taus, np.arange(K))
            lo = int(np.floor(pos))
            hi = min(lo + 1, K - 1)
            w = pos - lo
            out[i] = (1.0 - w) * Q[:, lo] + w * Q[:, hi]
        return out

    def _predict_categorical(self, X, *, method: str, levels=None):
        """Predict onto a discrete target's level lattice (``discretize=``/``categorical_levels=``).

        ``snap-*`` methods discretize a point prediction with the collapse the
        method names (mean/median), regardless of the configured
        ``quantile_collapse`` — so ``snap-mean`` always snaps the mean; they
        work for every checkpoint, as does ``prior-match`` (point + training
        priors). ``map-cell``/``median-cell``/``expected-level`` need the
        quantile bank and share ``_predict_distribution``'s pinball-checkpoint
        requirement (``expected-level`` returns CONTINUOUS values).
        """
        # validate before the (expensive) forward pass; discretize_predictions
        # re-checks as the canonical gate for direct module users
        if method not in DISCRETIZE_METHODS:
            raise ValueError(
                f"Unknown discretize method {method!r}; expected one of "
                f"{DISCRETIZE_METHODS}."
            )
        if not hasattr(self, "X_train_"):
            raise ValueError("Call fit(X, y) before predict(X).")
        lattice = target_levels(self.y_train_ if levels is None else levels)
        if method in SNAP_METHODS or method == "prior-match":
            collapse = "median" if method == "snap-median" else "mean"
            point = self._predict_point(
                X, quantile_collapse=collapse, bar_point_estimator=collapse)
            return discretize_predictions(
                method, lattice, point=point, y_train=self.y_train_)
        try:
            dist = self._predict_distribution(X, output_type="full", quantiles=None)
        except LargeContextUnsupportedOutputError:
            # Not the checkpoint's doing, and it already names its own cause and remedy.
            # Rewrapping it here sent the caller hunting a bar_distribution problem they
            # do not have.
            raise
        except NotImplementedError as err:
            raise NotImplementedError(
                f"discretize={method!r} needs the quantile bank, which this "
                "bar_distribution checkpoint does not expose. Use "
                "discretize='snap-mean', 'snap-median', or 'prior-match' instead."
            ) from err
        return discretize_predictions(
            method, lattice, Q=dist["quantiles"], taus=dist["taus"])


def infer(
    X_train,
    y_train,
    X_test,
    *,
    task: Task = "regression",
    model_path: str | None = None,
    model: str | None = None,
    token: str | bool | None = None,
    categorical_columns=CATEGORICAL_AUTO,
    max_categorical_cardinality: int = DEFAULT_MAX_CARDINALITY,
    categorical_encoding: str = DEFAULT_CATEGORICAL_ENCODING,
    text_columns=None,
    svd_dim: int | None = 128,
    embedder="minilm",
    text_normalize: bool | None = None,
    discretize: str | None = None,
    categorical_levels=None,
    **kwargs,
):
    """Fit on context rows and infer labels for query rows.

    Accepts Python lists, numpy arrays, or pandas DataFrames. DataFrames use the
    same fitted schema as :class:`NoriRegressor`: ``categorical_columns="auto"``
    encodes remaining non-numeric columns, a sequence encodes exactly those
    columns, and ``None`` disables categorical inference. Named ``text_columns``
    are embedded separately. Ordinal categories are learned from ``X_train`` in
    deterministic order; missing values remain ``NaN`` and rare/unseen values
    use one bounded ``other`` code. Query columns are aligned by name without
    changing the fitted schema. Positional list/array inputs must already be
    numeric.

    ``model`` selects a variant (e.g. ``"nori-30m"``); ``model_path`` still takes
    an explicit local checkpoint and wins over ``model`` when both are given.
    ``discretize`` / ``categorical_levels`` map predictions onto a discrete
    target's levels — see ``NoriRegressor.predict``.
    """
    if task in ("regression", "reg"):
        estimator = NoriRegressor(
            model_path=model_path,
            model=model,
            token=token,
            categorical_columns=categorical_columns,
            categorical_encoding=categorical_encoding,
            max_categorical_cardinality=max_categorical_cardinality,
            text_columns=text_columns,
            svd_dim=svd_dim,
            embedder=embedder,
            text_normalize=text_normalize,
            **kwargs,
        ).fit(X_train, y_train)
        return estimator.predict(
            X_test,
            discretize=discretize,
            categorical_levels=categorical_levels,
        )
    raise ValueError(f"Unsupported task: {task!r}")


def predict(
    X_train,
    y_train,
    X_test,
    *,
    task: Task = "regression",
    model_path: str | None = None,
    model: str | None = None,
    token: str | bool | None = None,
    categorical_columns=CATEGORICAL_AUTO,
    max_categorical_cardinality: int = DEFAULT_MAX_CARDINALITY,
    categorical_encoding: str = DEFAULT_CATEGORICAL_ENCODING,
    text_columns=None,
    svd_dim: int | None = 128,
    embedder="minilm",
    text_normalize: bool | None = None,
    **kwargs,
):
    """Alias for infer()."""
    return infer(
        X_train,
        y_train,
        X_test,
        task=task,
        model_path=model_path,
        model=model,
        token=token,
        categorical_columns=categorical_columns,
        max_categorical_cardinality=max_categorical_cardinality,
        categorical_encoding=categorical_encoding,
        text_columns=text_columns,
        svd_dim=svd_dim,
        embedder=embedder,
        text_normalize=text_normalize,
        **kwargs,
    )
