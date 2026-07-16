"""Small public inference API."""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from synthefy_nori.discretize import (
    DEFAULT_DISCRETIZE_METHOD,
    DISCRETIZE_METHODS,
    SNAP_METHODS,
    discretize_predictions,
    target_levels,
)
from synthefy_nori.text_features import MultimodalPreprocessor


Task = Literal["regression", "reg"]


def config_path(filename: str) -> str:
    """Return an absolute path for a bundled inference config."""
    return str(files("synthefy_nori.configs").joinpath(filename))


def _default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _as_device(device):
    if device is None:
        return _default_device()
    return torch.device(device)


def _resolve_model_path(model_path: str | None, token: str | bool | None = None,
                        model: str | None = None) -> str:
    if model_path is not None:
        return model_path
    from synthefy_nori.hf import download_checkpoint

    return download_checkpoint(model=model, token=token)


class NoriRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn regression estimator wrapping the Synthefy checkpoint.

    Subclasses ``BaseEstimator``/``RegressorMixin`` so it works directly with the
    scikit-learn ecosystem (``clone``, ``get_params``/``set_params``, ``score``,
    partial dependence, sequential feature selection) and with shapiq — see
    ``synthefy_nori.interpretability``. The ``__init__`` arguments are stored
    verbatim (the only normalizations applied are idempotent), so ``clone`` round
    trips correctly.
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
        text_columns=None,
        svd_dim: int | None = 128,
        embedder="minilm",
        text_max_cardinality: int = 128,
        text_normalize: bool | None = None,
    ) -> None:
        """Configure the estimator (arguments are stored verbatim; see class docs).

        Args:
            model_path: path to a local ``.pt`` checkpoint. ``None`` (default)
                downloads/caches the public Hugging Face checkpoint.
            model: variant selector (``"nori"`` default / ``"nori-6m"`` /
                ``"nori-30m"``), resolved to a Hugging Face repo. Ignored when
                ``model_path`` is given.
            device: torch device for inference (``"cuda:0"``, ``"cpu"``, ...).
                ``None`` picks CUDA when available, else CPU.
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
            text_columns: enables the zero-shot text path when set (requires ``X``
                to be a :class:`pandas.DataFrame`). A list of column names embeds
                those columns; ``[]`` is a valid numeric+categorical-only fit (no
                embedder loaded). ``None`` (default) treats ``X`` as a plain numeric
                array. Columns must be named explicitly; the estimator does not
                infer which columns are text.
            svd_dim: width of the TruncatedSVD text block appended to the numeric
                features (fit on train only). Default 128; ``None`` appends the full
                raw embedding.
            embedder: sentence encoder for ``text_columns`` — a short name / HF id
                string (e.g. ``"minilm"``; needs the optional
                ``sentence-transformers`` extra), a preloaded encoder object, or a
                callable ``texts -> ndarray``.
            text_max_cardinality: a non-numeric column with more than this many
                distinct values is embedded rather than label-encoded; categorical
                encoding also caps at this many values (rarer/unseen -> "other").
            text_normalize: cosine-normalize the text embeddings. ``None`` (default)
                auto-enables for known LLM encoders; set True/False to override
                (needed for a preloaded encoder object).

        The discretize/categorical_levels pair are estimator-level defaults; the
        same-named ``predict`` kwargs override them per call. The text_* params
        take effect at ``fit``.
        """
        self.model_path = model_path
        # Variant selector: "nori" (default, ~6M base) / "nori-6m" / "nori-30m", resolved to a
        # Hugging Face repo via synthefy_nori.hf.NORI_MODELS. Ignored when model_path is given.
        self.model = model
        self.device = device
        self.token = token
        self.inference_config = inference_config or config_path(
            "reg_allordinal_poly10_adaptive_svd256.json"
        )
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
        # Zero-shot text config — constructor params (not fit kwargs) so the
        # feature round-trips through clone/get_params/GridSearchCV/cross_val_score
        # and pickle, like discretize/categorical_levels above.
        #   text_columns: None -> numeric-array path; [] -> DataFrame numeric+
        #     categorical only; list of names -> embed those columns.
        #   svd_dim: SVD width of the appended text block (None = raw embedding).
        #   embedder: short name / preloaded encoder / callable.
        self.text_columns = text_columns
        self.svd_dim = svd_dim
        self.embedder = embedder
        self.text_max_cardinality = int(text_max_cardinality)
        self.text_normalize = text_normalize
        self._predictor = None
        # Fitted text preprocessor (embed -> SVD -> extra columns), built by fit()
        # when text_columns is not None; None = numeric-only.
        self._text_preprocessor = None

    def fit(self, X, y):
        """Fit the in-context regressor on ``(X, y)``.

        Numeric-only (default, ``text_columns=None``): ``X`` is any array-like
        coerced to a float32 matrix. Zero-shot text: set ``text_columns`` in the
        constructor and pass ``X`` as a DataFrame — the named columns are embedded
        by a frozen sentence encoder, reduced to ``svd_dim`` columns (SVD fit on
        train), and appended; the encoder and Nori stay frozen. ``predict`` replays
        the transform, so its ``X`` must be a DataFrame with these columns.
        """
        if self.text_columns is not None:
            self._text_preprocessor = MultimodalPreprocessor(
                self.text_columns, svd_dim=self.svd_dim, embedder=self.embedder,
                device=self.device, max_cardinality=self.text_max_cardinality,
                normalize=self.text_normalize)
            X_mat = self._text_preprocessor.fit_transform(X)
            # sklearn contract: n_features_in_ counts INPUT features, not the
            # widened SVD block; record column names when we have them.
            self.n_features_in_ = X.shape[1]
            if hasattr(X, "columns"):
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self._text_preprocessor = None
            X_mat = np.asarray(X, dtype=np.float32)
            self.n_features_in_ = X_mat.shape[1]
        self.X_train_ = X_mat.astype(np.float32)
        self.y_train_ = np.asarray(y, dtype=np.float64)
        self.y_mean_ = float(self.y_train_.mean())
        y_std = float(self.y_train_.std())
        self.y_std_ = y_std if y_std >= 1e-12 else 1.0
        return self

    def _prepare_query_features(self, X) -> np.ndarray:
        """Coerce query rows to the model's float32 matrix, applying the fitted
        text transform (embed -> SVD -> append) when fit configured one."""
        if getattr(self, "_text_preprocessor", None) is not None:
            return self._text_preprocessor.transform(X).astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def _get_predictor(self):
        if self._predictor is None:
            from synthefy_nori.inference.predictor import NoriPredictor

            self._predictor = NoriPredictor(
                device=_as_device(self.device),
                model_path=_resolve_model_path(self.model_path, self.token, self.model),
                inference_config=self.inference_config,
                augmentations=self.augmentations,
                yj_skew_threshold=self.yj_skew_threshold,
                quantile_collapse=self.quantile_collapse,
                bar_temperature=self.bar_temperature,
                bar_point_estimator=self.bar_point_estimator,
                discrete_y_snap_max_unique=self.discrete_y_snap_max_unique,
            )
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

        predictor = self._get_predictor()
        saved = (predictor.quantile_collapse, predictor.bar_point_estimator)
        try:
            predictor.quantile_collapse = quantile_collapse
            predictor.bar_point_estimator = bar_point_estimator
            pred = predictor.predict(self.X_train_, y_norm, X_test)
        finally:
            predictor.quantile_collapse, predictor.bar_point_estimator = saved
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float64).squeeze()
        return pred * self.y_std_ + self.y_mean_

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
    discretize: str | None = None,
    categorical_levels=None,
    **kwargs,
):
    """Fit on context rows and infer labels for query rows.

    ``model`` selects a variant (e.g. ``"nori-30m"``); ``model_path`` still takes an explicit
    local checkpoint and wins over ``model`` when both are given.
    ``discretize`` / ``categorical_levels`` map predictions onto a discrete
    target's levels — see ``NoriRegressor.predict``.
    """
    if task in ("regression", "reg"):
        estimator = NoriRegressor(
            model_path=model_path, model=model, token=token, **kwargs
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
    **kwargs,
):
    """Alias for infer()."""
    return infer(
        X_train, y_train, X_test, task=task, model_path=model_path, model=model, token=token, **kwargs
    )
