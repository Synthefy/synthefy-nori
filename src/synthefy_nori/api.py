"""Small public inference API."""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin


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


def _resolve_model_path(model_path: str | None, token: str | bool | None = None) -> str:
    if model_path is not None:
        return model_path
    from synthefy_nori.hf import download_checkpoint

    return download_checkpoint(token=token)


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
        device=None,
        inference_config: str | None = None,
        token: str | bool | None = None,
        augmentations: tuple[str, ...] | list[str] | None = ("yj",),
        yj_skew_threshold: float = 10.0,
        quantile_collapse: str = "mean",
        bar_temperature: float = 1.0,
        bar_point_estimator: str = "mean",
    ) -> None:
        self.model_path = model_path
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
        self._predictor = None

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.n_features_in_ = self.X_train_.shape[1]
        self.y_train_ = np.asarray(y, dtype=np.float64)
        self.y_mean_ = float(self.y_train_.mean())
        y_std = float(self.y_train_.std())
        self.y_std_ = y_std if y_std >= 1e-12 else 1.0
        return self

    def _get_predictor(self):
        if self._predictor is None:
            from synthefy_nori.inference.predictor import NoriPredictor

            self._predictor = NoriPredictor(
                device=_as_device(self.device),
                model_path=_resolve_model_path(self.model_path, self.token),
                inference_config=self.inference_config,
                augmentations=self.augmentations,
                yj_skew_threshold=self.yj_skew_threshold,
                quantile_collapse=self.quantile_collapse,
                bar_temperature=self.bar_temperature,
                bar_point_estimator=self.bar_point_estimator,
            )
        return self._predictor

    def predict(self, X, *, output_type: str = "mean", quantiles: list[float] | None = None):
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
        """
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

        if not hasattr(self, "X_train_"):
            raise ValueError("Call fit(X, y) before predict(X).")

        X_test = np.asarray(X, dtype=np.float32)
        y_norm = ((self.y_train_ - self.y_mean_) / self.y_std_).astype(np.float32)

        predictor = self._get_predictor()
        # Drive the predictor's distribution-collapse from output_type. "mean"
        # restores the regressor's configured collapse so the default path is
        # byte-for-byte the prior behavior; "median"/"mode" override it for this
        # call. A quantile head has no native mode, so "mode" falls back to the
        # median there, while bar-distribution heads decode a true mode.
        if output_type == "mean":
            predictor.quantile_collapse = self.quantile_collapse
            predictor.bar_point_estimator = self.bar_point_estimator
        else:
            predictor.quantile_collapse = "median"
            predictor.bar_point_estimator = output_type

        pred = predictor.predict(self.X_train_, y_norm, X_test)
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
        X_test = np.asarray(X, dtype=np.float32)
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

        X_test = np.asarray(X, dtype=np.float32)
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


def infer(
    X_train,
    y_train,
    X_test,
    *,
    task: Task = "regression",
    model_path: str | None = None,
    token: str | bool | None = None,
    **kwargs,
):
    """Fit on context rows and infer labels for query rows."""
    if task in ("regression", "reg"):
        model = NoriRegressor(model_path=model_path, token=token, **kwargs).fit(X_train, y_train)
        return model.predict(X_test)
    raise ValueError(f"Unsupported task: {task!r}")


def predict(
    X_train,
    y_train,
    X_test,
    *,
    task: Task = "regression",
    model_path: str | None = None,
    token: str | bool | None = None,
    **kwargs,
):
    """Alias for infer()."""
    return infer(X_train, y_train, X_test, task=task, model_path=model_path, token=token, **kwargs)
