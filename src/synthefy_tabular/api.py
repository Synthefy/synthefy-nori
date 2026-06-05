"""Small public inference API.

The heavy numerical stack is imported lazily so `import synthefy_tabular`
works before optional accelerator dependencies are installed.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Literal


Task = Literal["classification", "regression", "cls", "reg"]


def config_path(filename: str) -> str:
    """Return an absolute path for a bundled inference config."""
    return str(files("synthefy_tabular.configs").joinpath(filename))


def _default_device():
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _as_device(device):
    import torch

    if device is None:
        return _default_device()
    return torch.device(device)


def _resolve_model_path(model_path: str | None, token: str | bool | None = None) -> str:
    if model_path is not None:
        return model_path
    from synthefy_tabular.hf import download_checkpoint

    return download_checkpoint(token=token)


class SynthefyTabularRegressor:
    """Scikit-learn-style regression wrapper around the Synthefy checkpoint."""

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
        import numpy as np

        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.y_train_ = np.asarray(y, dtype=np.float64)
        self.y_mean_ = float(self.y_train_.mean())
        y_std = float(self.y_train_.std())
        self.y_std_ = y_std if y_std >= 1e-12 else 1.0
        return self

    def _get_predictor(self):
        if self._predictor is None:
            from synthefy_tabular.inference.predictor import LimiXPredictor

            self._predictor = LimiXPredictor(
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

    def predict(self, X):
        import numpy as np
        import torch

        if not hasattr(self, "X_train_"):
            raise ValueError("Call fit(X, y) before predict(X).")

        X_test = np.asarray(X, dtype=np.float32)
        y_norm = ((self.y_train_ - self.y_mean_) / self.y_std_).astype(np.float32)
        pred = self._get_predictor().predict(
            self.X_train_,
            y_norm,
            X_test,
            task_type="Regression",
        )
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float64).squeeze()
        return pred * self.y_std_ + self.y_mean_


class SynthefyTabularClassifier:
    """Scikit-learn-style classification wrapper around the Synthefy checkpoint."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device=None,
        inference_config: str | None = None,
        token: str | bool | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.token = token
        self.inference_config = inference_config or config_path("cls_default_noretrieval.json")
        self._predictor = None

    def fit(self, X, y):
        import numpy as np

        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.y_train_ = encoded.astype(np.int64)
        self.n_classes_ = len(self.classes_)
        return self

    def _get_predictor(self):
        if self._predictor is None:
            from synthefy_tabular.inference.predictor import LimiXPredictor

            self._predictor = LimiXPredictor(
                device=_as_device(self.device),
                model_path=_resolve_model_path(self.model_path, self.token),
                inference_config=self.inference_config,
            )
        return self._predictor

    def predict_proba(self, X):
        import numpy as np
        import torch

        if not hasattr(self, "X_train_"):
            raise ValueError("Call fit(X, y) before predict_proba(X).")

        X_test = np.asarray(X, dtype=np.float32)
        pred = self._get_predictor().predict(
            self.X_train_,
            self.y_train_,
            X_test,
            task_type="Classification",
        )
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim == 2 and pred.shape[1] > self.n_classes_:
            pred = pred[:, : self.n_classes_]
        row_sums = pred.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return pred / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


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
        model = SynthefyTabularRegressor(model_path=model_path, token=token, **kwargs).fit(X_train, y_train)
        return model.predict(X_test)
    if task in ("classification", "cls"):
        model = SynthefyTabularClassifier(model_path=model_path, token=token, **kwargs).fit(X_train, y_train)
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
