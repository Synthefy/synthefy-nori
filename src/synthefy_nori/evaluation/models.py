"""Model registry for the unified evaluation pipeline.

Wraps Nori checkpoints for benchmarking.
"""

from __future__ import annotations

import gc
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from synthefy_nori.configs import DEFAULT_INFERENCE_CONFIG, config_path
from synthefy_nori.model.quantile_dist import quantile_dist_mean_numpy


def package_config_path(filename: str) -> str:
    """Long-standing alias for :func:`synthefy_nori.configs.config_path`."""
    return config_path(filename)


# ---------------------------------------------------------------------------
# Base model wrapper
# ---------------------------------------------------------------------------


class BaseModelWrapper(ABC):
    """Abstract base for all model wrappers in the eval pipeline."""

    @abstractmethod
    def predict_regression(self, X_train, y_train, X_test):
        """Return predictions: np.ndarray [n_test]"""
        pass

    @abstractmethod
    def cleanup(self):
        """Free GPU memory and resources."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def device_str(self) -> str:
        return "cpu"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class NoriWrapper(BaseModelWrapper):
    """Wrapper around NoriPredictor for unified eval."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str = "cuda:0",
        reg_config_path: str | None = None,
        base_config_path: Optional[str] = None,
        augmentations: tuple | list | None = None,
        yj_skew_threshold: float = 10.0,
        quantile_collapse: str = "mean",
        bar_temperature: float = 1.0,
        bar_point_estimator: str = "mean",
        memory_policy=None,
    ):
        self._name = model_name
        self.model_path = model_path
        self.device = torch.device(device)
        self.reg_config_path = reg_config_path or config_path(DEFAULT_INFERENCE_CONFIG)
        self.base_config_path = base_config_path
        self.augmentations = tuple(augmentations) if augmentations else ()
        self.yj_skew_threshold = float(yj_skew_threshold)
        self.quantile_collapse = quantile_collapse
        self.bar_temperature = float(bar_temperature)
        self.bar_point_estimator = bar_point_estimator
        self.memory_policy = memory_policy
        self._reg_predictor = None

    @property
    def name(self):
        return self._name

    @property
    def device_str(self):
        return str(self.device)

    def _get_reg_predictor(self):
        if self._reg_predictor is None:
            from synthefy_nori.inference.predictor import NoriPredictor
            from synthefy_nori.utils.loading import load_model

            model = load_model(
                self.model_path,
                mask_prediction=False,
                base_config_path=self.base_config_path,
            )
            self._reg_predictor = NoriPredictor(
                device=self.device,
                inference_config=self.reg_config_path,
                model=model,
                augmentations=self.augmentations,
                yj_skew_threshold=self.yj_skew_threshold,
                quantile_collapse=self.quantile_collapse,
                bar_temperature=self.bar_temperature,
                bar_point_estimator=self.bar_point_estimator,
                memory_policy=self.memory_policy,
            )
        return self._reg_predictor

    def predict_regression(self, X_train, y_train, X_test):
        predictor = self._get_reg_predictor()
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float64)

        # Normalize y for the model
        y_mean, y_std = y_train.mean(), y_train.std()
        if y_std < 1e-12:
            y_std = 1.0
        y_train_norm = (y_train - y_mean) / y_std

        pred = predictor.predict(X_train, y_train_norm.astype(np.float32), X_test)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        pred = np.asarray(pred, dtype=np.float64).squeeze()

        # Denormalize
        return pred * y_std + y_mean

    def predict_distribution(self, X_train, y_train, X_test):
        """Return ``(quantiles, taus, mean)`` for a pinball checkpoint."""
        predictor = self._get_reg_predictor()
        predictor.memory_report_ = None
        regression_head = getattr(predictor, "regression_head", None)
        if regression_head is None:
            regression_head = "mse" if int(getattr(predictor, "num_reg_quantiles", 1)) == 1 else "pinball"
        if regression_head != "pinball":
            raise NotImplementedError(
                "predict_distribution needs the pinball (quantile-head) checkpoint; "
                f"a {regression_head} checkpoint is not supported."
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float64)
        y_mean, y_std = y_train.mean(), y_train.std()
        if y_std < 1e-12:
            y_std = 1.0
        y_norm = ((y_train - y_mean) / y_std).astype(np.float32)
        bank = predictor.predict(X_train, y_norm, X_test, return_distribution=True)
        if isinstance(bank, torch.Tensor):
            bank = bank.detach().cpu().numpy()
        bank = np.asarray(bank, dtype=np.float64)
        if bank.ndim == 1:
            bank = bank[None, :]
        bank = bank * y_std + y_mean
        quantiles = np.sort(bank, axis=1)
        count = quantiles.shape[1]
        taus = np.asarray(predictor.regression_quantiles, dtype=np.float64)
        if taus.shape != (count,):
            raise RuntimeError(
                "Checkpoint quantile metadata does not match decoder output: "
                f"{taus.shape[0]} levels for {count} columns."
            )
        mean = quantile_dist_mean_numpy(
            quantiles,
            taus,
            enforce_monotone_first=False,
        )
        return quantiles, taus, mean

    def cleanup(self):
        self._reg_predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model Entry and Registry
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    """Metadata about a registered model."""

    name: str
    wrapper: BaseModelWrapper
    model_type: str  # "synthefy", "custom"
    description: str = ""
    metadata: dict = field(default_factory=dict)


class ModelRegistry:
    """Central registry for all models to evaluate."""

    def __init__(self, device="cuda:0"):
        self.device = device
        self._models: Dict[str, ModelEntry] = {}

    def list_models(self):
        return sorted(self._models.keys())

    def get(self, name):
        return self._models.get(name)

    def register(self, entry: ModelEntry):
        self._models[entry.name] = entry
        print(f"[ModelRegistry] Registered: {entry.name}")

    # ------------------------------------------------------------------
    # Convenience registration methods
    # ------------------------------------------------------------------
    def add_checkpoint(
        self,
        name,
        model_path,
        device=None,
        reg_config=None,
        base_config_path=None,
        description="",
        augmentations=None,
        yj_skew_threshold: float = 10.0,
        quantile_collapse: str = "mean",
        bar_temperature: float = 1.0,
        bar_point_estimator: str = "mean",
    ):
        device = device or self.device
        wrapper = NoriWrapper(
            model_name=name,
            model_path=model_path,
            device=device,
            reg_config_path=reg_config,
            base_config_path=base_config_path,
            augmentations=augmentations,
            yj_skew_threshold=yj_skew_threshold,
            quantile_collapse=quantile_collapse,
            bar_temperature=bar_temperature,
            bar_point_estimator=bar_point_estimator,
        )
        self.register(
            ModelEntry(
                name=name,
                wrapper=wrapper,
                model_type="synthefy",
                description=description,
                metadata={"model_path": model_path, "device": device},
            )
        )

    def cleanup_all(self):
        for entry in self._models.values():
            entry.wrapper.cleanup()
