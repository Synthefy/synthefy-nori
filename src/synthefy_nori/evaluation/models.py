"""Model registry for the unified evaluation pipeline.

Wraps Nori checkpoints for benchmarking.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Dict, Optional

import numpy as np
import torch


def package_config_path(filename: str) -> str:
    return str(files("synthefy_nori.configs").joinpath(filename))


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
        augmentations: tuple|list|None = None,
        yj_skew_threshold: float = 10.0,
        quantile_collapse: str = 'mean',
        bar_temperature: float = 1.0,
        bar_point_estimator: str = 'mean',
        memory_policy=None,
    ):
        self._name = model_name
        self.model_path = model_path
        self.device = torch.device(device)
        self.reg_config_path = reg_config_path or package_config_path(
            "reg_allordinal_poly10_adaptive_svd256.json"
        )
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
    model_type: str  # "synthefy" or "custom"
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
        quantile_collapse: str = 'mean',
        bar_temperature: float = 1.0,
        bar_point_estimator: str = 'mean',
        memory_policy=None,
        metadata=None,
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
            memory_policy=memory_policy,
        )
        identity = {
            **(metadata or {}),
            "device": device,
            "memory_policy": memory_policy,
        }
        for private_key in (
            "model_path", "checkpoint_path", "reg_config", "reg_config_path",
        ):
            identity.pop(private_key, None)
        self.register(ModelEntry(
            name=name, wrapper=wrapper, model_type="synthefy",
            description=description,
            metadata=identity,
        ))

    def cleanup_all(self):
        for entry in self._models.values():
            entry.wrapper.cleanup()
