"""Model registry for the unified evaluation pipeline.

Wraps Nori checkpoints for benchmarking.
"""

from __future__ import annotations

import gc
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional

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


class NoriEnsembleWrapper(BaseModelWrapper):
    """Average predictions from N Synthefy checkpoints.

    Each component is a fully-configured NoriWrapper (its own checkpoint,
    inference config, augmentations). The ensemble runs each component for
    every dataset and averages predictions in y-space.

    Component weights default to uniform (1/N each). Pass `weights` for a
    weighted ensemble.

    Memory: holds all N models in GPU memory simultaneously. For 5-12M-param
    models on H200 (143 GB), 2-4 component ensembles fit comfortably; for
    larger ensembles consider sequential load+predict+free, but that adds
    model-load overhead per dataset.
    """

    def __init__(self, model_name: str, components: list,
                 weights: list | None = None):
        if not components:
            raise ValueError("NoriEnsembleWrapper requires at least one component")
        self._name = model_name
        self.components = components
        self._device = components[0].device_str
        if weights is None:
            self.weights = np.ones(len(components)) / float(len(components))
        else:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != len(components):
                raise ValueError(f"weights length {len(w)} != components {len(components)}")
            self.weights = w / w.sum()  # normalize to sum=1

    @property
    def name(self):
        return self._name

    @property
    def device_str(self):
        return self._device

    def predict_regression(self, X_train, y_train, X_test):
        preds = None
        for w, c in zip(self.weights, self.components):
            p = np.asarray(c.predict_regression(X_train, y_train, X_test), dtype=np.float64)
            if preds is None:
                preds = w * p
            else:
                preds = preds + w * p
        return preds

    def cleanup(self):
        for c in self.components:
            try:
                c.cleanup()
            except Exception:
                pass
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
    model_type: str  # "synthefy", "synthefy_ensemble", "custom"
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
        self.register(ModelEntry(
            name=name, wrapper=wrapper, model_type="synthefy",
            description=description,
            metadata={"model_path": model_path, "device": device},
        ))

    def add_synthefy_ensemble(
        self,
        ensemble_name: str,
        component_specs: list,  # list of dicts: {path, label, reg_config, ...}
        device=None,
        default_reg_config=None,
        augmentations=None,
        yj_skew_threshold: float = 10.0,
        weights=None,
        description="",
    ):
        """Register an ensemble of N Synthefy checkpoints.

        Each component_spec is a dict with keys:
          - path (required): checkpoint path
          - label (optional): logging label, defaults to filename
          - reg_config (optional): override default reg config for this component
        Predictions are averaged in y-space.
        """
        device = device or self.device
        components: list[NoriWrapper] = []
        for spec in component_specs:
            if isinstance(spec, str):
                spec = {"path": spec}
            path = spec["path"]
            label = spec.get("label") or os.path.splitext(os.path.basename(path))[0]
            cmp_reg_config = spec.get("reg_config") or default_reg_config
            wrapper = NoriWrapper(
                model_name=label,
                model_path=path,
                device=device,
                reg_config_path=cmp_reg_config,
                augmentations=augmentations,
                yj_skew_threshold=yj_skew_threshold,
            )
            components.append(wrapper)

        ens = NoriEnsembleWrapper(ensemble_name, components, weights=weights)
        self.register(ModelEntry(
            name=ensemble_name, wrapper=ens, model_type="synthefy_ensemble",
            description=description or f"Ensemble of {len(components)} Synthefy checkpoints",
            metadata={"components": [c.model_path for c in components],
                      "weights": (weights if weights else [1.0/len(components)]*len(components))},
        ))

    def cleanup_all(self):
        for entry in self._models.values():
            entry.wrapper.cleanup()
