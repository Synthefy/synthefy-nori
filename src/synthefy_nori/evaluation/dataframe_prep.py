"""Shared preprocessing for pandas-backed benchmark loaders."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from synthefy_nori.featurize import DEFAULT_MAX_CARDINALITY, align_and_featurize

logger = logging.getLogger(__name__)


def featurize_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    max_categorical_cardinality: int = DEFAULT_MAX_CARDINALITY,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Encode categoricals on train only using the package's public featurizer."""
    Xtr_df, Xte_df = align_and_featurize(
        X_train,
        X_test,
        max_categorical_cardinality=max_categorical_cardinality,
    )
    Xtr = np.ascontiguousarray(np.asarray(Xtr_df, dtype=np.float32))
    Xte = np.ascontiguousarray(np.asarray(Xte_df, dtype=np.float32))
    if Xtr.ndim != 2 or Xte.ndim != 2 or Xtr.shape[1] != Xte.shape[1]:
        raise ValueError(f"featurized train/test feature mismatch: {Xtr.shape} vs {Xte.shape}")
    return Xtr, Xte, Xtr.shape[1]


def validate_target(y, name: str = "", source: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """Coerce a regression target to float64 and return its finite-row mask."""
    y_numeric = pd.to_numeric(pd.Series(np.asarray(y).ravel()), errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(y_numeric)
    if len(valid) and valid.mean() < 0.5:
        logger.warning(
            "target %r/%r: %.0f%% of values are non-numeric/non-finite",
            source,
            name,
            100.0 * (1.0 - valid.mean()),
        )
    return y_numeric, valid
