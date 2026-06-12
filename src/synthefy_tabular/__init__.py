"""Public API for Synthefy Tabular."""

from __future__ import annotations

from synthefy_tabular.api import (
    SynthefyTabularRegressor,
    config_path,
    infer,
    predict,
)

__version__ = "0.2.2"

__all__ = [
    "SynthefyTabularRegressor",
    "config_path",
    "infer",
    "predict",
]
