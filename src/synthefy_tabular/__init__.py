"""Public API for Synthefy Tabular."""

from synthefy_tabular.api import (
    SynthefyTabularClassifier,
    SynthefyTabularRegressor,
    config_path,
    infer,
    predict,
)

__version__ = "0.1.0"

__all__ = [
    "SynthefyTabularClassifier",
    "SynthefyTabularRegressor",
    "config_path",
    "infer",
    "predict",
]
