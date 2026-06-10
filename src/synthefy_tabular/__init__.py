"""Public API for Synthefy Tabular."""

from synthefy_tabular.api import (
    SynthefyTabularRegressor,
    config_path,
    infer,
    predict,
)

__version__ = "0.2.1"

__all__ = [
    "SynthefyTabularRegressor",
    "config_path",
    "infer",
    "predict",
]
