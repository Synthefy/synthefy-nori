"""Compatibility facade for historical Nori time-series feature imports.

Synthefy 7 owns the canonical implementation in
``synthefy.nori_ts.tsfeatures``. The existing deep modules in this package
remain only as the exact released-6.3 fallback until the compatibility floor
moves to Synthefy 7.
"""

from __future__ import annotations

import importlib
import sys


_CANONICAL_PACKAGE = "synthefy.nori_ts.tsfeatures"
_DEEP_MODULES = (
    "auto_features",
    "basic_features",
    "data_preparation",
    "feature_generator_base",
    "feature_transformer",
    "ts_dataframe",
)

try:
    _canonical = importlib.import_module(_CANONICAL_PACKAGE)
except ModuleNotFoundError as exc:
    if exc.name not in {"synthefy.nori_ts", _CANONICAL_PACKAGE}:
        raise

    from synthefy_nori.nori_ts.tsfeatures.auto_features import AutoSeasonalFeature
    from synthefy_nori.nori_ts.tsfeatures.basic_features import (
        AdditionalCalendarFeature,
        CalendarFeature,
        RunningIndexFeature,
    )
    from synthefy_nori.nori_ts.tsfeatures.data_preparation import generate_test_X
    from synthefy_nori.nori_ts.tsfeatures.feature_generator_base import FeatureGenerator
    from synthefy_nori.nori_ts.tsfeatures.feature_transformer import FeatureTransformer
    from synthefy_nori.nori_ts.tsfeatures.ts_dataframe import TimeSeriesDataFrame
else:
    for _module_name in _DEEP_MODULES:
        _module = importlib.import_module(f"{_CANONICAL_PACKAGE}.{_module_name}")
        sys.modules[f"{__name__}.{_module_name}"] = _module
        globals()[_module_name] = _module

    TimeSeriesDataFrame = _canonical.TimeSeriesDataFrame
    generate_test_X = _canonical.generate_test_X
    FeatureGenerator = _canonical.FeatureGenerator
    FeatureTransformer = _canonical.FeatureTransformer
    RunningIndexFeature = _canonical.RunningIndexFeature
    CalendarFeature = _canonical.CalendarFeature
    AdditionalCalendarFeature = _canonical.AdditionalCalendarFeature
    AutoSeasonalFeature = _canonical.AutoSeasonalFeature

__all__ = [
    "TimeSeriesDataFrame",
    "generate_test_X",
    "FeatureGenerator",
    "FeatureTransformer",
    "RunningIndexFeature",
    "CalendarFeature",
    "AdditionalCalendarFeature",
    "AutoSeasonalFeature",
]
