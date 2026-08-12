"""Compatibility facade for the historical forecasting implementation path."""

from synthefy.nori_ts.core import (
    DEFAULT_QUANTILES,
    AutoSeasonalFeature,
    CalendarFeature,
    FeatureTransformer,
    NoriTSForecaster,
    RunningIndexFeature,
    TimeSeriesDataFrame,
    _default_features,
    generate_test_X,
    _TARGET,
)

__all__ = ["NoriTSForecaster", "DEFAULT_QUANTILES"]
