"""Vendored time-series feature engineering from PriorLabs/tabpfn-time-series.

These modules (feature generators, FeatureTransformer, TimeSeriesDataFrame,
generate_test_X) are copied verbatim from tabpfn-time-series with only their
intra-package imports rewritten to this subpackage — so Nori-TS reuses the
*exact* feature pipeline TabPFN-TS uses, without pulling in the tabpfn /
tabpfn_client runtime dependencies. TimeSeriesDataFrame itself is adapted from
AutoGluon (see the header in ts_dataframe.py). Upstream: https://github.com/PriorLabs/tabpfn-time-series
"""

from synthefy_nori.nori_ts.tsfeatures.ts_dataframe import TimeSeriesDataFrame
from synthefy_nori.nori_ts.tsfeatures.data_preparation import generate_test_X
from synthefy_nori.nori_ts.tsfeatures.feature_generator_base import FeatureGenerator
from synthefy_nori.nori_ts.tsfeatures.feature_transformer import FeatureTransformer
from synthefy_nori.nori_ts.tsfeatures.basic_features import (
    RunningIndexFeature,
    CalendarFeature,
    AdditionalCalendarFeature,
    PeriodicSinCosineFeature,
)
from synthefy_nori.nori_ts.tsfeatures.auto_features import AutoSeasonalFeature

__all__ = [
    "TimeSeriesDataFrame",
    "generate_test_X",
    "FeatureGenerator",
    "FeatureTransformer",
    "RunningIndexFeature",
    "CalendarFeature",
    "AdditionalCalendarFeature",
    "PeriodicSinCosineFeature",
    "AutoSeasonalFeature",
]
