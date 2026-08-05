"""Vendored time-series feature engineering from PriorLabs/tabpfn-time-series.

Vendored from PriorLabs/tabpfn-time-series @ `d4b456d` (2026-06-17), i.e. the
feature pipeline TabPFN-TS used *at that revision* — not necessarily today's
upstream `main`. Upstream: https://github.com/PriorLabs/tabpfn-time-series

Against `d4b456d`, four of the six vendored modules are byte-identical apart from
having their intra-package imports rewritten to this subpackage:

    basic_features.py  auto_features.py  feature_generator_base.py
    feature_transformer.py

The remaining two carry deliberate local changes, each named in its own file
header:

    data_preparation.py  `generate_test_X` takes an explicit `freq` (re-inferring
                         it from a NaN-dropped index yields None). The only
                         behavioral delta in this subpackage.
    ts_dataframe.py      unmodified by Synthefy; its header records the AutoGluon
                         -> Prior Labs -> here chain.

Vendoring (rather than depending on tabpfn-time-series) keeps the tabpfn /
tabpfn_client runtime dependencies out of the tree. No TabPFN model code or
weights are included.

Re-checking the pin: normalize the import paths and diff against upstream, e.g.

    sed 's#synthefy_nori.nori_ts.tsfeatures#tabpfn_time_series.features#g' \\
        basic_features.py | diff - <(git -C <upstream> show d4b456d:\\
        tabpfn_time_series/features/basic_features.py)

Known drift: upstream `main` has since landed #142 (featurization host-memory
peak) and #144 (many-series featurizer speed; flips the generator contract to
whole-frame + downcasts generated columns to float32). Taking those changes moves
GIFT-eval numbers, so they are deliberately not vendored here — see the
re-vendoring follow-up issue.
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
