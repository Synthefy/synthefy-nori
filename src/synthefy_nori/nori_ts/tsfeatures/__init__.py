"""Vendored time-series feature engineering from PriorLabs/tabpfn-time-series.

Vendored from PriorLabs/tabpfn-time-series @ `a756ae3` (2026-07-13), which was
upstream `main` at vendoring time. Upstream:
https://github.com/PriorLabs/tabpfn-time-series

Five of the six vendored modules are byte-identical to that revision apart from
having their intra-package imports rewritten to this subpackage:

    basic_features.py  auto_features.py  feature_generator_base.py
    feature_transformer.py  ts_dataframe.py

`data_preparation.py` carries the one deliberate local change, named in its own
file header: `generate_test_X` takes an explicit `freq`, because re-inferring it
from a NaN-dropped index yields None. That is the only behavioral delta in this
subpackage. (`ts_dataframe.py` is unmodified by Synthefy; its header records the
AutoGluon -> Prior Labs -> here chain.)

Vendoring (rather than depending on tabpfn-time-series) keeps the tabpfn /
tabpfn_client runtime dependencies out of the tree. No TabPFN model code or
weights are included.

Re-checking the pin: normalize the import paths and diff against upstream, e.g.

    sed 's#synthefy_nori.nori_ts.tsfeatures#tabpfn_time_series.features#g' \\
        basic_features.py | diff - <(git -C <upstream> show a756ae3:\\
        tabpfn_time_series/features/basic_features.py)

Generator contract (changed by upstream #144, and the reason `tsfeatures/` must be
re-vendored as a unit rather than file by file): `FeatureGenerator.generate`
receives the *whole* `(item_id, timestamp)`-indexed frame with every series at
once, and must group on the `item_id` level for any per-series computation. It is
no longer called once per series.

One consequence for callers: `PeriodicSinCosineFeature` computes its phase from a
frame-global `np.arange(len(df))` rather than a per-series counter, so it is only
correct on a single-series frame. It is not part of the default feature set —
`AutoSeasonalFeature` stopped delegating to it in #144 and now derives per-series
phase from `groupby.cumcount()` — and it is deliberately not re-exported from this
package. Don't reach for it on a multi-series frame.
"""

from synthefy_nori.nori_ts.tsfeatures.ts_dataframe import TimeSeriesDataFrame
from synthefy_nori.nori_ts.tsfeatures.data_preparation import generate_test_X
from synthefy_nori.nori_ts.tsfeatures.feature_generator_base import FeatureGenerator
from synthefy_nori.nori_ts.tsfeatures.feature_transformer import FeatureTransformer
from synthefy_nori.nori_ts.tsfeatures.basic_features import (
    RunningIndexFeature,
    CalendarFeature,
    AdditionalCalendarFeature,
)
from synthefy_nori.nori_ts.tsfeatures.auto_features import AutoSeasonalFeature

# PeriodicSinCosineFeature is intentionally absent: under the post-#144 whole-frame
# generator contract its phase comes from a frame-global counter, so it is only
# correct on a single-series frame. Import it from .basic_features explicitly if you
# know that holds.
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
