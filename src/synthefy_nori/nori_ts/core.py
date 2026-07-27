"""Nori-TS core: feature engineering + per-series Nori quantile forecasting.

Pipeline (mirrors TabPFN-TS):
  1. A time series becomes a table indexed by (item_id, timestamp).
  2. `FeatureTransformer` adds time features to every row of train + horizon:
       - RunningIndexFeature : 0..T running counter (captures trend / position)
       - CalendarFeature     : `year` + sin/cos of second/minute/hour/day-of-week/
                               day-of-month/day-of-year/week/month seasonalities
       - AutoSeasonalFeature : FFT-detected dominant periods -> sin/cos pairs
  3. Per series, Nori regresses target ~ features (train rows), then predicts the
     horizon rows. Nori's quantile head yields probabilistic forecasts directly.

The horizon ("test") rows carry NaN targets and known future time features, so
this is target-history + known-future-calendar regression — no leakage.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from synthefy_nori.nori_ts.tsfeatures import (
    TimeSeriesDataFrame,
    FeatureTransformer,
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
    generate_test_X,
)

# GIFT-eval / TabPFN-TS default quantile grid.
DEFAULT_QUANTILES: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

_TARGET = "target"


def _default_features():
    # Fresh instances (AutoSeasonalFeature holds config; cheap to rebuild).
    return [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]


class NoriTSForecaster:
    """Forecast univariate series by tabular regression with Nori.

    Parameters
    ----------
    device : str | None
        Torch device for Nori (e.g. "cuda:0"); None auto-selects.
    context_length : int
        Cap on history rows per series (last-N kept), matching TabPFN-TS's 4096.
    quantiles : list[float]
        Forecast quantile levels (also used to derive the point/median forecast).
    features : list | None
        Feature generators; defaults to the TabPFN-TS default set.
    model : str
        Nori variant to use when ``model_path`` is None ("nori-6m" base default,
        or "nori-30m"). Ignored when ``model_path`` is set.
    model_path : str | None
        Explicit Nori checkpoint; overrides ``model``. None -> resolve ``model``.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        context_length: int = 4096,
        quantiles: Optional[List[float]] = None,
        features=None,
        model: str = "nori-6m",
        model_path: Optional[str] = None,
    ):
        self.device = device
        self.context_length = context_length
        # Sorted ascending so the quantile column labels (q_names) stay aligned
        # with the value-sorted forecast rows — otherwise a caller passing e.g.
        # [0.9, 0.5, 0.1] would get columns labelled in input order but ordered
        # by value.
        self.quantiles = sorted(list(quantiles) if quantiles is not None else DEFAULT_QUANTILES)
        self.features = features if features is not None else _default_features()
        self.model = model
        self.model_path = model_path
        self._model = None  # lazily built, reused across series (context model)

    # ------------------------------------------------------------------ model
    def _get_model(self):
        if self._model is None:
            from synthefy_nori import NoriRegressor

            self._model = NoriRegressor(
                model=self.model,
                model_path=self.model_path,
                device=self.device,
            )
        return self._model

    # -------------------------------------------------------------- inference
    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
    ) -> TimeSeriesDataFrame:
        """Predict the horizon rows in `test_tsdf` given history in `train_tsdf`.

        Returns a TimeSeriesDataFrame indexed like `test_tsdf` with a `target`
        column (median point forecast) and one column per quantile level named
        by its string level (e.g. "0.1", ..., "0.9").
        """
        # Apply the documented history cap here too, so direct predict() callers
        # get it (predict_df and the eval wrapper also cap; this is idempotent).
        if self.context_length and self.context_length > 0:
            train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)
        train_feat, test_feat = FeatureTransformer(self.features).transform(
            train_tsdf, test_tsdf, target_column=_TARGET
        )
        model = self._get_model()
        q_levels = self.quantiles
        q_names = [str(q) for q in q_levels]
        median_idx = q_levels.index(0.5) if 0.5 in q_levels else None

        # One groupby pass each instead of an xs() per item (O(n) vs O(n^2) on
        # the 300+-series datasets); droplevel mirrors what xs() returned.
        train_groups = {k: v.droplevel("item_id")
                        for k, v in train_feat.groupby(level="item_id")}
        test_groups = {k: v.droplevel("item_id")
                       for k, v in test_feat.groupby(level="item_id")}
        out_frames = []
        for item_id in test_feat.item_ids:
            tr = train_groups[item_id]
            te = test_groups[item_id]
            feat_cols = [c for c in tr.columns if c != _TARGET]

            X_tr = tr[feat_cols].to_numpy(np.float32)
            y_tr = tr[_TARGET].to_numpy(np.float64)
            X_te = te[feat_cols].to_numpy(np.float32)

            # (K, n_horizon) quantile forecasts from Nori's quantile head.
            q_pred = np.asarray(
                model.fit(X_tr, y_tr).predict(
                    X_te, output_type="quantiles", quantiles=q_levels
                ),
                dtype=np.float64,
            )
            # Redundant safety: predict() already returns monotone quantiles per
            # row (api.py sorts its inverse-CDF), and self.quantiles is ascending,
            # so this is a no-op guard rather than a real de-crossing step.
            q_pred = np.sort(q_pred, axis=0)

            point = q_pred[median_idx] if median_idx is not None else q_pred.mean(0)
            data = {_TARGET: point}
            for name, row in zip(q_names, q_pred):
                data[name] = row
            # Rebuild the (item_id, timestamp) MultiIndex that xs() dropped.
            idx = pd.MultiIndex.from_product(
                [[item_id], te.index], names=["item_id", "timestamp"]
            )
            out_frames.append(pd.DataFrame(data, index=idx))

        pred = pd.concat(out_frames)
        return TimeSeriesDataFrame(pred)

    # --------------------------------------------------------- convenience API
    def predict_df(
        self,
        context_df: pd.DataFrame,
        prediction_length: int,
    ) -> pd.DataFrame:
        """Forecast from a plain DataFrame.

        `context_df` needs columns `timestamp`, `target`, and optionally
        `item_id` (defaults to a single series with id 0). Returns a plain
        DataFrame indexed by (item_id, timestamp) with `target` + quantile cols.
        """
        df = context_df.copy()
        if "item_id" not in df.columns:
            df["item_id"] = 0
        train_tsdf = TimeSeriesDataFrame.from_data_frame(df)
        if self.context_length and self.context_length > 0:
            train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)
        test_tsdf = generate_test_X(train_tsdf, prediction_length=prediction_length)
        pred = self.predict(train_tsdf, test_tsdf)
        return pd.DataFrame(pred)
