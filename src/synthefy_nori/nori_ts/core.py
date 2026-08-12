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

Why this needs no change to Nori's attention
--------------------------------------------
Nori's row axis is a pure exchangeable set: `call_sequence_attention`
(`model/layer.py`) applies no positional encoding, no causal mask and no ordering
signal, and the shipped checkpoint runs `feature_positional_embedding_type=none`.
Rows are unordered as far as the model is concerned.

Forecasting still works, because time enters as *feature values* — running index,
calendar sin/cos, FFT-detected periods — rather than as attention geometry. The
model never needs to know that row t precedes row t+1; it reads position off the
columns. That is the TabPFN-TS thesis, and it reproduces on Nori: competitive
GIFT-eval numbers from an exchangeable set transformer with no temporal inductive
bias in the architecture at all.

The practical consequence: the exchangeable set transformer is *not* the
bottleneck for time series, so do not add row-positional encoding (RoPE or
otherwise) to chase forecasting quality. If these numbers need to improve, the
headroom is in the data, not the attention — the synthetic training prior has no
temporal structure (nothing makes row t depend on row t-1), so this checkpoint
has never seen a time series during pretraining.

Each series remains a separate one-shot regression request: one context and one
horizon per `item_id`, with no cross-series pooling, faithful to TabPFN-TS. An
injected Synthefy client executes remote or SageMaker requests. Until the
forecaster moves into the lightweight package, its legacy local path remains but
requires an explicit model selector or checkpoint; it never chooses a default
model. Synthefy's `nori-demand-forecasting` skill takes the opposite bet (pool the
whole panel into one table, lag features) and finds that pooling is the bigger win
on short-history and cold-start series. Neither has been run against the other;
see the pooled-vs-local follow-up issue.
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
_REQUIRED = object()
_SUPPORTED_MODES = frozenset({"local", "remote", "sagemaker"})


def _default_features():
    # Fresh instances (AutoSeasonalFeature holds config; cheap to rebuild).
    return [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]


class NoriTSForecaster:
    """Forecast univariate series by tabular regression with Nori.

    Parameters
    ----------
    device : str | None
        Torch device for the transitional local path; None auto-selects.
    context_length : int
        Cap on history rows per series (last-N kept), matching TabPFN-TS's 4096.
    quantiles : list[float]
        Forecast quantile levels (also used to derive the point/median forecast).
    features : list | None
        Feature generators; defaults to the TabPFN-TS default set.
    model : str
        Explicit Nori selector for the transitional local path. Required unless
        ``model_path`` or ``client`` is provided; there is no default model.
    model_path : str | None
        Explicit local checkpoint. Satisfies the local model requirement and
        overrides ``model`` when both are supplied.
    client : object | None
        A fully configured ``SynthefyNoriClient``-compatible client. It must
        expose explicit ``mode`` and ``model`` attributes and is mutually
        exclusive with ``device``, ``model``, and ``model_path``.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        context_length: int = 4096,
        quantiles: Optional[List[float]] = None,
        features=None,
        model=_REQUIRED,
        model_path: Optional[str] = None,
        client=None,
    ):
        if client is not None:
            conflicting = []
            if device is not None:
                conflicting.append("device")
            if model is not _REQUIRED:
                conflicting.append("model")
            if model_path is not None:
                conflicting.append("model_path")
            if conflicting:
                names = ", ".join(f"{name}=" for name in conflicting)
                raise ValueError(
                    "client= already owns backend configuration; do not also pass "
                    f"{names}"
                )
            if getattr(client, "mode", None) not in _SUPPORTED_MODES:
                raise ValueError(
                    "client= must carry an explicit mode: 'local', 'remote', or "
                    "'sagemaker'"
                )
            if getattr(client, "model", None) is None:
                raise ValueError("client= must carry an explicit model")
            resolved_model = None
        else:
            if (model is _REQUIRED or model is None) and model_path is None:
                raise ValueError(
                    "model= or model_path= is required when client= is not provided; "
                    "there is no default model"
                )
            resolved_model = None if model is _REQUIRED else model

        self.device = device
        self.context_length = context_length
        # Sorted ascending so the quantile column labels (q_names) stay aligned
        # with the value-sorted forecast rows — otherwise a caller passing e.g.
        # [0.9, 0.5, 0.1] would get columns labelled in input order but ordered
        # by value.
        self.quantiles = sorted(list(quantiles) if quantiles is not None else DEFAULT_QUANTILES)
        self.features = features if features is not None else _default_features()
        self.model = resolved_model
        self.model_path = model_path
        self.client = client
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

    def _predict_quantiles(self, X_train, y_train, X_test, quantiles):
        if self.client is not None:
            return self.client.predict(
                X_train,
                y_train,
                X_test,
                output_type="quantiles",
                quantiles=quantiles,
            )
        return self._get_model().fit(X_train, y_train).predict(
            X_test,
            output_type="quantiles",
            quantiles=quantiles,
        )

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
        # Check this before feature engineering: an orphan horizon series has an
        # all-NaN target, which blows up inside AutoSeasonalFeature's detrend with a
        # far less legible error than the one below.
        history_ids = set(train_tsdf.item_ids)
        missing_history = [i for i in test_tsdf.item_ids if i not in history_ids]
        if missing_history:
            raise ValueError(
                f"{len(missing_history)} series appear in the horizon but have no "
                f"history rows (ids {missing_history[:5]}…); cannot forecast them"
            )
        # Apply the documented history cap here too, so direct predict() callers
        # get it (predict_df and the eval wrapper also cap; this is idempotent).
        if self.context_length and self.context_length > 0:
            train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)
        train_feat, test_feat = FeatureTransformer(self.features).transform(
            train_tsdf, test_tsdf, target_column=_TARGET
        )
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
                self._predict_quantiles(X_tr, y_tr, X_te, q_levels),
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
