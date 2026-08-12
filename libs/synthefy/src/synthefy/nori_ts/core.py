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
horizon per `item_id`, with no cross-series pooling, faithful to TabPFN-TS. Every
request executes through ``SynthefyNoriClient``; the forecaster either constructs
the client from an explicit backend and model or accepts a fully configured client.
Synthefy's `nori-demand-forecasting` skill takes the opposite bet (pool the whole
panel into one table, lag features) and finds that pooling is the bigger win on
short-history and cold-start series. Neither has been run against the other; see
the pooled-vs-local follow-up issue.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from synthefy.nori_client import SynthefyNoriClient
from synthefy.nori_ts.tsfeatures import (
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
    mode : {"local", "remote", "sagemaker"}
        Explicit regression execution backend. Required unless ``client`` is
        provided. There is no automatic backend selection.
    model : str
        Explicit Nori model selector. Required unless ``client`` is provided;
        there is no default model or implicit base-model fallback.
    client : object | None
        A fully configured ``SynthefyNoriClient``-compatible client. It must
        expose explicit ``mode`` and ``model`` attributes. Mutually exclusive
        with the forecaster backend-construction arguments.
    api_key : str | None
        Hosted Nori API key. Used only while constructing the client.
    endpoint_name : str | None
        SageMaker endpoint name. Used only while constructing the client.
    region_name : str | None
        SageMaker region. Used only while constructing the client.
    context_length : int
        Cap on history rows per series (last-N kept), matching TabPFN-TS's 4096.
    quantiles : list[float]
        Forecast quantile levels (also used to derive the point/median forecast).
    features : list | None
        Feature generators; defaults to the TabPFN-TS default set.
    """

    def __init__(
        self,
        mode=_REQUIRED,
        model=_REQUIRED,
        *,
        client=None,
        api_key: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        region_name: Optional[str] = None,
        context_length: int = 4096,
        quantiles: Optional[List[float]] = None,
        features=None,
    ):
        if client is not None:
            conflicting = []
            if mode is not _REQUIRED:
                conflicting.append("mode")
            if model is not _REQUIRED:
                conflicting.append("model")
            if api_key is not None:
                conflicting.append("api_key")
            if endpoint_name is not None:
                conflicting.append("endpoint_name")
            if region_name is not None:
                conflicting.append("region_name")
            if conflicting:
                names = ", ".join(f"{name}=" for name in conflicting)
                raise ValueError(f"client= already owns backend configuration; do not also pass {names}")
            if getattr(client, "mode", None) not in _SUPPORTED_MODES:
                raise ValueError("client= must carry an explicit mode: 'local', 'remote', or 'sagemaker'")
            if getattr(client, "model", None) is None:
                raise ValueError("client= must carry an explicit model")
            if not callable(getattr(client, "predict", None)):
                raise ValueError("client= must expose a callable predict() method")
            resolved_client = client
        else:
            missing = []
            if mode is _REQUIRED or mode is None:
                missing.append("mode")
            if model is _REQUIRED or model is None:
                missing.append("model")
            if missing:
                names = " and ".join(f"{name}=" for name in missing)
                raise ValueError(
                    f"{names} required when client= is not provided; there are no backend or model defaults"
                )
            if mode not in _SUPPORTED_MODES:
                choices = ", ".join(sorted(_SUPPORTED_MODES))
                raise ValueError(f"mode must be one of: {choices}; got {mode!r}")
            resolved_client = SynthefyNoriClient(
                api_key=api_key,
                mode=mode,
                model=model,
                endpoint_name=endpoint_name,
                region_name=region_name,
            )

        self.client = resolved_client
        self.context_length = context_length
        # Sorted ascending so the quantile column labels (q_names) stay aligned
        # with the value-sorted forecast rows — otherwise a caller passing e.g.
        # [0.9, 0.5, 0.1] would get columns labelled in input order but ordered
        # by value.
        self.quantiles = sorted(list(quantiles) if quantiles is not None else DEFAULT_QUANTILES)
        self.features = features if features is not None else _default_features()

    def _predict_quantiles(self, X_train, y_train, X_test, quantiles):
        return self.client.predict(
            X_train,
            y_train,
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
        train_groups = {k: v.droplevel("item_id") for k, v in train_feat.groupby(level="item_id")}
        test_groups = {k: v.droplevel("item_id") for k, v in test_feat.groupby(level="item_id")}
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
            idx = pd.MultiIndex.from_product([[item_id], te.index], names=["item_id", "timestamp"])
            out_frames.append(pd.DataFrame(data, index=idx))

        pred = pd.concat(out_frames)
        return TimeSeriesDataFrame(pred)

    # --------------------------------------------------------- convenience API
    def predict_df(
        self,
        context_df: pd.DataFrame,
        prediction_length: int,
        *,
        freq=None,
    ) -> pd.DataFrame:
        """Forecast from a plain DataFrame.

        `context_df` needs columns `timestamp`, `target`, and optionally
        `item_id` (defaults to a single series with id 0). Returns a plain
        DataFrame indexed by (item_id, timestamp) with `target` + quantile cols.
        Pass ``freq=`` when the history is gappy and its cadence cannot be
        inferred reliably (for example, ``freq="h"`` for hourly data).
        """
        df = context_df.copy()
        if "item_id" not in df.columns:
            df["item_id"] = 0
        train_tsdf = TimeSeriesDataFrame.from_data_frame(df)
        if self.context_length and self.context_length > 0:
            train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)
        test_tsdf = generate_test_X(
            train_tsdf,
            prediction_length=prediction_length,
            freq=freq,
        )
        pred = self.predict(train_tsdf, test_tsdf)
        return pd.DataFrame(pred)
