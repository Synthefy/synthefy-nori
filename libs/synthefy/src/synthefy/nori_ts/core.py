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
        prediction_length: Optional[int] = None,
        future_df: Optional[pd.DataFrame] = None,
        target_column: str = _TARGET,
        *,
        freq=None,
    ) -> pd.DataFrame:
        """Forecast from plain DataFrames — the high-level, TabPFN-TS-style entry point.

        Pass **exactly one** of:

        * ``prediction_length`` — forecast this many steps past the end of history,
          extrapolating the calendar (no future covariates), or
        * ``future_df`` — an explicit horizon frame carrying the future timestamps
          (and any known-future covariates). Its target must be absent or all-NaN.

        Parameters
        ----------
        context_df : DataFrame
            History. Needs a ``timestamp`` column and the target column; ``item_id``
            is optional (defaults to a single series with id ``0``). Extra **numeric**
            columns are treated as covariates.
        prediction_length : int, optional
            Horizon length. Mutually exclusive with ``future_df``.
        future_df : DataFrame, optional
            Explicit horizon: one row per future ``(item_id, timestamp)``, plus any
            known-future covariates. Its ``item_id`` set must match history exactly,
            and every numeric covariate used in history must be present here. Mutually
            exclusive with ``prediction_length``. Rows are ordered by item and
            timestamp before feature generation, and every horizon timestamp must
            follow that item's history.
        target_column : str, default "target"
            Name of the single target column in ``context_df``. Multiple target
            columns are not supported. The selected column is normalised to
            ``"target"`` internally and restored under its original name in the
            output.
        freq : optional
            Explicit frequency used with ``prediction_length`` when the history cadence
            cannot be inferred (for example, ``freq="h"``).

        Returns
        -------
        DataFrame indexed by ``(item_id, timestamp)`` with the point forecast under
        ``target_column`` and one column per quantile level (``"0.1"`` … ``"0.9"``).
        """
        train_tsdf, test_tsdf = self._build_forecast_frames(
            context_df, prediction_length, future_df, target_column, freq=freq
        )
        pred = self.predict(train_tsdf, test_tsdf)
        out = pd.DataFrame(pred)
        if target_column != _TARGET:  # restore the caller's target name
            out = out.rename(columns={_TARGET: target_column})
        return out

    def _build_forecast_frames(
        self,
        context_df: pd.DataFrame,
        prediction_length: Optional[int],
        future_df: Optional[pd.DataFrame],
        target_column: str,
        *,
        freq=None,
    ):
        """Validate inputs and build (train_tsdf, test_tsdf) for :meth:`predict`.

        Kept separate from ``predict_df`` so the whole DataFrame contract — the
        one-of check, target normalisation, no-leakage guard and covariate
        selection — is unit-testable without a checkpoint. Returns two
        ``TimeSeriesDataFrame``s whose target is named ``"target"`` and which carry
        an identical set of used covariates.
        """
        if not isinstance(target_column, str):
            raise ValueError(
                "`target_column` must be one column name; multiple target columns "
                "are not supported yet"
            )
        if (prediction_length is None) == (future_df is None):
            raise ValueError(
                "provide exactly one of `prediction_length` or `future_df` "
                "(got both or neither)"
            )

        history = context_df.copy()
        if "timestamp" not in history.columns:
            raise ValueError("`context_df` must have a `timestamp` column")
        if target_column not in history.columns:
            raise ValueError(
                f"target column {target_column!r} not found in `context_df` "
                f"(columns: {list(history.columns)})"
            )
        if "item_id" not in history.columns:
            history["item_id"] = 0
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        history = history.sort_values(["item_id", "timestamp"], kind="stable")

        # Normalise the target name to the internal "target" the rest of the
        # pipeline expects. Guard the corner case where a *different* column is
        # already literally called "target" (it would be silently clobbered).
        if target_column != _TARGET:
            if _TARGET in history.columns:
                raise ValueError(
                    f"`context_df` already has a {_TARGET!r} column while "
                    f"target_column={target_column!r}; rename it to avoid a collision"
                )
            history = history.rename(columns={target_column: _TARGET})
        if history[_TARGET].isna().any():
            raise ValueError("the history target has missing values; fill or drop them first")

        reserved = {"item_id", "timestamp", _TARGET}
        # Covariates are the numeric non-reserved history columns; non-numeric extras
        # (labels, strings) are ignored rather than fed to the regressor.
        hist_cov = [c for c in history.columns
                    if c not in reserved and pd.api.types.is_numeric_dtype(history[c])]

        if future_df is not None:
            future = future_df.copy()
            if "timestamp" not in future.columns:
                raise ValueError("`future_df` must have a `timestamp` column")
            if len(future) == 0:
                raise ValueError("`future_df` is empty; nothing to forecast")
            if "item_id" not in future.columns:
                future["item_id"] = 0
            future["timestamp"] = pd.to_datetime(future["timestamp"])
            if future.duplicated(["item_id", "timestamp"]).any():
                raise ValueError(
                    "`future_df` must contain at most one row per "
                    "(`item_id`, `timestamp`)"
                )
            # No leakage: a target in the horizon must be absent or entirely NaN.
            for tcol in {target_column, _TARGET} & set(future.columns):
                if not future[tcol].isna().all():
                    raise ValueError(
                        f"`future_df` column {tcol!r} carries target values; the "
                        "horizon target must be absent or all-NaN (no leakage)"
                    )
            future = future.drop(columns=[c for c in {target_column, _TARGET} if c in future.columns])
            # item_ids must line up exactly with history.
            h_ids, f_ids = set(history["item_id"]), set(future["item_id"])
            if h_ids != f_ids:
                raise ValueError(
                    f"`future_df` item_ids {sorted(f_ids)} do not match "
                    f"history item_ids {sorted(h_ids)}"
                )
            history_end = history.groupby("item_id", sort=False)["timestamp"].max()
            future_start = future.groupby("item_id", sort=False)["timestamp"].min()
            overlapping = [
                item_id
                for item_id in history_end.index
                if future_start.loc[item_id] <= history_end.loc[item_id]
            ]
            if overlapping:
                raise ValueError(
                    "every `future_df` timestamp must be later than its item's "
                    f"history (violations for item_ids {overlapping[:5]})"
                )
            future = future.sort_values(["item_id", "timestamp"], kind="stable")
            # Every numeric covariate used in history must be supplied (and numeric)
            # for the whole horizon; use exactly that shared set.
            used_cov = []
            for c in hist_cov:
                if c not in future.columns:
                    raise ValueError(
                        f"covariate {c!r} is in history but missing from `future_df`; "
                        "known-future covariates must be supplied across the whole horizon"
                    )
                if not pd.api.types.is_numeric_dtype(future[c]):
                    raise ValueError(f"covariate {c!r} in `future_df` is not numeric")
                used_cov.append(c)
            future[_TARGET] = np.nan
            test_df = future[["item_id", "timestamp", _TARGET] + used_cov]
            test_tsdf = TimeSeriesDataFrame.from_data_frame(test_df)
        else:
            if not isinstance(prediction_length, (int, np.integer)) or prediction_length <= 0:
                raise ValueError(
                    f"`prediction_length` must be a positive integer, got {prediction_length!r}"
                )
            used_cov = []  # no way to know future covariates without an explicit horizon
            test_tsdf = None  # built from the (capped) history below

        # History carries only the target + the covariates we'll actually use.
        train_df = history[["item_id", "timestamp", _TARGET] + used_cov]
        train_tsdf = TimeSeriesDataFrame.from_data_frame(train_df)
        if self.context_length and self.context_length > 0:
            train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)
        if test_tsdf is None:
            test_tsdf = generate_test_X(train_tsdf, prediction_length=prediction_length, freq=freq)
        return train_tsdf, test_tsdf
