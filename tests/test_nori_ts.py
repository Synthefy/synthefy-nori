"""Tests for nori_ts forecasting.

Offline unit tests exercise the feature engineering / horizon construction with
no checkpoint (they need the `timeseries` extra: gluonts, statsmodels, datasets).
The end-to-end forecast is marked `slow` — it downloads a checkpoint and runs
real inference.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gluonts")
pytest.importorskip("statsmodels")
pytest.importorskip("datasets")

from synthefy_nori.nori_ts import NoriTSForecaster
from synthefy_nori.nori_ts.core import _default_features, _TARGET
from synthefy_nori.nori_ts.tsfeatures import (
    FeatureTransformer,
    TimeSeriesDataFrame,
    generate_test_X,
)


def _tsdf(n=60, freq="h", item_id=0, start="2020-01-01", gappy=False):
    ts = pd.date_range(start, periods=n, freq=freq)
    target = np.arange(n, dtype=float)
    if gappy:  # drop an interior chunk -> irregular index (mimics NaN-drop)
        keep = np.ones(n, bool)
        keep[20:30] = False
        ts, target = ts[keep], target[keep]
    df = pd.DataFrame({"item_id": item_id, "timestamp": ts, "target": target})
    return TimeSeriesDataFrame.from_data_frame(df)


def test_quantiles_sorted_in_ctor():
    # Non-ascending input must be sorted so column labels stay aligned with the
    # value-sorted forecast rows.
    assert NoriTSForecaster(quantiles=[0.9, 0.1, 0.5]).quantiles == [0.1, 0.5, 0.9]


def test_generate_test_X_horizon():
    train = _tsdf(n=50, freq="h")
    test = generate_test_X(train, prediction_length=10, freq="h")
    assert len(test) == 10
    assert test[_TARGET].isna().all()  # horizon targets are unknown
    tstamps = train.index.get_level_values("timestamp")
    horizon = test.index.get_level_values("timestamp")
    assert horizon[0] == tstamps.max() + pd.Timedelta(hours=1)  # contiguous
    assert (horizon.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_generate_test_X_gappy_uses_explicit_freq():
    # A3 regression: a gappy (NaN-dropped) index makes freq re-inference return
    # None; passing freq= explicitly must still produce a valid horizon.
    train = _tsdf(n=50, freq="h", gappy=True)
    test = generate_test_X(train, prediction_length=5, freq="h")
    assert len(test) == 5
    assert test[_TARGET].isna().all()


def test_feature_columns_match_and_nan_split():
    train = _tsdf(n=60, freq="h")
    test = generate_test_X(train, prediction_length=12, freq="h")
    tr, te = FeatureTransformer(_default_features()).transform(
        train, test, target_column=_TARGET
    )
    tr_cols = sorted(c for c in tr.columns if c != _TARGET)
    te_cols = sorted(c for c in te.columns if c != _TARGET)
    assert tr_cols == te_cols  # identical feature schema across the boundary
    assert not tr[_TARGET].isna().any()  # train targets known
    assert te[_TARGET].isna().all()  # horizon targets masked (no leakage)


def test_running_index_contiguous_across_boundary():
    train = _tsdf(n=40, freq="h")
    test = generate_test_X(train, prediction_length=8, freq="h")
    tr, te = FeatureTransformer(_default_features()).transform(
        train, test, target_column=_TARGET
    )
    ri = np.concatenate([tr["running_index"].to_numpy(), te["running_index"].to_numpy()])
    assert (np.diff(ri) == 1).all()  # continues, doesn't restart at the horizon


@pytest.mark.slow
def test_predict_df_end_to_end():
    # Real checkpoint + the groupby predict path, on a synthetic daily-cycle series.
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n)
    series = 10 + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n)
    hist = pd.DataFrame(
        {"timestamp": pd.date_range("2021-01-01", periods=n, freq="h"), "target": series}
    )
    out = NoriTSForecaster(quantiles=[0.1, 0.5, 0.9]).predict_df(hist, prediction_length=24)
    assert len(out) == 24
    assert {"0.1", "0.5", "0.9"}.issubset(set(out.columns))
    assert np.isfinite(out["0.5"].to_numpy()).all()
    # quantiles are monotone (no crossing) at every horizon step
    q = out[["0.1", "0.5", "0.9"]].to_numpy()
    assert (np.diff(q, axis=1) >= -1e-6).all()
