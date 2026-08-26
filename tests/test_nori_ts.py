"""Tests for nori_ts forecasting.

Offline unit tests exercise the feature engineering / horizon construction with
no checkpoint (they need the `forecasting` extra: gluonts, statsmodels, datasets).
The end-to-end forecast is marked `slow` — it downloads a checkpoint and runs
real inference.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gluonts")
pytest.importorskip("statsmodels")
pytest.importorskip("datasets")

from synthefy.nori_ts import NoriTSForecaster
from synthefy.nori_ts.core import _TARGET, _default_features
from synthefy.nori_ts.tsfeatures import (
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
    assert NoriTSForecaster(mode="local", model="nori-30m", quantiles=[0.9, 0.1, 0.5]).quantiles == [0.1, 0.5, 0.9]


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
    tr, te = FeatureTransformer(_default_features()).transform(train, test, target_column=_TARGET)
    tr_cols = sorted(c for c in tr.columns if c != _TARGET)
    te_cols = sorted(c for c in te.columns if c != _TARGET)
    assert tr_cols == te_cols  # identical feature schema across the boundary
    assert not tr[_TARGET].isna().any()  # train targets known
    assert te[_TARGET].isna().all()  # horizon targets masked (no leakage)


def test_running_index_contiguous_across_boundary():
    train = _tsdf(n=40, freq="h")
    test = generate_test_X(train, prediction_length=8, freq="h")
    tr, te = FeatureTransformer(_default_features()).transform(train, test, target_column=_TARGET)
    ri = np.concatenate([tr["running_index"].to_numpy(), te["running_index"].to_numpy()])
    assert (np.diff(ri) == 1).all()  # continues, doesn't restart at the horizon


def test_predict_rejects_horizon_series_without_history():
    # A series present in the horizon but absent from history used to KeyError deep
    # in the groupby lookup; it must fail with a clear message instead.
    train = _tsdf(n=30, freq="h", item_id=0)
    test = generate_test_X(train, prediction_length=5, freq="h")
    orphan = generate_test_X(_tsdf(n=30, freq="h", item_id=1), prediction_length=5, freq="h")
    combined = TimeSeriesDataFrame(pd.concat([pd.DataFrame(test), pd.DataFrame(orphan)]))
    with pytest.raises(ValueError, match="no history rows"):
        NoriTSForecaster(mode="local", model="nori-30m").predict(train, combined)


def test_generators_group_per_series_on_a_multi_series_frame():
    # Post-#144 the generator contract is whole-frame: generate() receives every
    # series at once and must group on item_id itself. A generator that indexed the
    # frame globally would give series 1 a running_index continuing from series 0,
    # and seasonal phase computed off the wrong offset — silently, with no error.
    frames = []
    for item in (0, 1, 2):
        ts = pd.date_range("2021-01-01", periods=48, freq="h")
        t = np.arange(48)
        frames.append(
            pd.DataFrame(
                {
                    "item_id": item,
                    "timestamp": ts,
                    "target": 10.0 + item + np.sin(2 * np.pi * t / 24),
                }
            )
        )
    train = TimeSeriesDataFrame.from_data_frame(pd.concat(frames, ignore_index=True))
    test = generate_test_X(train, prediction_length=6, freq="h")
    tr, te = FeatureTransformer(_default_features()).transform(train, test, target_column=_TARGET)
    for item in (0, 1, 2):
        ri = tr.xs(item, level="item_id")["running_index"].to_numpy()
        assert ri[0] == 0, f"series {item} running_index must restart at 0, got {ri[0]}"
        assert (np.diff(ri) == 1).all()
        # horizon continues that series' own counter, not the frame's
        ri_te = te.xs(item, level="item_id")["running_index"].to_numpy()
        assert ri_te[0] == ri[-1] + 1


def test_generated_feature_columns_are_float32():
    # #144 downcasts generated columns to float32 to halve the featurized frame that
    # is held in host RAM. core.py casts to float32 anyway, so this is lossless for
    # inference — but the target must keep its own dtype.
    train = _tsdf(n=48, freq="h")
    test = generate_test_X(train, prediction_length=6, freq="h")
    tr, _ = FeatureTransformer(_default_features()).transform(train, test, target_column=_TARGET)
    generated = [c for c in tr.columns if c != _TARGET]
    assert generated, "expected generated feature columns"
    # Only float64 columns are downcast; integer features (running_index, year) keep
    # their own dtype, and nothing generated should still be float64.
    floats = [c for c in generated if tr[c].dtype.kind == "f"]
    assert floats, "expected float feature columns (calendar/seasonal sin-cos)"
    assert all(tr[c].dtype == np.float32 for c in floats), {
        c: str(tr[c].dtype) for c in floats if tr[c].dtype != np.float32
    }
    assert all(tr[c].dtype.kind in "iuf" for c in generated)
    assert tr[_TARGET].dtype == np.float64


def test_feature_transformer_static_features_merge():
    # _merge_static_features must cover every item_id in train OR horizon; a missing
    # row would drop static columns for that series.
    train = _tsdf(n=30, freq="h")
    test = generate_test_X(train, prediction_length=6, freq="h")
    train.static_features = pd.DataFrame({"cat": [7]}, index=pd.Index([0], name="item_id"))
    tr, te = FeatureTransformer(_default_features()).transform(train, test, target_column=_TARGET)
    assert tr.static_features is not None
    assert set(tr.static_features.index) == {0}
    assert te.static_features.loc[0, "cat"] == 7


@pytest.mark.slow
def test_predict_df_end_to_end():
    # Real checkpoint + the groupby predict path, on a synthetic daily-cycle series.
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n)
    series = 10 + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n)
    hist = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=n, freq="h"), "target": series})
    out = NoriTSForecaster(mode="local", model="nori-6m", quantiles=[0.1, 0.5, 0.9]).predict_df(
        hist, prediction_length=24
    )
    assert len(out) == 24
    assert {"0.1", "0.5", "0.9"}.issubset(set(out.columns))
    assert np.isfinite(out["0.5"].to_numpy()).all()
    # quantiles are monotone (no crossing) at every horizon step
    q = out[["0.1", "0.5", "0.9"]].to_numpy()
    assert (np.diff(q, axis=1) >= -1e-6).all()


@pytest.mark.slow
def test_predict_df_end_to_end_custom_target_multiseries():
    rng = np.random.default_rng(1)
    n = 200
    frames = []
    for item in (0, 1):
        t = np.arange(n)
        series = 5 + item + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.2, n)
        frames.append(
            pd.DataFrame(
                {
                    "item_id": item,
                    "timestamp": pd.date_range("2021-01-01", periods=n, freq="h"),
                    "sales": series,
                }
            )
        )
    history = pd.concat(frames, ignore_index=True)

    output = NoriTSForecaster(mode="local", model="nori-6m", quantiles=[0.1, 0.5, 0.9]).predict_df(
        history, prediction_length=12, target_column="sales"
    )

    assert set(output.index.get_level_values("item_id")) == {0, 1}
    assert len(output) == 24
    assert "sales" in output.columns
    assert np.isfinite(output["sales"].to_numpy()).all()


@pytest.mark.slow
def test_predict_df_end_to_end_future_known_covariate():
    rng = np.random.default_rng(2)
    n, horizon = 240, 24
    t = np.arange(n)
    temperature = 15 + 10 * np.sin(2 * np.pi * t / 24)
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=n, freq="h"),
            "target": 2 * temperature + rng.normal(0, 0.5, n),
            "temperature": temperature,
        }
    )
    future_t = np.arange(n, n + horizon)
    future = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-11", periods=horizon, freq="h"),
            "temperature": 15 + 10 * np.sin(2 * np.pi * future_t / 24),
        }
    )

    output = NoriTSForecaster(mode="local", model="nori-6m", quantiles=[0.1, 0.5, 0.9]).predict_df(
        history, future_df=future
    )

    assert len(output) == horizon
    assert np.isfinite(output["0.5"].to_numpy()).all()
