"""Model-free contract tests for canonical time-series feature preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("datasets")
pytest.importorskip("gluonts")
pytest.importorskip("joblib")
pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

from synthefy.nori_ts.tsfeatures import (
    AutoSeasonalFeature,
    CalendarFeature,
    FeatureTransformer,
    RunningIndexFeature,
    TimeSeriesDataFrame,
    generate_test_X,
)


def _tsdf(*, n: int = 48, item_id: int = 0) -> TimeSeriesDataFrame:
    frame = pd.DataFrame(
        {
            "item_id": item_id,
            "timestamp": pd.date_range("2021-01-01", periods=n, freq="h"),
            "target": np.arange(n, dtype=float),
        }
    )
    return TimeSeriesDataFrame.from_data_frame(frame)


def _features():
    return [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]


def test_generate_test_x_builds_a_contiguous_unknown_horizon():
    train = _tsdf()
    test = generate_test_X(train, prediction_length=6, freq="h")

    assert len(test) == 6
    assert test["target"].isna().all()
    timestamps = test.index.get_level_values("timestamp")
    assert timestamps[0] == train.index.get_level_values("timestamp").max() + pd.Timedelta(
        hours=1
    )


def test_generate_test_x_uses_explicit_frequency_for_gappy_input():
    train = TimeSeriesDataFrame.from_data_frame(
        pd.DataFrame(
            {
                "item_id": [0, 0],
                "timestamp": pd.to_datetime(["2021-01-01 00:00", "2021-01-01 03:00"]),
                "target": [1.0, 2.0],
            }
        )
    )

    test = generate_test_X(train, prediction_length=2, freq="h")

    assert test.index.get_level_values("timestamp").tolist() == list(
        pd.date_range("2021-01-01 04:00", periods=2, freq="h")
    )


def test_canonical_transform_preserves_schema_and_boundary_state():
    train = _tsdf()
    test = generate_test_X(train, prediction_length=6, freq="h")

    transformed_train, transformed_test = FeatureTransformer(_features()).transform(
        train,
        test,
        target_column="target",
    )

    assert list(transformed_train.columns) == list(transformed_test.columns)
    assert not transformed_train["target"].isna().any()
    assert transformed_test["target"].isna().all()
    running = np.concatenate(
        [
            transformed_train["running_index"].to_numpy(),
            transformed_test["running_index"].to_numpy(),
        ]
    )
    assert (np.diff(running) == 1).all()


def test_generated_float_columns_are_float32_without_downcasting_target():
    train = _tsdf()
    test = generate_test_X(train, prediction_length=6, freq="h")

    transformed_train, transformed_test = FeatureTransformer(_features()).transform(
        train,
        test,
        target_column="target",
    )

    assert transformed_train["target"].dtype == np.dtype("float64")
    assert transformed_test["target"].dtype == np.dtype("float64")
    assert transformed_train["hour_of_day_sin"].dtype == np.dtype("float32")
    assert transformed_test["hour_of_day_sin"].dtype == np.dtype("float32")


def test_static_features_survive_train_and_horizon_transform():
    train = _tsdf()
    train.static_features = pd.DataFrame(
        {"segment": ["retail"]},
        index=pd.Index([0], name="item_id"),
    )
    test = generate_test_X(train, prediction_length=4, freq="h")

    transformed_train, transformed_test = FeatureTransformer(_features()).transform(
        train,
        test,
        target_column="target",
    )

    pd.testing.assert_frame_equal(transformed_train.static_features, train.static_features)
    pd.testing.assert_frame_equal(transformed_test.static_features, train.static_features)


def test_canonical_generators_restart_for_each_series():
    train = TimeSeriesDataFrame(
        pd.concat([pd.DataFrame(_tsdf(item_id=item)) for item in (0, 1)])
    )
    test = generate_test_X(train, prediction_length=4, freq="h")

    transformed_train, transformed_test = FeatureTransformer(_features()).transform(
        train,
        test,
        target_column="target",
    )

    for item in (0, 1):
        history = transformed_train.xs(item, level="item_id")["running_index"].to_numpy()
        horizon = transformed_test.xs(item, level="item_id")["running_index"].to_numpy()
        assert history[0] == 0
        assert horizon[0] == history[-1] + 1
