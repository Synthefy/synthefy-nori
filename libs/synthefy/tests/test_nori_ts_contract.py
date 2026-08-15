"""Model-free contract tests for the canonical Nori forecasting facade."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("datasets")
pytest.importorskip("gluonts")
pytest.importorskip("joblib")
pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

from synthefy import SynthefyNoriClient
from synthefy.nori_ts import NoriTSForecaster
from synthefy.nori_ts.core import _TARGET
from synthefy.nori_ts.tsfeatures import RunningIndexFeature


class _FakeClient:
    mode = "remote"
    model = "nori-6m"

    def __init__(self):
        self.calls = []

    def predict(self, X_train, y_train, X_test, **kwargs):
        self.calls.append(
            {
                "X_train": np.asarray(X_train).copy(),
                "y_train": np.asarray(y_train).copy(),
                "X_test": np.asarray(X_test).copy(),
                **kwargs,
            }
        )
        return np.zeros((len(kwargs["quantiles"]), len(X_test)), dtype=float)


def _forecaster(**kwargs):
    return NoriTSForecaster(client=_FakeClient(), **kwargs)


def _history(
    n=40,
    freq="h",
    target_column="target",
    item_id=0,
    start="2021-01-01",
    **covariates,
):
    data = {
        "timestamp": pd.date_range(start, periods=n, freq=freq),
        target_column: np.arange(n, dtype=float),
    }
    if item_id is not None:
        data["item_id"] = item_id
    data.update({name: np.asarray(values, dtype=float) for name, values in covariates.items()})
    return pd.DataFrame(data)


def _future(n, freq="h", item_id=0, start=None, **covariates):
    data = {"timestamp": pd.date_range(start, periods=n, freq=freq)}
    if item_id is not None:
        data["item_id"] = item_id
    data.update({name: np.asarray(values, dtype=float) for name, values in covariates.items()})
    return pd.DataFrame(data)


def test_predict_df_requires_exactly_one_horizon_argument():
    forecaster = _forecaster()
    history = _history()

    with pytest.raises(ValueError, match="exactly one"):
        forecaster.predict_df(history)
    with pytest.raises(ValueError, match="exactly one"):
        forecaster.predict_df(
            history,
            prediction_length=5,
            future_df=_future(5, start="2021-01-02 16:00"),
        )


def test_predict_df_rejects_missing_timestamp_and_target():
    forecaster = _forecaster()

    with pytest.raises(ValueError, match="timestamp"):
        forecaster.predict_df(pd.DataFrame({"target": [1.0, 2.0]}), prediction_length=3)
    with pytest.raises(ValueError, match="target column 'sales' not found"):
        forecaster.predict_df(
            _history(), prediction_length=3, target_column="sales"
        )


def test_predict_df_rejects_multiple_target_columns_clearly():
    forecaster = _forecaster()
    history = _history(target_column="sales")
    history["units"] = np.arange(len(history), dtype=float)

    with pytest.raises(ValueError, match="multiple target columns are not supported yet"):
        forecaster.predict_df(
            history,
            prediction_length=3,
            target_column=["sales", "units"],
        )


def test_predict_df_rejects_non_positive_horizon():
    forecaster = _forecaster()

    for invalid in (0, -4):
        with pytest.raises(ValueError, match="positive integer"):
            forecaster.predict_df(_history(), prediction_length=invalid)


def test_prediction_length_drops_history_covariates_without_future_values():
    forecaster = _forecaster()
    history = _history(n=30, temperature=np.linspace(0, 1, 30))

    train, test = forecaster._build_forecast_frames(
        history,
        prediction_length=6,
        future_df=None,
        target_column="target",
    )

    assert list(train.columns) == [_TARGET]
    assert list(test.columns) == [_TARGET]
    assert test[_TARGET].isna().all()
    assert len(test) == 6


def test_custom_target_is_normalized_and_name_collisions_are_rejected():
    forecaster = _forecaster()
    history = _history(target_column="sales")

    train, _ = forecaster._build_forecast_frames(
        history,
        prediction_length=4,
        future_df=None,
        target_column="sales",
    )
    assert _TARGET in train.columns
    assert "sales" not in train.columns

    history["target"] = 0.0
    with pytest.raises(ValueError, match="collision"):
        forecaster._build_forecast_frames(
            history,
            prediction_length=4,
            future_df=None,
            target_column="sales",
        )


def test_future_covariates_flow_through_fake_backend_in_timestamp_order():
    n, horizon = 30, 6
    temperature = np.linspace(10, 20, n)
    future_temperature = np.linspace(20, 21, horizon)
    history = _history(
        n=n,
        target_column="sales",
        temperature=temperature,
    ).iloc[::-1]
    history["region"] = "north"
    future = _future(
        horizon,
        start="2021-01-02 06:00",
        temperature=future_temperature,
    ).iloc[::-1]
    forecaster = _forecaster(
        features=[RunningIndexFeature()],
        quantiles=[0.1, 0.5, 0.9],
    )

    output = forecaster.predict_df(
        history,
        future_df=future,
        target_column="sales",
    )

    call = forecaster.client.calls[0]
    np.testing.assert_allclose(call["y_train"], np.arange(n, dtype=float))
    np.testing.assert_allclose(call["X_train"][:, 0], temperature)
    np.testing.assert_allclose(call["X_test"][:, 0], future_temperature)
    np.testing.assert_array_equal(call["X_test"][:, 1], np.arange(n, n + horizon))
    assert list(output.index.get_level_values("timestamp")) == sorted(future["timestamp"])
    assert "sales" in output.columns
    assert _TARGET not in output.columns


def test_predict_df_runs_through_public_synthefy_nori_client(monkeypatch):
    n, horizon = 30, 6
    history = _history(
        n=n,
        target_column="sales",
        temperature=np.linspace(10, 20, n),
    )
    future = _future(
        horizon,
        start="2021-01-02 06:00",
        temperature=np.linspace(20, 21, horizon),
    )
    calls = []
    client = SynthefyNoriClient(
        api_key="test-key",
        mode="remote",
        model="nori-6m",
    )

    def predict(X_train, y_train, X_test, **kwargs):
        calls.append((X_train, y_train, X_test, kwargs))
        return np.vstack(
            [np.full(len(X_test), level, dtype=float) for level in kwargs["quantiles"]]
        )

    monkeypatch.setattr(client, "predict", predict)
    try:
        output = NoriTSForecaster(
            client=client,
            features=[RunningIndexFeature()],
            quantiles=[0.1, 0.5, 0.9],
        ).predict_df(
            history,
            future_df=future,
            target_column="sales",
        )
    finally:
        client.close()

    assert len(calls) == 1
    _, _, X_test, kwargs = calls[0]
    assert len(X_test) == horizon
    assert kwargs == {
        "output_type": "quantiles",
        "quantiles": [0.1, 0.5, 0.9],
    }
    assert list(output.columns) == ["sales", "0.1", "0.5", "0.9"]
    np.testing.assert_allclose(output["sales"], 0.5)


def test_future_target_values_are_rejected_as_leakage():
    future = _future(5, start="2021-01-02 16:00")
    future["target"] = 1.0

    with pytest.raises(ValueError, match="no leakage"):
        _forecaster().predict_df(_history(), future_df=future)


def test_future_item_ids_must_match_history():
    history = _history(n=20, item_id=0)
    future = _future(5, start="2021-01-01 20:00", item_id=1)

    with pytest.raises(ValueError, match="item_ids"):
        _forecaster().predict_df(history, future_df=future)


def test_history_covariate_must_be_numeric_and_present_in_future():
    history = _history(n=20, temperature=np.linspace(0, 1, 20))
    future = _future(5, start="2021-01-01 20:00")

    with pytest.raises(ValueError, match="missing from `future_df`"):
        _forecaster().predict_df(history, future_df=future)

    future["temperature"] = "unknown"
    with pytest.raises(ValueError, match="not numeric"):
        _forecaster().predict_df(history, future_df=future)


def test_future_rejects_duplicate_and_history_overlapping_timestamps():
    history = _history(n=20)
    duplicate = _future(3, start="2021-01-01 20:00")
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="at most one row"):
        _forecaster().predict_df(history, future_df=duplicate)

    overlapping = _future(3, start="2021-01-01 19:00")
    with pytest.raises(ValueError, match="later than"):
        _forecaster().predict_df(history, future_df=overlapping)
