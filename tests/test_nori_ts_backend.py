"""Explicit backend, model, and request ownership for NoriTSForecaster."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gluonts")
pytest.importorskip("statsmodels")
pytest.importorskip("datasets")

from synthefy.nori_ts import NoriTSForecaster
from synthefy.nori_ts.tsfeatures import TimeSeriesDataFrame, generate_test_X


class _RecordingClient:
    def __init__(self, *, mode="remote", model="synthefy/nori-30m"):
        self.mode = mode
        self.model = model
        self.calls = []

    def predict(self, X_train, y_train, X_test, **kwargs):
        self.calls.append(
            {
                "X_train": X_train.copy(),
                "y_train": y_train.copy(),
                "X_test": X_test.copy(),
                "kwargs": kwargs,
            }
        )
        return np.vstack([np.full(len(X_test), level * 10.0) for level in kwargs["quantiles"]])


def _multi_series_frame():
    frames = []
    for item in (0, 1):
        timestamps = pd.date_range("2021-01-01", periods=48, freq="h")
        target = item + np.sin(2 * np.pi * np.arange(48) / 24)
        frames.append(pd.DataFrame({"item_id": item, "timestamp": timestamps, "target": target}))
    return TimeSeriesDataFrame.from_data_frame(pd.concat(frames, ignore_index=True))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mode": "local"},
        {"model": "nori-30m"},
        {"mode": "local", "model": None},
    ],
)
def test_mode_and_model_have_no_defaults(kwargs):
    with pytest.raises(ValueError, match="required when client= is not provided"):
        NoriTSForecaster(**kwargs)


def test_auto_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        NoriTSForecaster(mode="auto", model="nori-30m")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"mode": "local", "model": "nori-6m"},
            {
                "api_key": None,
                "mode": "local",
                "model": "nori-6m",
                "endpoint_name": None,
                "region_name": None,
            },
        ),
        (
            {"mode": "remote", "model": "nori-30m-thinking-medium", "api_key": "key"},
            {
                "api_key": "key",
                "mode": "remote",
                "model": "nori-30m-thinking-medium",
                "endpoint_name": None,
                "region_name": None,
            },
        ),
        (
            {
                "mode": "sagemaker",
                "model": "nori-30m",
                "endpoint_name": "ep",
                "region_name": "us-east-1",
            },
            {
                "api_key": None,
                "mode": "sagemaker",
                "model": "nori-30m",
                "endpoint_name": "ep",
                "region_name": "us-east-1",
            },
        ),
    ],
)
def test_forecaster_constructs_the_client_with_exact_configuration(monkeypatch, kwargs, expected):
    class RecordingConstructor:
        def __init__(self, **received):
            self.received = received

    monkeypatch.setattr("synthefy.nori_ts.core.SynthefyNoriClient", RecordingConstructor)

    forecaster = NoriTSForecaster(**kwargs)

    assert isinstance(forecaster.client, RecordingConstructor)
    assert forecaster.client.received == expected


def test_injected_client_owns_backend_configuration():
    client = _RecordingClient(model="synthefy/nori-30m-thinking-medium")
    for kwargs in (
        {"mode": "remote"},
        {"model": "nori-30m"},
        {"api_key": "key"},
        {"endpoint_name": "ep"},
        {"region_name": "us-east-1"},
    ):
        with pytest.raises(ValueError, match="client= already owns"):
            NoriTSForecaster(client=client, **kwargs)


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (SimpleNamespace(mode="auto", model="nori-30m"), "explicit mode"),
        (SimpleNamespace(mode="remote", model=None), "explicit model"),
        (SimpleNamespace(mode="remote", model="nori-30m"), "callable predict"),
    ],
)
def test_injected_client_must_be_fully_configured(client, message):
    with pytest.raises(ValueError, match=message):
        NoriTSForecaster(client=client)


def test_injected_client_receives_prepared_requests_and_returns_forecasts():
    train = _multi_series_frame()
    test = generate_test_X(train, prediction_length=6, freq="h")
    client = _RecordingClient()

    result = NoriTSForecaster(
        client=client,
        quantiles=[0.9, 0.1, 0.5],
    ).predict(train, test)

    assert len(client.calls) == 2
    for call in client.calls:
        assert call["X_train"].shape[0] == 48
        assert call["X_test"].shape[0] == 6
        assert call["X_train"].shape[1] == call["X_test"].shape[1]
        assert call["X_train"].dtype == np.float32
        assert call["X_test"].dtype == np.float32
        assert call["y_train"].dtype == np.float64
        assert call["kwargs"] == {
            "output_type": "quantiles",
            "quantiles": [0.1, 0.5, 0.9],
        }

    assert list(result.columns) == ["target", "0.1", "0.5", "0.9"]
    np.testing.assert_allclose(result["target"], 5.0)
    np.testing.assert_allclose(result["0.1"], 1.0)
    np.testing.assert_allclose(result["0.5"], 5.0)
    np.testing.assert_allclose(result["0.9"], 9.0)


def test_predict_df_accepts_explicit_frequency_for_gappy_history():
    timestamps = pd.date_range("2021-01-01", periods=48, freq="h").delete(10)
    context = pd.DataFrame(
        {
            "item_id": 0,
            "timestamp": timestamps,
            "target": np.sin(2 * np.pi * np.arange(len(timestamps)) / 24),
        }
    )
    client = _RecordingClient()
    forecaster = NoriTSForecaster(client=client, quantiles=[0.5])

    with pytest.raises(ValueError, match="Pass freq= explicitly"):
        forecaster.predict_df(context, prediction_length=3)

    result = forecaster.predict_df(context, prediction_length=3, freq="h")

    assert len(client.calls) == 1
    assert list(result.index.get_level_values("timestamp")) == list(
        pd.date_range(timestamps[-1] + pd.Timedelta(hours=1), periods=3, freq="h")
    )
    np.testing.assert_allclose(result["target"], 5.0)
