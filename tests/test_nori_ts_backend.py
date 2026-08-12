"""Backend-neutral request and model ownership for the transitional forecaster."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gluonts")
pytest.importorskip("statsmodels")
pytest.importorskip("datasets")

from synthefy_nori.nori_ts import NoriTSForecaster
from synthefy_nori.nori_ts.tsfeatures import TimeSeriesDataFrame, generate_test_X


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
        return np.vstack(
            [np.full(len(X_test), level * 10.0) for level in kwargs["quantiles"]]
        )


class _RecordingEstimator:
    def __init__(self):
        self.calls = []
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        return self

    def predict(self, X_test, **kwargs):
        self.calls.append(
            {
                "X_train": self._X_train,
                "y_train": self._y_train,
                "X_test": X_test.copy(),
                "kwargs": kwargs,
            }
        )
        return np.vstack(
            [np.full(len(X_test), level * 10.0) for level in kwargs["quantiles"]]
        )


def _multi_series_frame():
    frames = []
    for item in (0, 1):
        timestamps = pd.date_range("2021-01-01", periods=48, freq="h")
        target = item + np.sin(2 * np.pi * np.arange(48) / 24)
        frames.append(
            pd.DataFrame(
                {"item_id": item, "timestamp": timestamps, "target": target}
            )
        )
    return TimeSeriesDataFrame.from_data_frame(pd.concat(frames, ignore_index=True))


@pytest.mark.parametrize("kwargs", [{}, {"model": None}])
def test_transitional_local_path_has_no_model_default(kwargs):
    with pytest.raises(ValueError, match="model= or model_path= is required"):
        NoriTSForecaster(**kwargs)


@pytest.mark.parametrize("model", ["nori-6m", "nori-30m"])
def test_transitional_local_path_preserves_explicit_model(model):
    forecaster = NoriTSForecaster(model=model)
    assert forecaster.model == model


def test_explicit_model_path_satisfies_local_model_requirement():
    forecaster = NoriTSForecaster(model_path="/tmp/custom-nori.pt")
    assert forecaster.model is None
    assert forecaster.model_path == "/tmp/custom-nori.pt"


def test_injected_client_owns_backend_configuration():
    client = _RecordingClient(model="synthefy/nori-30m-thinking-medium")
    for kwargs in (
        {"device": "cpu"},
        {"model": "nori-30m"},
        {"model_path": "/tmp/nori.pt"},
    ):
        with pytest.raises(ValueError, match="client= already owns"):
            NoriTSForecaster(client=client, **kwargs)


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (SimpleNamespace(mode="auto", model="nori-30m"), "explicit mode"),
        (SimpleNamespace(mode="remote", model=None), "explicit model"),
    ],
)
def test_injected_client_must_be_fully_configured(client, message):
    with pytest.raises(ValueError, match=message):
        NoriTSForecaster(client=client)


def test_injected_client_receives_same_prepared_requests_as_local_path():
    train = _multi_series_frame()
    test = generate_test_X(train, prediction_length=6, freq="h")
    client = _RecordingClient()

    result = NoriTSForecaster(
        client=client,
        quantiles=[0.9, 0.1, 0.5],
    ).predict(train, test)

    estimator = _RecordingEstimator()
    local_forecaster = NoriTSForecaster(
        model="nori-30m",
        quantiles=[0.9, 0.1, 0.5],
    )
    local_forecaster._model = estimator
    local_result = local_forecaster.predict(train, test)

    assert len(client.calls) == len(estimator.calls) == 2
    for call, local_call in zip(client.calls, estimator.calls):
        np.testing.assert_array_equal(call["X_train"], local_call["X_train"])
        np.testing.assert_array_equal(call["y_train"], local_call["y_train"])
        np.testing.assert_array_equal(call["X_test"], local_call["X_test"])
        assert call["kwargs"] == local_call["kwargs"]
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

    pd.testing.assert_frame_equal(pd.DataFrame(result), pd.DataFrame(local_result))
    assert list(result.columns) == ["target", "0.1", "0.5", "0.9"]
    np.testing.assert_allclose(result["target"], 5.0)
    np.testing.assert_allclose(result["0.1"], 1.0)
    np.testing.assert_allclose(result["0.5"], 5.0)
    np.testing.assert_allclose(result["0.9"], 9.0)
