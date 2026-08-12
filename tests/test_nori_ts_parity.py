"""Offline backend-parity contract for the shared Nori forecasting workflow."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("datasets")
pytest.importorskip("gluonts")
pytest.importorskip("statsmodels")

from synthefy import nori_client as client_module
from synthefy.nori_ts import NoriTSForecaster
from synthefy.nori_ts.tsfeatures import TimeSeriesDataFrame, generate_test_X


_LEVELS = [0.1, 0.5, 0.9]
_CONTRACT_FIELDS = (
    "X_train",
    "y_train",
    "X_test",
    "task",
    "output_type",
    "quantiles",
)


def _multi_series_frame() -> TimeSeriesDataFrame:
    frames = []
    for item in (0, 1):
        timestamps = pd.date_range("2021-01-01", periods=48, freq="h")
        target = item + np.sin(2 * np.pi * np.arange(48) / 24)
        frames.append(pd.DataFrame({"item_id": item, "timestamp": timestamps, "target": target}))
    return TimeSeriesDataFrame.from_data_frame(pd.concat(frames, ignore_index=True))


def _quantile_rows(n_query: int) -> list[list[float]]:
    return [[1.0 + row, 5.0 + row, 9.0 + row] for row in range(n_query)]


def _hosted_response(payload: dict, *, model: str | None = None) -> dict:
    rows = _quantile_rows(len(payload["X_test"]))
    response = {
        "task": "regression",
        "predictions": [row[1] for row in rows],
        "output_type": "quantiles",
        "quantiles": rows,
        "taus": _LEVELS,
    }
    if model is not None:
        response["model"] = model
    return response


def _numeric_contract(payload: dict) -> dict:
    return {field: payload[field] for field in _CONTRACT_FIELDS}


class _RecordingFakeClient:
    mode = "remote"
    model = "synthefy/nori-30m"

    def __init__(self, requests: list[dict]):
        self.requests = requests

    def predict(self, X_train, y_train, X_test, **kwargs):
        payload = {
            "X_train": np.asarray(X_train).tolist(),
            "y_train": np.asarray(y_train).tolist(),
            "X_test": np.asarray(X_test).tolist(),
            "task": "regression",
            "output_type": kwargs["output_type"],
            "quantiles": kwargs["quantiles"],
        }
        self.requests.append(payload)
        return np.asarray(_quantile_rows(len(X_test)), dtype=float).T


class _EventStream:
    def __init__(self, body: bytes):
        self.body = body
        self.closed = False

    def __iter__(self):
        yield {"PayloadPart": {"Bytes": b" \n"}}
        yield {"PayloadPart": {"Bytes": self.body}}

    def close(self):
        self.closed = True


def test_forecasting_preparation_and_reconstruction_match_every_backend(monkeypatch):
    """Only transport details may differ across fake/local/Baseten/SageMaker."""
    requests: dict[str, list[dict]] = {
        "fake": [],
        "local": [],
        "remote": [],
        "sagemaker": [],
    }
    local_models = []
    remote_headers = []
    sagemaker_invocations = []

    class LocalRegressor:
        def __init__(self, model=None):
            local_models.append(model)
            self.X_train = None
            self.y_train = None

        def fit(self, X_train, y_train):
            self.X_train = X_train
            self.y_train = y_train
            return self

        def predict(self, X_test, *, output_type="mean", quantiles=None):
            requests["local"].append(
                {
                    "X_train": self.X_train,
                    "y_train": self.y_train,
                    "X_test": X_test,
                    "task": "regression",
                    "output_type": output_type,
                    "quantiles": quantiles,
                }
            )
            return np.asarray(_quantile_rows(len(X_test)), dtype=float).T

    monkeypatch.setattr(client_module, "_load_local_regressor", lambda: LocalRegressor)

    def remote_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests["remote"].append(payload)
        remote_headers.append(dict(request.headers))
        return httpx.Response(200, json=_hosted_response(payload))

    class SageMakerRuntime:
        def __init__(self):
            self.closed = False

        def invoke_endpoint_with_response_stream(self, **kwargs):
            payload = json.loads(kwargs["Body"])
            requests["sagemaker"].append(payload)
            sagemaker_invocations.append(kwargs)
            response = _hosted_response(payload, model=payload["model"])
            return {"Body": _EventStream(json.dumps(response).encode())}

        def close(self):
            self.closed = True

    runtime = SageMakerRuntime()
    monkeypatch.setattr(
        client_module,
        "_create_sagemaker_runtime_client",
        lambda **_kwargs: runtime,
    )

    fake = NoriTSForecaster(
        client=_RecordingFakeClient(requests["fake"]),
        quantiles=list(reversed(_LEVELS)),
    )
    local = NoriTSForecaster(
        mode="local",
        model="nori-30m",
        quantiles=list(reversed(_LEVELS)),
    )
    remote = NoriTSForecaster(
        mode="remote",
        model="nori-30m",
        api_key="test-key",
        quantiles=list(reversed(_LEVELS)),
    )
    remote.client.close()
    remote.client.client = httpx.Client(
        base_url=remote.client.base_url,
        transport=httpx.MockTransport(remote_handler),
    )
    sagemaker = NoriTSForecaster(
        mode="sagemaker",
        model="nori-30m",
        endpoint_name="nori-parity",
        region_name="us-east-1",
        quantiles=list(reversed(_LEVELS)),
    )

    train = _multi_series_frame()
    test = generate_test_X(train, prediction_length=6, freq="h")
    forecasters = {
        "fake": fake,
        "local": local,
        "remote": remote,
        "sagemaker": sagemaker,
    }
    results = {name: forecaster.predict(train, test) for name, forecaster in forecasters.items()}
    remote.client.close()
    sagemaker.client.close()

    expected_contracts = [_numeric_contract(payload) for payload in requests["fake"]]
    assert len(expected_contracts) == 2
    for backend, captured in requests.items():
        assert [_numeric_contract(payload) for payload in captured] == expected_contracts, backend

    expected = pd.DataFrame(results["fake"])
    assert list(expected.columns) == ["target", "0.1", "0.5", "0.9"]
    for backend, result in results.items():
        pd.testing.assert_frame_equal(pd.DataFrame(result), expected, obj=backend)
        for item_id in (0, 1):
            np.testing.assert_allclose(
                result.xs(item_id, level="item_id")["target"],
                np.arange(6, dtype=float) + 5.0,
            )

    assert local_models == ["nori-30m"]
    assert all(payload["model"] == "synthefy/nori-30m" for payload in requests["remote"])
    assert all(header["authorization"] == "Bearer test-key" for header in remote_headers)
    assert all(payload["model"] == "nori-30m" for payload in requests["sagemaker"])
    assert all(
        invocation["EndpointName"] == "nori-parity" and invocation["CustomAttributes"] == "synthefy-response-stream=v1"
        for invocation in sagemaker_invocations
    )
    assert runtime.closed is True
