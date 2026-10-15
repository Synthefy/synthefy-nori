"""Offline backend-parity contract for the shared Nori forecasting workflow."""

from __future__ import annotations

import json
import struct
from pathlib import Path

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


def _quantile_rows(n_query: int, levels: list[float]) -> list[list[float]]:
    # A skewed distribution: averaging interval endpoints is not its median.
    return [[100.0 * level**2 + row for level in levels] for row in range(n_query)]


def _hosted_response(payload: dict, *, model: str | None = None) -> dict:
    rows = _quantile_rows(len(payload["X_test"]), payload["quantiles"])
    response = {
        "task": "regression",
        "predictions": [25.0 + row for row in range(len(rows))],
        "output_type": "quantiles",
        "quantiles": rows,
        "taus": payload["quantiles"],
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
        return np.asarray(_quantile_rows(len(X_test), kwargs["quantiles"]), dtype=float).T


class _MissingS3Object(Exception):
    response = {"Error": {"Code": "404"}}


def _read_tensor_request(path: str) -> dict:
    with Path(path).open("rb") as source:
        magic, header_size = struct.unpack("<8sI", source.read(12))
        assert magic == b"NORI_REQ"
        header = json.loads(source.read(header_size))
    payload = dict(header["metadata"])
    for name, descriptor in header["tensors"].items():
        payload[name] = np.memmap(
            path,
            dtype=descriptor["dtype"],
            mode="r",
            offset=descriptor["offset"],
            shape=tuple(descriptor["shape"]),
        ).tolist()
    return payload


@pytest.mark.parametrize("levels", [_LEVELS, [0.1, 0.9], [0.9], [0.9, 0.1, 0.1]])
def test_forecasting_preparation_and_reconstruction_match_every_backend(monkeypatch, levels):
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
            return np.asarray(_quantile_rows(len(X_test), quantiles), dtype=float).T

    monkeypatch.setattr(client_module, "_load_local_regressor", lambda: LocalRegressor)

    def remote_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests["remote"].append(payload)
        remote_headers.append(dict(request.headers))
        return httpx.Response(200, json=_hosted_response(payload))

    class SageMakerS3:
        def __init__(self):
            self.closed = False
            self.objects = {}
            self.last_payload = None

        def upload_file(self, path, bucket, key):
            self.last_payload = _read_tensor_request(path)
            requests["sagemaker"].append(self.last_payload)
            self.objects[(bucket, key)] = Path(path).read_bytes()

        def head_object(self, *, Bucket, Key):
            if (Bucket, Key) not in self.objects:
                raise _MissingS3Object()
            return {"ContentLength": len(self.objects[(Bucket, Key)])}

        def download_file(self, bucket, key, path):
            Path(path).write_bytes(self.objects[(bucket, key)])

        def delete_object(self, *, Bucket, Key):
            self.objects.pop((Bucket, Key), None)

        def close(self):
            self.closed = True

    class SageMakerRuntime:
        def __init__(self, s3):
            self.s3 = s3
            self.closed = False

        def invoke_endpoint_async(self, **kwargs):
            payload = self.s3.last_payload
            sagemaker_invocations.append(kwargs)
            response = _hosted_response(payload, model=payload["model"])
            invocation_id = kwargs["InferenceId"]
            output = f"s3://results/invocations/{invocation_id}/result.json.out"
            failure = f"s3://results/invocations/{invocation_id}/failure.json.out"
            self.s3.objects[("results", output.split("results/", 1)[1])] = json.dumps(response).encode()
            return {
                "InferenceId": invocation_id,
                "OutputLocation": output,
                "FailureLocation": failure,
            }

        def close(self):
            self.closed = True

    s3 = SageMakerS3()
    runtime = SageMakerRuntime(s3)
    monkeypatch.setattr(
        client_module,
        "create_aws_clients",
        lambda _service_names, **_kwargs: (runtime, s3),
    )

    fake = NoriTSForecaster(
        client=_RecordingFakeClient(requests["fake"]),
        quantiles=list(reversed(levels)),
    )
    local = NoriTSForecaster(
        mode="local",
        model="nori-30m",
        quantiles=list(reversed(levels)),
    )
    remote = NoriTSForecaster(
        mode="remote",
        model="nori-30m",
        api_key="test-key",
        quantiles=list(reversed(levels)),
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
        s3_staging_uri="s3://customer-bucket/nori/parity",
        quantiles=list(reversed(levels)),
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
    expected_levels = sorted(levels if 0.5 in levels else levels + [0.5])
    for backend, captured in requests.items():
        assert [_numeric_contract(payload) for payload in captured] == expected_contracts, backend

    expected = pd.DataFrame(results["fake"])
    assert list(expected.columns) == ["target"] + [str(level) for level in sorted(set(levels))]
    for backend, result in results.items():
        pd.testing.assert_frame_equal(pd.DataFrame(result), expected, obj=backend)
        for item_id in (0, 1):
            np.testing.assert_allclose(
                result.xs(item_id, level="item_id")["target"],
                np.arange(6, dtype=float) + 25.0,
            )

        for level in sorted(set(levels)):
            np.testing.assert_allclose(
                result.xs(0, level="item_id")[str(level)],
                np.arange(6, dtype=float) + 100.0 * level**2,
            )
        assert forecasters[backend].quantiles == sorted(levels)

    assert all(payload["quantiles"] == expected_levels for payload in expected_contracts)
    assert local_models == ["nori-30m"]
    assert all(payload["model"] == "synthefy/nori-30m" for payload in requests["remote"])
    assert all(header["authorization"] == "Bearer test-key" for header in remote_headers)
    assert all(payload["model"] == "nori-30m" for payload in requests["sagemaker"])
    assert all(invocation["EndpointName"] == "nori-parity" for invocation in sagemaker_invocations)
    assert all(
        invocation["ContentType"] == "application/vnd.synthefy.nori.tensorfile"
        and invocation["InputLocation"].startswith("s3://customer-bucket/nori/parity/")
        and "Body" not in invocation
        for invocation in sagemaker_invocations
    )
    assert runtime.closed is True
    assert s3.closed is True
