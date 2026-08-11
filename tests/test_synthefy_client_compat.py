"""Freeze the imported Synthefy 6.3 Nori behavior during repository migration.

The fixture was captured once from the exact source blobs recorded in
``libs/synthefy/SOURCE_SNAPSHOT.json``. CI compares against it; CI never
regenerates it from the migrated implementation.
"""

import hashlib
import importlib.machinery
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

import synthefy.nori_client as nori_client_module
from synthefy import NoriPredictRequest, NoriPredictResponse, SynthefyNoriClient
from synthefy.api_client import BadRequestError

_ROOT = Path(__file__).resolve().parents[1]
_GOLDEN_PATH = _ROOT / "tests" / "compat" / "synthefy_6_3_nori_goldens.json"
_MANIFEST_PATH = _ROOT / "libs" / "synthefy" / "SOURCE_SNAPSHOT.json"
_GOLDEN_SHA256 = (
    "f357863508260242746021594d8616a8baecb6284736f54fac8f8e771e0e7903"
)
_ORACLE = {
    "repository": "Synthefy/synthefy",
    "commit": "9ecc3d2fad8e37e95869379cc05f328597e258f9",
    "version": "6.3.0",
    "nori_client_blob": "a6fb1d1d91677ff9aa8c99186088bf63fd24eea7",
    "data_models_blob": "725e79d92c73313da6ab1c4154b73701e61458f6",
}


def _normalized(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "<NaN>"
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, tuple):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, pd.Series):
        return {
            "kind": "Series",
            "name": value.name,
            "index": value.index.tolist(),
            "values": _normalized(value.tolist()),
        }
    if isinstance(value, pd.DataFrame):
        return {
            "kind": "DataFrame",
            "index": value.index.tolist(),
            "columns": value.columns.tolist(),
            "values": _normalized(value.values.tolist()),
        }
    return value


def _remote_call(response: dict, *args: Any, **kwargs: Any) -> dict:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(
            raw=request.content.decode(),
            body=json.loads(request.content),
            path=request.url.path,
            authorization=request.headers["authorization"],
        )
        return httpx.Response(
            200,
            json=response,
            headers={"x-request-id": "req-golden"},
        )

    client = SynthefyNoriClient(api_key="golden-key", model="nori-30m")
    client.close()
    client.client = httpx.Client(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
    )
    result = client.predict(*args, **kwargs)
    client.close()
    return {"transport": captured, "result": _normalized(result)}


def _public_trace(monkeypatch) -> dict:
    trace = {"oracle": dict(_ORACLE)}

    trace["remote_default"] = _remote_call(
        {"task": "regression", "predictions": [10.0, None]},
        [[0.0, 1.0], [1.0, 0.0]],
        [1.0, 2.0],
        [[2.0, 2.0], [3.0, float("nan")]],
    )
    trace["remote_dataframe"] = _remote_call(
        {"task": "regression", "predictions": [3.5, 4.5]},
        pd.DataFrame(
            {"number": [1.0, 2.0, 3.0], "color": ["red", "blue", None]}
        ),
        pd.Series([1.0, 2.0, 3.0], name="score"),
        pd.DataFrame(
            {"color": ["green", "red"], "number": [4.0, 5.0]},
            index=["row-a", "row-b"],
        ),
        as_pandas=True,
    )
    trace["remote_memory"] = _remote_call(
        {
            "task": "regression",
            "predictions": [2.5],
            "memory_report": {"rung": "resident_int8", "resident_gb": 0.25},
        },
        [[0.0], [1.0]],
        [0.0, 1.0],
        [[2.0]],
        memory_policy={"cache_dtype": "int8"},
    )
    trace["remote_quantiles"] = _remote_call(
        {
            "task": "regression",
            "predictions": [2.0, 3.0],
            "output_type": "quantiles",
            "quantiles": [[2.8, 1.2], [3.8, 2.2]],
            "taus": [0.9, 0.1],
        },
        [[0.0], [1.0]],
        [0.0, 1.0],
        [[2.0], [3.0]],
        output_type="quantiles",
        quantiles=[0.9, 0.1],
    )

    local_capture = {}
    fake_nori = types.ModuleType("synthefy_nori")
    fake_nori.__spec__ = importlib.machinery.ModuleSpec("synthefy_nori", loader=None)

    def fake_predict(
        X_train: Any,
        y_train: Any,
        X_test: Any,
        *,
        task: Any = None,
        model: Any = None,
        **kwargs: Any,
    ) -> list:
        local_capture.update(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            task=task,
            model=model,
            kwargs=kwargs,
        )
        return [7.0]

    fake_nori.predict = fake_predict
    monkeypatch.setitem(sys.modules, "synthefy_nori", fake_nori)
    local_client = SynthefyNoriClient(mode="local", model="nori-30m")
    local_result = local_client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]])
    trace["local_default"] = {
        "call": _normalized(local_capture),
        "result": _normalized(local_result),
    }

    aws_capture = {}

    class FakeStream:
        def __iter__(self):
            yield {
                "PayloadPart": {
                    "Bytes": json.dumps(
                        {
                            "task": "regression",
                            "model": "nori-30m",
                            "predictions": [8.0],
                        }
                    ).encode()
                }
            }

        def close(self) -> None:
            aws_capture["closed"] = True

    class FakeRuntime:
        def invoke_endpoint_with_response_stream(self, **kwargs: Any) -> dict:
            aws_capture["request"] = {**kwargs, "Body": kwargs["Body"].decode()}
            return {"Body": FakeStream()}

        def close(self) -> None:
            pass

    class FakeSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            aws_capture["session"] = {"args": list(args), "kwargs": kwargs}

        def client(self, service_name: str, **kwargs: Any) -> FakeRuntime:
            aws_capture["service"] = service_name
            return FakeRuntime()

    class FakeConfig:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    fake_boto3 = types.SimpleNamespace(Session=FakeSession)
    monkeypatch.setattr(
        nori_client_module,
        "_load_aws_sdk",
        lambda: (fake_boto3, FakeConfig),
    )
    aws_client = SynthefyNoriClient(
        mode="sagemaker",
        model="nori-30m",
        endpoint_name="golden-endpoint",
        region_name="us-east-1",
    )
    trace["sagemaker_default"] = {
        "result": _normalized(
            aws_client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]])
        ),
        "transport": _normalized(aws_capture),
    }

    def bad_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "golden invalid",
                    "code": "invalid_request",
                }
            },
            headers={"x-request-id": "req-error"},
        )

    error_client = SynthefyNoriClient(
        api_key="bad", model="nori-30m", max_retries=0
    )
    error_client.close()
    error_client.client = httpx.Client(
        base_url=error_client.base_url,
        transport=httpx.MockTransport(bad_handler),
    )
    try:
        error_client.predict([[0.0]], [0.0], [[1.0]])
    except BadRequestError as exc:
        trace["remote_error"] = {
            "class": type(exc).__name__,
            "message": str(exc),
            "status_code": exc.status_code,
            "request_id": exc.request_id,
            "error_code": exc.error_code,
            "response_body": exc.response_body,
        }

    trace["wire_models"] = {
        "default": NoriPredictRequest(
            X_train=[[1.0, None]],
            y_train=[2.0],
            X_test=[[3.0, 4.0]],
        ).to_wire(),
        "distribution": NoriPredictRequest(
            X_train=[[1.0]],
            y_train=[2.0],
            X_test=[[3.0]],
            output_type="quantiles",
            quantiles=[0.9, 0.1],
        ).to_wire(),
        "response": NoriPredictResponse(
            task="regression", predictions=[1.0, None]
        ).model_dump(),
    }
    return trace


def test_fixture_is_pinned_to_the_imported_client_snapshot():
    golden_bytes = _GOLDEN_PATH.read_bytes()
    assert hashlib.sha256(golden_bytes).hexdigest() == _GOLDEN_SHA256
    assert json.loads(golden_bytes)["oracle"] == _ORACLE

    manifest = json.loads(_MANIFEST_PATH.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["source"]["repository"] == _ORACLE["repository"]
    assert manifest["source"]["commit"] == _ORACLE["commit"]
    assert manifest["source"]["version"] == _ORACLE["version"]
    assert manifest["import"]["included_blobs"]["src/synthefy/nori_client.py"] == (
        _ORACLE["nori_client_blob"]
    )
    assert manifest["import"]["included_blobs"]["src/synthefy/data_models.py"] == (
        _ORACLE["data_models_blob"]
    )


def test_migrated_client_matches_the_frozen_6_3_behavior(monkeypatch):
    assert _public_trace(monkeypatch) == json.loads(_GOLDEN_PATH.read_text())
