"""Unit tests for the Synthefy Nori client.

Remote mode is tested against a mocked httpx transport so these tests never hit
the network. The real local-inference test is marked ``slow`` and skips unless
the optional ``synthefy-nori`` package is installed.
"""

import builtins
import importlib.util
import json
import math
import struct
import sys
import types
import warnings
from pathlib import Path
from typing import Optional, Callable, Dict, List

import boto3
import botocore.session
import httpx
import numpy as np
import pandas as pd
import pytest
from botocore.stub import Stubber
from botocore.validate import ParamValidator
from synthefy import SynthefyNoriClient
from synthefy import _aws as aws_module
from synthefy import nori_client as nori_client_module
from synthefy.errors import (
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    SageMakerInvocationError,
)
from synthefy.data_models import NoriPredictRequest, NoriPredictResponse
from synthefy.nori_client import (
    GATEWAY_ENDPOINT,
    NORI_VARIANTS,
    SAGEMAKER_MAX_INPUT_BYTES,
    SAGEMAKER_VARIANTS,
    _MODEL_NAMES,
    _build_nori_request,
    _is_thinking_model,
    _nullable_matrix,
    _resolve_local_variant,
    _resolve_text_device,
    _validate_multi_target_controls,
    _widen_text_columns,
)
from synthefy.nori_data_models import FINE_TUNED_MODEL_NAME_PATTERN

Handler = Callable[[httpx.Request], httpx.Response]


def _text_runtime_available() -> bool:
    if importlib.util.find_spec("sklearn") is None:
        return False
    return importlib.util.find_spec("synthefy.text_features") is not None


requires_text_runtime = pytest.mark.skipif(
    not _text_runtime_available(),
    reason="needs the synthefy text extra",
)


def test_nori_models_are_canonical_data_models_with_compatible_exports():
    from synthefy import NoriPredictRequest as PublicRequest
    from synthefy import NoriPredictResponse as PublicResponse
    from synthefy.nori_client import NoriPredictRequest as ClientRequest
    from synthefy.nori_client import NoriPredictResponse as ClientResponse

    assert PublicRequest is ClientRequest is NoriPredictRequest
    assert PublicResponse is ClientResponse is NoriPredictResponse


def test_the_response_keeps_the_model_prefix_legal():
    # `model_version` sits in the `model_` namespace pydantic protected by default until
    # 2.10, and this package still allows pydantic 2.0. Declaring the namespace empty is
    # what keeps the field from warning on import there, which a consumer running
    # warnings as errors would experience as a failed `import synthefy`.
    import warnings

    assert NoriPredictResponse.model_config.get("protected_namespaces") == ()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NoriPredictResponse.model_rebuild(force=True)


def _attach_mock(client: SynthefyNoriClient, handler: Handler) -> None:
    """Swap the client's httpx transport for an in-memory mock (no network)."""
    client.close()
    client.client = httpx.Client(
        base_url=client.base_url,
        transport=httpx.MockTransport(handler),
    )


def _ok_handler(predictions: List[Optional[float]], capture: Dict) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        capture["path"] = request.url.path
        capture["headers"] = request.headers
        capture["body"] = json.loads(request.content)
        return httpx.Response(200, json={"task": "regression", "predictions": predictions})

    return handler


# --------------------------------------------------------------------------- #
# AWS SageMaker deployment -- signed transport (botocore Stubber, no network)
# --------------------------------------------------------------------------- #


def test_aws_factory_uses_argument_free_boto3_session(monkeypatch):
    """The public transport must use boto3's chain, never accept/pass raw keys."""
    capture: Dict = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            capture["config"] = kwargs

    class FakeClient:
        def __init__(self, name):
            self.name = name

        def close(self):
            capture.setdefault("closed", []).append(self.name)

    class FakeSession:
        def __init__(self, *args, **kwargs):
            capture["session_args"] = args
            capture["session_kwargs"] = kwargs

        def client(self, service_name, **kwargs):
            capture.setdefault("clients", []).append((service_name, kwargs))
            return FakeClient(service_name)

    class FakeBoto3:
        Session = FakeSession

    monkeypatch.setattr(aws_module, "load_aws_sdk", lambda: (FakeBoto3, FakeConfig))

    client = SynthefyNoriClient(
        mode="sagemaker",
        model="nori-30m",
        endpoint_name="nori-dev-123",
        region_name="us-east-1",
        s3_staging_uri="s3://customer-bucket/nori/staging/",
        timeout=75.0,
        max_retries=3,
    )

    assert capture["session_args"] == ()
    assert capture["session_kwargs"] == {}
    assert [name for name, _ in capture["clients"]] == ["sagemaker-runtime", "s3"]
    assert all(kwargs["region_name"] == "us-east-1" for _, kwargs in capture["clients"])
    assert capture["config"]["read_timeout"] == 75.0
    assert capture["config"]["retries"]["total_max_attempts"] == 4
    client.close()
    assert capture["closed"] == ["sagemaker-runtime", "s3"]


def test_botocore_stubber_accepts_the_async_request_shape():
    """Keep the exact async operation and signed parameter names checked by botocore."""
    runtime = boto3.client(
        "sagemaker-runtime",
        region_name="us-east-1",
        aws_access_key_id="unit-test",
        aws_secret_access_key="unit-test",
    )
    request = {
        "EndpointName": "nori-dev-123",
        "InputLocation": "s3://customer-bucket/nori/staging/request.tensorfile",
        "ContentType": "application/vnd.synthefy.nori.tensorfile",
        "Accept": "application/json",
        "InferenceId": "invocation-123",
        "RequestTTLSeconds": 21_600,
        "InvocationTimeoutSeconds": 3_600,
        "S3OutputPathExtension": "invocations/invocation-123",
        "Filename": "result.json",
    }
    stubber = Stubber(runtime)
    stubber.add_response(
        "invoke_endpoint_async",
        {
            "InferenceId": "invocation-123",
            "OutputLocation": ("s3://customer-bucket/nori/output/invocations/invocation-123/result.json.out"),
            "FailureLocation": ("s3://customer-bucket/nori/failure/invocations/invocation-123/result.json.out"),
        },
        request,
    )

    with stubber:
        response = runtime.invoke_endpoint_async(**request)
        assert response["InferenceId"] == "invocation-123"
        stubber.assert_no_pending_responses()


class _MissingObject(Exception):
    response = {"Error": {"Code": "404"}}


class _FakeS3:
    def __init__(self):
        self.objects = {}
        self.uploads = []
        self.deletes = []
        self.fail_upload = None
        self.fail_download = None

    def upload_file(self, filename, bucket, key):
        if self.fail_upload:
            raise self.fail_upload
        with Path(filename).open("rb") as request:
            magic, header_size = struct.unpack("<8sI", request.read(12))
            assert magic == b"NORI_REQ"
            header = json.loads(request.read(header_size))
        self.uploads.append(
            {
                "bucket": bucket,
                "key": key,
                "metadata": header["metadata"],
                "dtypes": {name: descriptor["dtype"] for name, descriptor in header["tensors"].items()},
                "shapes": {name: descriptor["shape"] for name, descriptor in header["tensors"].items()},
            }
        )
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _MissingObject()
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def download_file(self, bucket, key, filename):
        if self.fail_download:
            raise self.fail_download
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def delete_object(self, *, Bucket, Key):
        self.deletes.append((Bucket, Key))
        self.objects.pop((Bucket, Key), None)

    def close(self):
        pass


class _FakeRuntime:
    def __init__(self, s3, *, result="success"):
        self.s3 = s3
        self.result = result
        self.requests = []
        self.failure = None
        self.failure_message = "container rejected the table"
        self.response_overrides = {}

    def invoke_endpoint_async(self, **request):
        if self.failure:
            raise self.failure
        self.requests.append(request)
        invocation_id = request["InferenceId"]
        output = f"s3://results/output/invocations/{invocation_id}/result.json.out"
        failure = f"s3://results/failure/invocations/{invocation_id}/result.json.out"
        if self.result == "success":
            model = self.s3.uploads[-1]["metadata"]["model"]
            response = {
                "task": "regression",
                "model": model,
                "predictions": [10.0, 20.0],
                **self.response_overrides,
            }
            self.s3.objects[("results", output.split("results/", 1)[1])] = json.dumps(response).encode()
        elif self.result == "failure":
            self.s3.objects[("results", failure.split("results/", 1)[1])] = json.dumps(
                {"ErrorMessage": self.failure_message}
            ).encode()
        return {
            "InferenceId": invocation_id,
            "OutputLocation": output,
            "FailureLocation": failure,
        }

    def invoke_endpoint_with_response_stream(self, **_request):
        raise AssertionError("the removed response-stream operation was called")

    def close(self):
        pass


def _stubbed_aws_client(monkeypatch, *, model="nori-30m", result="success", timeout=5):
    s3 = _FakeS3()
    runtime = _FakeRuntime(s3, result=result)
    monkeypatch.setattr(
        nori_client_module,
        "create_aws_clients",
        lambda _service_names, **_kwargs: (runtime, s3),
    )
    client = SynthefyNoriClient(
        mode="sagemaker",
        model=model,
        endpoint_name="nori-shared",
        region_name="us-east-1",
        s3_staging_uri="s3://customer-bucket/nori/staging/",
        timeout=timeout,
    )
    return client, runtime, s3


@pytest.mark.parametrize("model", SAGEMAKER_VARIANTS)
def test_aws_predict_uses_s3_and_invoke_endpoint_async_for_every_model(monkeypatch, model):
    """Every peer uses one binary S3 path; no model can fall back to streaming."""
    client, runtime, s3 = _stubbed_aws_client(monkeypatch, model=model)
    predictions = client.predict(
        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [1.0, 1.0, 2.0],
        [[2.0, 2.0], [3.0, 3.0]],
    )
    assert predictions == [10.0, 20.0]
    upload = s3.uploads[0]
    assert upload["bucket"] == "customer-bucket"
    assert upload["key"].startswith("nori/staging/nori-shared/")
    assert upload["metadata"]["model"] == model
    assert upload["shapes"] == {"X_test": [2, 2], "X_train": [3, 2], "y_train": [3]}
    assert upload["dtypes"] == {"X_test": "<f4", "X_train": "<f4", "y_train": "<f8"}
    request = runtime.requests[0]
    assert request["EndpointName"] == "nori-shared"
    assert request["InputLocation"] == f"s3://customer-bucket/{upload['key']}"
    assert request["ContentType"] == "application/vnd.synthefy.nori.tensorfile"
    assert request["RequestTTLSeconds"] == 21_600
    assert request["InvocationTimeoutSeconds"] == 3_600
    assert "Body" not in request and "CustomAttributes" not in request
    deleted_keys = {key for _, key in s3.deletes}
    assert upload["key"] in deleted_keys
    assert any(f"invocations/{request['InferenceId']}" in key for key in deleted_keys)


def test_sagemaker_large_context_uses_the_shared_wire_and_report(monkeypatch):
    report = {
        "applied": True,
        "policy": "cluster_route_g4",
        "threshold": 1,
        "seed": 11,
        "reason": None,
        "window": 2,
        "n_train": 3,
        "n_test": 2,
        "shards_available": 1,
        "nori_calls": 2,
        "full_context": False,
        "reused_train_state": False,
        "gate_winner": None,
    }
    client, runtime, s3 = _stubbed_aws_client(monkeypatch)
    runtime.response_overrides["large_context_report"] = report
    assert client.predict(
        _X_TRAIN,
        _Y_TRAIN,
        _X_TEST,
        large_context_policy="cluster_route_g4",
        large_context_threshold=1,
        large_context_seed=11,
    ) == [10.0, 20.0]
    metadata = s3.uploads[0]["metadata"]
    assert metadata["large_context_policy"] == "cluster_route_g4"
    assert metadata["large_context_threshold"] == 1
    assert metadata["large_context_seed"] == 11
    assert client.last_large_context_report == report


def test_sagemaker_array_request_keeps_compact_feature_arrays_without_a_copy():
    """A contiguous float32 table must not gain a payload-sized float64 twin."""
    X_train = np.arange(24, dtype=np.float32).reshape(6, 4)
    X_test = np.arange(8, dtype=np.float32).reshape(2, 4)
    y_train = np.arange(6, dtype=np.float64)
    request = _build_nori_request(
        X_train,
        y_train,
        X_test,
        array_backed=True,
    )
    assert np.shares_memory(request.X_train, X_train)
    assert np.shares_memory(request.X_test, X_test)
    assert np.shares_memory(request.y_train, y_train)


def test_sagemaker_invocation_ids_are_unique(monkeypatch):
    client, runtime, _ = _stubbed_aws_client(monkeypatch, model="nori-6m")
    runtime.response_overrides["predictions"] = [10.0]
    for _ in range(2):
        client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]])
    assert len({request["InferenceId"] for request in runtime.requests}) == 2


def test_sagemaker_response_model_mismatch_fails_closed(monkeypatch):
    client, runtime, _ = _stubbed_aws_client(monkeypatch, model="nori-30m")
    runtime.response_overrides["model"] = "nori-6m"

    with pytest.raises(ValueError, match="model identity mismatch"):
        client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]])


def test_sagemaker_failure_object_is_a_useful_client_exception(monkeypatch):
    client, runtime, s3 = _stubbed_aws_client(monkeypatch, result="failure")
    with pytest.raises(SageMakerInvocationError, match="container rejected") as caught:
        client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]])
    assert caught.value.invocation_id == runtime.requests[0]["InferenceId"]
    assert not s3.objects


def test_sagemaker_timeout_preserves_input_for_the_queued_invocation(monkeypatch):
    client, runtime, s3 = _stubbed_aws_client(monkeypatch, result="pending", timeout=1)
    unrelated = ("customer-bucket", "nori/staging/someone-else/request.tensorfile")
    s3.objects[unrelated] = b"keep"
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(nori_client_module.time, "monotonic", lambda: next(clock))
    with pytest.raises(APITimeoutError, match="does not cancel"):
        client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]])
    assert unrelated in s3.objects
    assert unrelated not in s3.deletes
    assert runtime.requests
    upload = s3.uploads[0]
    # Simulate the queued worker fetching its payload after the caller timed out.
    assert s3.head_object(Bucket=upload["bucket"], Key=upload["key"])["ContentLength"] > 0
    assert not s3.deletes


@pytest.mark.parametrize("failure_at", ["poll", "invoke-response"])
def test_sagemaker_uncertain_completion_preserves_staged_input(monkeypatch, failure_at):
    client, runtime, s3 = _stubbed_aws_client(monkeypatch, result="pending")

    def fail(*args, **kwargs):
        raise ConnectionError("connection lost")

    if failure_at == "poll":
        monkeypatch.setattr(s3, "head_object", fail)
    else:
        invoke = runtime.invoke_endpoint_async

        def accepted_but_response_lost(**request):
            invoke(**request)
            fail()

        monkeypatch.setattr(runtime, "invoke_endpoint_async", accepted_but_response_lost)
    with pytest.raises(ConnectionError, match="connection lost"):
        client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]])
    upload = s3.uploads[0]
    assert (upload["bucket"], upload["key"]) in s3.objects
    assert not s3.deletes


@pytest.mark.parametrize("failure_at", ["upload", "invoke", "download"])
def test_sagemaker_failures_cleanup_only_before_submission(monkeypatch, failure_at):
    client, runtime, s3 = _stubbed_aws_client(monkeypatch)
    if failure_at == "upload":
        s3.fail_upload = RuntimeError("upload failed")
    elif failure_at == "invoke":
        runtime.failure = RuntimeError("invoke failed")
    else:
        s3.fail_download = RuntimeError("download failed")
    with pytest.raises(RuntimeError, match=f"{failure_at} failed"):
        client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]])
    if failure_at == "upload":
        assert any(key.endswith("request.tensorfile") for _, key in s3.deletes)
    else:
        upload = s3.uploads[0]
        assert (upload["bucket"], upload["key"]) in s3.objects
        assert not s3.deletes


def test_sagemaker_never_deletes_an_unrelated_returned_location(monkeypatch):
    client, runtime, s3 = _stubbed_aws_client(monkeypatch)

    def unrelated_output(**request):
        runtime.requests.append(request)
        return {"OutputLocation": "s3://results/unrelated/result.json"}

    runtime.invoke_endpoint_async = unrelated_output
    s3.objects[("results", "unrelated/result.json")] = b"keep"
    with pytest.raises(RuntimeError, match="outside this invocation"):
        client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]])
    assert ("results", "unrelated/result.json") in s3.objects
    assert ("results", "unrelated/result.json") not in s3.deletes


def test_sagemaker_rejects_over_one_gb_without_allocating_it(monkeypatch):
    client, _, s3 = _stubbed_aws_client(monkeypatch)
    monkeypatch.setattr(
        nori_client_module.Path,
        "stat",
        lambda _path: types.SimpleNamespace(st_size=SAGEMAKER_MAX_INPUT_BYTES + 1),
    )
    with pytest.raises(ValueError, match="1 GB"):
        client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]])
    assert not s3.uploads


def test_aws_constructor_and_predict_reject_transport_mismatches(monkeypatch):
    monkeypatch.setattr(
        nori_client_module,
        "create_aws_clients",
        lambda _service_names, **_kwargs: (object(), object()),
    )

    with pytest.raises(ValueError, match="model is required"):
        SynthefyNoriClient(
            mode="sagemaker",
            endpoint_name="nori-dev",
            s3_staging_uri="s3://customer-bucket/nori/staging",
        )
    with pytest.raises(ValueError, match="endpoint_name is required"):
        SynthefyNoriClient(
            mode="sagemaker",
            model="nori-30m",
            s3_staging_uri="s3://customer-bucket/nori/staging",
        )
    with pytest.raises(ValueError, match="s3_staging_uri is required"):
        SynthefyNoriClient(mode="sagemaker", endpoint_name="nori-dev", model="nori-30m")
    with pytest.raises(ValueError, match="published Nori inference specification"):
        SynthefyNoriClient(
            mode="sagemaker",
            endpoint_name="nori-dev",
            model="synthefy/custom",
            s3_staging_uri="s3://customer-bucket/nori/staging",
        )
    with pytest.raises(ValueError, match="api_key is not used"):
        SynthefyNoriClient(
            mode="sagemaker",
            endpoint_name="nori-dev",
            model="nori-30m",
            api_key="secret",
            s3_staging_uri="s3://customer-bucket/nori/staging",
        )

    client, runtime, _ = _stubbed_aws_client(monkeypatch)
    runtime.response_overrides["predictions"] = [10.0]
    with pytest.raises(ValueError, match="extra_headers"):
        client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]], extra_headers={"x-test": "no"})
    with pytest.warns(UserWarning, match="Per-prediction timeout is ignored"):
        client.predict([[0.0], [1.0]], [0.0, 1.0], [[2.0]], timeout=60)


def test_aws_constructor_requires_a_scoped_customer_staging_prefix(monkeypatch):
    monkeypatch.setattr(
        nori_client_module,
        "create_aws_clients",
        lambda _service_names, **_kwargs: (object(), object()),
    )
    with pytest.raises(ValueError, match="include a customer-owned prefix"):
        SynthefyNoriClient(
            mode="sagemaker",
            endpoint_name="nori-dev",
            model="nori-30m",
            s3_staging_uri="s3://customer-bucket",
        )


def test_sagemaker_input_limit_matches_async_inference():
    assert SAGEMAKER_MAX_INPUT_BYTES == 1024**3


@pytest.mark.parametrize(
    "model",
    ["synthefy/ft-acme-churn-v1", "acme-churn-v1", "nori-6n", "Custom_Model", "ft-" + "a" * 61, 42],
    ids=["namespaced", "no-prefix", "mistyped-released", "uppercase-and-underscore", "64-characters", "not-a-string"],
)
def test_sagemaker_refuses_a_name_that_is_neither_released_nor_fine_tuned(monkeypatch, model):
    monkeypatch.setattr(
        nori_client_module,
        "create_aws_clients",
        lambda _service_names, **_kwargs: (object(), object()),
    )
    with pytest.raises(ValueError, match="published Nori inference specification") as caught:
        SynthefyNoriClient(
            mode="sagemaker",
            endpoint_name="nori-dev",
            model=model,
            s3_staging_uri="s3://customer-bucket/nori/staging",
        )
    assert "a fine-tuned model deployed on the endpoint" in str(caught.value)
    assert FINE_TUNED_MODEL_NAME_PATTERN in str(caught.value)


@pytest.mark.parametrize("model", ["ft-acme-churn-v1", "ft-" + "a" * 60], ids=["hyphenated", "63-characters"])
def test_sagemaker_sends_a_fine_tuned_name_as_the_request_model(monkeypatch, model):
    client, runtime, s3 = _stubbed_aws_client(monkeypatch, model=model)

    assert client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST) == [10.0, 20.0]
    assert s3.uploads[0]["metadata"]["model"] == model
    assert runtime.requests[0]["EndpointName"] == "nori-shared"


def test_sagemaker_refuses_an_answer_from_a_released_model_to_a_fine_tuned_name(monkeypatch):
    client, runtime, _ = _stubbed_aws_client(monkeypatch, model="ft-acme-churn-v1")
    runtime.response_overrides["model"] = "nori-6m"

    with pytest.raises(ValueError, match="requested 'ft-acme-churn-v1', received 'nori-6m'"):
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)


def test_sagemaker_leaves_a_fine_tuned_name_it_cannot_see_served_for_the_endpoint_to_refuse(monkeypatch):
    client, runtime, s3 = _stubbed_aws_client(monkeypatch, model="ft-acme-churn-v9", result="failure")
    runtime.failure_message = (
        "Requested model 'ft-acme-churn-v9' is not available on this endpoint. "
        "Choose one of ['nori-6m', 'nori-30m', 'ft-acme-churn-v1'] or its synthefy/ gateway slug."
    )

    with pytest.raises(SageMakerInvocationError, match="'ft-acme-churn-v9' is not available on this endpoint"):
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    assert s3.uploads[0]["metadata"]["model"] == "ft-acme-churn-v9"


def test_a_fine_tuned_name_containing_thinking_predicts_a_target_matrix(monkeypatch):
    client, runtime, s3 = _stubbed_aws_client(monkeypatch, model="ft-churn-thinking-v1")
    runtime.response_overrides.update(
        {"predictions": [[10.0, 11.0], [20.0, 21.0]], "multi_target_prediction_strategy": "copula"}
    )

    predictions = client.predict([[0.0], [1.0]], [[0.0, 1.0], [1.0, 0.0]], [[2.0], [3.0]])

    assert predictions == [[10.0, 11.0], [20.0, 21.0]]
    assert s3.uploads[0]["shapes"]["y_train"] == [2, 2]


def test_a_released_thinking_name_refuses_a_target_matrix_on_sagemaker(monkeypatch):
    client, runtime, s3 = _stubbed_aws_client(monkeypatch, model="nori-30m-thinking-medium")

    with pytest.raises(ValueError, match="not available on Nori Thinking"):
        client.predict([[0.0], [1.0]], [[0.0, 1.0], [1.0, 0.0]], [[2.0], [3.0]])
    assert not s3.uploads and not runtime.requests


def test_sagemaker_last_model_version_is_the_latest_accepted_answer(monkeypatch):
    digest = "sha256:" + "ab" * 32
    client, runtime, _ = _stubbed_aws_client(monkeypatch, model="ft-acme-churn-v1")
    runtime.response_overrides["model_version"] = digest

    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    assert client.last_model_version == digest

    runtime.response_overrides["model"] = "nori-6m"
    with pytest.raises(ValueError, match="model identity mismatch"):
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    assert client.last_model_version is None

    del runtime.response_overrides["model"]
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    del runtime.response_overrides["model_version"]
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    assert client.last_model_version is None


def test_remote_last_model_version_is_set_from_the_response_and_cleared_by_the_next_call():
    digest = "sha256:" + "cd" * 32
    responses = iter(
        [
            {"task": "regression", "predictions": [1.0, 2.0], "model_version": digest},
            {"task": "regression", "predictions": [1.0, 2.0]},
        ]
    )
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, lambda _request: httpx.Response(200, json=next(responses)))

    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    assert client.last_model_version == digest
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    assert client.last_model_version is None


_REFUSED_ANSWERS = {
    "quantiles-without-a-quantile-block": (
        {"output_type": "quantiles", "quantiles": [0.1, 0.5, 0.9]},
        {"output_type": "quantiles"},
        "returned no quantile block",
    ),
    "different-quantile-levels": (
        {"output_type": "quantiles", "quantiles": [0.1, 0.5, 0.9]},
        {
            "output_type": "quantiles",
            "quantiles": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "taus": [0.2, 0.5, 0.8],
        },
        r"Requested quantile levels \[0.1, 0.5, 0.9\] but the server returned \[0.2, 0.5, 0.8\]",
    ),
    "wrong-multi-target-shape": (
        {"y_train": [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]},
        {"predictions": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "multi_target_prediction_strategy": "copula"},
        r"multi-target predictions of shape \(2, 3\); expected \(2, 2\)",
    ),
}


@pytest.mark.parametrize("transport", ["remote", "sagemaker"])
@pytest.mark.parametrize("case", sorted(_REFUSED_ANSWERS))
def test_an_answer_refused_after_parsing_leaves_last_model_version_none(monkeypatch, transport, case):
    """A digest on an answer the shape checks refuse must not be published as the latest model."""
    predict_kwargs, refused_answer, message = _REFUSED_ANSWERS[case]
    answer = {"model_version": "sha256:" + "ef" * 32, **refused_answer}
    if transport == "sagemaker":
        client, runtime, _ = _stubbed_aws_client(monkeypatch)
        runtime.response_overrides.update(answer)
    else:
        client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
        body = {"task": "regression", "predictions": [10.0, 20.0], **answer}
        _attach_mock(client, lambda _request: httpx.Response(200, json=body))
    arguments = {"X_train": _X_TRAIN, "y_train": _Y_TRAIN, "X_test": _X_TEST, **predict_kwargs}

    with pytest.raises(ValueError, match=message):
        client.predict(**arguments)
    assert client.last_model_version is None


def test_installed_botocore_knows_every_parameter_the_async_invocation_sends(monkeypatch):
    """Every name the client passes must exist in the installed botocore's SageMaker Runtime model.

    A parameter botocore does not know fails with ParamValidationError before any request is
    signed. CI's artifact legs install boto3 and botocore at the ``synthefy[aws]`` floor, so
    there this proves the floor knows what predict() sends; the workspace run checks the
    newest locked release.
    """
    client, runtime, _ = _stubbed_aws_client(monkeypatch)
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    service_model = botocore.session.get_session().get_service_model("sagemaker-runtime")
    shape = service_model.operation_model("InvokeEndpointAsync").input_shape

    report = ParamValidator().validate(runtime.requests[0], shape)
    assert not report.has_errors(), report.generate_report()


# --------------------------------------------------------------------------- #
# Remote mode -- happy path
# --------------------------------------------------------------------------- #


def test_predict_returns_predictions_and_sends_expected_request():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([10.0, 20.0], capture))

    preds = client.predict(
        X_train=[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        y_train=[1.0, 1.0, 2.0],
        X_test=[[2.0, 2.0], [3.0, 3.0]],
    )

    assert preds == [10.0, 20.0]
    assert client.mode == "remote"
    # Gateway endpoint: correct path + model field in the body.
    assert capture["path"] == GATEWAY_ENDPOINT
    # Gateway requires the Bearer scheme (the default auth_scheme).
    assert capture["headers"]["authorization"] == "Bearer test-key"
    assert capture["headers"]["content-type"] == "application/json"
    body = capture["body"]
    assert body["X_train"] == [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    assert body["y_train"] == [1.0, 1.0, 2.0]
    assert body["X_test"] == [[2.0, 2.0], [3.0, 3.0]]
    assert body["task"] == "regression"
    # The chosen size resolves to its gateway slug in the body.
    assert body["model"] == "synthefy/nori-30m"


def test_null_prediction_returns_as_nan():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([None], {}))

    predictions = client.predict(X_train=[[0.0], [1.0]], y_train=[0.0, 1.0], X_test=[[2.0]])

    assert len(predictions) == 1 and math.isnan(predictions[0])


def test_predict_accepts_numpy_arrays():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([42.0], capture))

    preds = client.predict(
        X_train=np.array([[0.0, 1.0], [1.0, 0.0]]),
        y_train=np.array([1.0, 2.0]),
        X_test=np.array([[2.0, 2.0]]),
    )

    assert preds == [42.0]
    # numpy inputs are serialized to plain JSON lists of floats.
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 0.0]]
    assert capture["body"]["y_train"] == [1.0, 2.0]


# --------------------------------------------------------------------------- #
# pandas inputs -- DataFrame / Series
# --------------------------------------------------------------------------- #


def test_predict_accepts_dataframes_and_series():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([5.0], capture))

    X_train = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    y_train = pd.Series([1.0, 2.0])
    X_test = pd.DataFrame({"a": [2.0], "b": [2.0]})

    preds = client.predict(X_train, y_train, X_test)

    assert preds == [5.0]
    # DataFrame/Series inputs serialize to plain JSON lists of floats.
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 0.0]]
    assert capture["body"]["y_train"] == [1.0, 2.0]
    assert capture["body"]["X_test"] == [[2.0, 2.0]]


def test_y_train_single_column_dataframe_is_accepted():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0]}),
        y_train=pd.DataFrame({"target": [1.0, 2.0]}),
        X_test=pd.DataFrame({"a": [2.0]}),
    )
    assert capture["body"]["y_train"] == [1.0, 2.0]


def test_dataframe_xtest_is_aligned_to_xtrain_by_column_name():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([9.0], capture))

    X_train = pd.DataFrame({"a": [0.0, 1.0], "b": [10.0, 11.0]})
    # X_test columns are in the opposite order; they must be realigned to a, b.
    X_test = pd.DataFrame({"b": [12.0], "a": [2.0]})

    client.predict(X_train, [1.0, 2.0], X_test)

    # Realigned to X_train's column order (a, b), not X_test's literal order.
    assert capture["body"]["X_test"] == [[2.0, 12.0]]


def test_dataframe_column_set_mismatch_raises():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    with pytest.raises(ValueError, match="same feature columns"):
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"a": [2.0], "c": [2.0]}),
        )


# --------------------------------------------------------------------------- #
# One-hot featurization of non-numeric DataFrame columns (fit on X_train)
# --------------------------------------------------------------------------- #


def test_non_numeric_columns_are_ordinal_encoded_by_default():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([5.0, 6.0], capture))

    out = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["y", "x", "y"]}),
        y_train=[1.0, 2.0, 3.0],
        # 'z' is unseen in training -> bounded other code K=2.
        X_test=pd.DataFrame({"a": [3.0, 4.0], "cat": ["x", "z"]}),
    )

    assert out == [5.0, 6.0]
    # one column per categorical, codes in sorted-category order: x=0, y=1
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]
    assert capture["body"]["X_test"] == [[3.0, 0.0], [4.0, 2.0]]


def test_ordinal_missing_categorical_is_forwarded_as_null():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["x", None, "y"]}),
        y_train=[1.0, 2.0, 3.0],
        X_test=pd.DataFrame({"a": [5.0], "cat": ["x"]}),
    )
    sent = capture["body"]["X_train"]
    # x=0, y=1; the missing row is a JSON null for server-side imputation.
    assert sent[0] == [0.0, 0.0] and sent[2] == [2.0, 1.0]
    assert sent[1][1] is None
    assert capture["body"]["X_test"] == [[5.0, 0.0]]


def test_ordinal_literal_nan_string_is_a_real_category():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": ["nan", "x"]}),
        y_train=[1.0, 2.0],
        X_test=pd.DataFrame({"a": [2.0], "cat": ["nan"]}),
    )
    # "nan" (the string) sorts before "x": nan=0, x=1 — not treated as missing.
    assert capture["body"]["X_train"] == [[0.0, 0.0], [1.0, 1.0]]
    assert capture["body"]["X_test"] == [[2.0, 0.0]]


def test_ordinal_name_value_collision_is_a_non_issue():
    # Under one-hot, column 'a' value 'b_x' and column 'a_b' value 'x' collide
    # in the '<column>_<value>' namespace; ordinal never generates columns.
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": ["b_x", "c"], "a_b": ["x", "y"]}),
        y_train=[1.0, 2.0],
        X_test=pd.DataFrame({"a": ["b_x"], "a_b": ["x"]}),
    )
    assert capture["body"]["X_train"] == [[0.0, 0.0], [1.0, 1.0]]
    assert capture["body"]["X_test"] == [[0.0, 0.0]]


def test_invalid_categorical_encoding_raises():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    df = pd.DataFrame({"cat": ["x", "y"]})
    with pytest.raises(ValueError, match="categorical_encoding"):
        client.predict(
            X_train=df,
            y_train=[1.0, 2.0],
            X_test=df,
            categorical_encoding="hashing",
        )


def test_non_numeric_columns_are_one_hot_encoded():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([5.0], capture))

    out = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": ["x", "y"]}),
        y_train=[1.0, 2.0],
        # 'z' is unseen in training -> its indicator group is all zeros.
        X_test=pd.DataFrame({"a": [2.0], "cat": ["z"]}),
        categorical_encoding="onehot",
    )

    assert out == [5.0]
    # columns: a, cat_x, cat_y  (numerics first, then sorted one-hot groups)
    assert capture["body"]["X_train"] == [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    assert capture["body"]["X_test"] == [[2.0, 0.0, 0.0]]


def test_one_hot_train_category_absent_in_test_is_kept_as_zero_column():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([9.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["x", "y", "z"]}),
        y_train=[1.0, 2.0, 3.0],
        X_test=pd.DataFrame({"a": [5.0], "cat": ["x"]}),
        categorical_encoding="onehot",
    )

    # train has 3 categories -> cat_x, cat_y, cat_z; test row 'x' -> [1,0,0]
    assert capture["body"]["X_test"] == [[5.0, 1.0, 0.0, 0.0]]


def test_auto_high_cardinality_column_requires_explicit_choice():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")

    with pytest.raises(ValueError, match="Automatic handling is ambiguous"):
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "hc": ["p", "q", "r"]}),
            y_train=[1.0, 2.0, 3.0],
            X_test=pd.DataFrame({"a": [3.0], "hc": ["p"]}),
            max_categorical_cardinality=2,
        )


def test_explicit_high_cardinality_column_uses_top_k_plus_other():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0, 2.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "hc": ["p", "q", "r"]}),
        y_train=[1.0, 2.0, 3.0],
        X_test=pd.DataFrame({"a": [3.0, 4.0], "hc": ["p", "new"]}),
        max_categorical_cardinality=2,
        categorical_columns=["hc"],
    )
    # p and q win the deterministic frequency tie; r/new share other=2.
    assert capture["body"]["X_train"] == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    assert capture["body"]["X_test"] == [[3.0, 0.0], [4.0, 2.0]]


def test_datetime_column_requires_explicit_conversion():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")

    with pytest.raises(ValueError, match="unsupported dtype"):
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0], "d": pd.to_datetime(["2024-01-01", "2024-01-02"])}),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"a": [2.0], "d": pd.to_datetime(["2024-01-03"])}),
        )


def test_bool_columns_pass_through_as_numeric():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    # bool is numeric (is_numeric_dtype) -> not one-hot; True/False -> 1.0/0.0
    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "flag": [True, False]}),
        y_train=[1.0, 2.0],
        X_test=pd.DataFrame({"a": [2.0], "flag": [True]}),
    )
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 0.0]]
    assert capture["body"]["X_test"] == [[2.0, 1.0]]


def test_all_numeric_dataframe_is_not_featurized():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any featurization warning would fail
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"a": [2.0], "b": [2.0]}),
        )
    assert capture["body"]["X_train"] == [[0.0, 1.0], [1.0, 0.0]]


def test_numpy_string_array_raises_pointing_to_dataframe():
    # A 2D numpy/list array is single-dtype, so a string column makes the WHOLE
    # array strings — there are no per-column types to one-hot. We raise and
    # point the caller to DataFrames (where each column keeps its own dtype).
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    with pytest.raises(ValueError, match="one-hot"):
        client.predict(
            X_train=np.array([[1.0, "x"], [2.0, "y"]]),
            y_train=[1.0, 2.0],
            X_test=np.array([[3.0, "z"]]),
        )


def test_column_numeric_in_train_but_not_test_raises_clearly():
    # A column numeric in X_train but object-dtype in X_test is caught with a
    # clear type-mismatch error (not a later cryptic float-cast failure).
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    with pytest.raises(ValueError, match="matching column types"):
        client.predict(
            X_train=pd.DataFrame({"b": [1.0, 2.0]}),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"b": ["x"]}),  # object dtype, not numeric
        )


def test_numeric_category_dtype_is_respected_as_categorical():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must NOT warn / drop / explode
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "r": pd.Categorical([1, 2, 3], categories=[1, 2, 3])}),
            y_train=[1.0, 2.0, 3.0],
            X_test=pd.DataFrame({"a": [5.0], "r": pd.Categorical([2], categories=[1, 2, 3])}),
        )
    assert capture["body"]["X_train"] == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    assert capture["body"]["X_test"] == [[5.0, 1.0]]


def test_all_missing_categorical_column_is_preserved_as_null():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": [None, None]}),
        y_train=[1.0, 2.0],
        X_test=pd.DataFrame({"a": [2.0], "cat": [None]}),
    )
    assert capture["body"]["X_train"] == [[0.0, None], [1.0, None]]
    assert capture["body"]["X_test"] == [[2.0, None]]


def test_timedelta_column_raises_unsupported():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    with pytest.raises(ValueError, match="timedelta"):
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0], "d": pd.to_timedelta(["1 days", "2 days"])}),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"a": [2.0], "d": pd.to_timedelta(["3 days"])}),
        )


def test_nan_in_categorical_gets_its_own_indicator_column():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["x", None, "y"]}),
        y_train=[1.0, 2.0, 3.0],
        X_test=pd.DataFrame({"a": [5.0], "cat": ["x"]}),
        categorical_encoding="onehot",
    )
    # columns: a, cat_x, cat_y, cat_nan (the missing row -> its own indicator)
    assert capture["body"]["X_train"] == [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0, 0.0],
    ]
    assert capture["body"]["X_test"] == [[5.0, 1.0, 0.0, 0.0]]


def test_integer_category_with_nan_does_not_crash():
    # Regression: demoting an int-category column to numeric must not choke on
    # NaN (it promotes to float and the NaN is forwarded for server imputation).
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"r": pd.Categorical([1, 2, None], categories=[1, 2, 3])}),
        y_train=[1.0, 2.0, 3.0],
        X_test=pd.DataFrame({"r": pd.Categorical([2], categories=[1, 2, 3])}),
    )
    sent = capture["body"]["X_train"]
    assert sent[0] == [0.0] and sent[1] == [1.0]
    assert sent[2][0] is None  # standards-compliant missing value, not a NaN token


def test_duplicate_column_names_raise():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    df = pd.DataFrame([[0.0, 1.0, 2.0]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate column name"):
        client.predict(X_train=df, y_train=[1.0], X_test=df)


def test_one_hot_name_value_collision_raises_clearly():
    # column 'a' value 'b_x' and column 'a_b' value 'x' both -> dummy 'a_b_x'
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    Xtr = pd.DataFrame({"a": ["b_x", "b_x"], "a_b": ["x", "y"]})
    Xte = pd.DataFrame({"a": ["b_x"], "a_b": ["x"]})
    with pytest.raises(ValueError, match="duplicate column names"):
        client.predict(X_train=Xtr, y_train=[1.0, 2.0], X_test=Xte, categorical_encoding="onehot")


def test_period_column_raises_unsupported():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    with pytest.raises(ValueError, match="unsupported dtype"):
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0], "p": pd.period_range("2024-01", periods=2, freq="M")}),
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame({"a": [2.0], "p": pd.period_range("2024-03", periods=1, freq="M")}),
        )


def test_zero_feature_columns_raises():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    with pytest.raises(ValueError, match="at least one feature column"):
        client.predict(
            X_train=pd.DataFrame(index=[0, 1]),  # 2 rows, 0 columns
            y_train=[1.0, 2.0],
            X_test=pd.DataFrame(index=[0]),
        )


def test_categorical_train_with_array_xtest_points_to_dataframe():
    # X_train is a DataFrame with a categorical column but X_test is an array:
    # we can't align/one-hot, so raise a message pointing at passing DataFrames.
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    with pytest.raises(ValueError, match="not a DataFrame"):
        client.predict(
            X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": ["x", "y"]}),
            y_train=[1.0, 2.0],
            X_test=np.array([[2.0, 0.0]]),
        )


def test_literal_nan_string_category_is_not_treated_as_missing():
    # A real category whose value is the string "nan" (no actual NaN) must encode
    # cleanly and NOT trip the duplicate-column guard against the dummy_na column.
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": ["nan", "x"]}),
        y_train=[1.0, 2.0],
        X_test=pd.DataFrame({"a": [2.0], "cat": ["nan"]}),
        categorical_encoding="onehot",
    )
    # columns: a, cat_nan, cat_x  — no NaN-indicator column, no error
    assert capture["body"]["X_train"] == [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    assert capture["body"]["X_test"] == [[2.0, 1.0, 0.0]]


def test_nonpositive_cardinality_cap_raises():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    for cap in (0, -5):
        with pytest.raises(ValueError, match="positive integer"):
            client.predict(
                X_train=pd.DataFrame({"a": [0.0, 1.0], "cat": ["x", "y"]}),
                y_train=[1.0, 2.0],
                X_test=pd.DataFrame({"a": [2.0], "cat": ["x"]}),
                max_categorical_cardinality=cap,
            )


# --------------------------------------------------------------------------- #
# as_pandas=True -- return a Series (named after y_train, indexed by X_test)
# --------------------------------------------------------------------------- #


def test_default_return_is_a_plain_list_not_series():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0, 2.0], capture))

    out = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0]}),
        y_train=pd.Series([1.0, 2.0], name="demand"),
        X_test=pd.DataFrame({"a": [2.0, 3.0]}),
    )
    assert isinstance(out, list)
    assert out == [1.0, 2.0]


def test_as_pandas_returns_series_named_after_y_train_and_indexed_by_xtest():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([10.0, 20.0], capture))

    X_test = pd.DataFrame({"a": [2.0, 3.0]}, index=["w1", "w2"])
    out = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0]}),
        y_train=pd.Series([1.0, 2.0], name="demand"),
        X_test=X_test,
        as_pandas=True,
    )

    assert isinstance(out, pd.Series)
    assert out.name == "demand"
    assert list(out.index) == ["w1", "w2"]
    assert out.tolist() == [10.0, 20.0]


def test_as_pandas_uses_single_column_dataframe_y_label():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([7.0], capture))

    out = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0]}),
        y_train=pd.DataFrame({"units": [1.0, 2.0]}),
        X_test=pd.DataFrame({"a": [2.0]}),
        as_pandas=True,
    )
    assert isinstance(out, pd.Series)
    assert out.name == "units"


def test_as_pandas_with_non_pandas_inputs_uses_defaults():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([5.0, 6.0], capture))

    out = client.predict(
        X_train=[[0.0], [1.0]],
        y_train=[1.0, 2.0],
        X_test=[[2.0], [3.0]],
        as_pandas=True,
    )
    assert isinstance(out, pd.Series)
    assert out.name == "prediction"  # no name available from a plain list
    assert list(out.index) == [0, 1]  # default RangeIndex
    assert out.tolist() == [5.0, 6.0]


# --------------------------------------------------------------------------- #
# Missing features are supported; hosted targets must be finite
# --------------------------------------------------------------------------- #


def test_nullable_matrix_vectorizes_non_finite_cells():
    values = np.array([[1.0, np.nan, np.inf, -np.inf]])

    assert _nullable_matrix(values) == [[1.0, None, None, None]]


def test_nan_is_sent_to_the_server_as_json_null():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    # Missing feature values must NOT raise; the model imputes them server-side.
    client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, np.nan]}),
        y_train=[1.0, 2.0],
        X_test=np.array([[2.0, 2.0]]),
    )

    sent = capture["body"]["X_train"]
    assert sent[1][1] is None


def _create_stubbed_hosted_target_client(monkeypatch, mode, *, multi_target=False):
    predictions = [[10.0, 11.0], [20.0, 21.0]] if multi_target else [10.0, 20.0]
    response = {"task": "regression", "predictions": predictions}
    if multi_target:
        response["multi_target_prediction_strategy"] = "copula"
    if mode == "sagemaker":
        client, runtime, s3 = _stubbed_aws_client(monkeypatch)
        runtime.response_overrides.update(response)
        return client, s3.uploads, runtime.requests

    requests = []

    def capture_hosted_prediction_request(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, capture_hosted_prediction_request)
    return client, [], requests


@pytest.mark.parametrize("mode", ["remote", "sagemaker"])
@pytest.mark.parametrize("missing", [None, np.nan, np.inf, -np.inf], ids=["null", "nan", "inf", "-inf"])
@pytest.mark.parametrize("target_format", ["list", "matrix", "series", "dataframe"])
def test_hosted_rejects_non_finite_targets_without_network(monkeypatch, mode, missing, target_format):
    y_train = [1.0, missing]
    if target_format == "matrix":
        y_train = np.array([[1.0, 2.0], [3.0, missing]], dtype=float)
    elif target_format == "series":
        y_train = pd.Series(y_train, dtype="Float64")
    elif target_format == "dataframe":
        y_train = pd.DataFrame({"a": [2.0, 3.0], "b": pd.Series(y_train, dtype="Float64")})
    client, uploads, requests = _create_stubbed_hosted_target_client(monkeypatch, mode)

    with client, pytest.raises(ValueError, match="y_train must contain only finite numbers"):
        client.predict([[0.0], [1.0]], y_train, [[2.0], [3.0]])

    assert uploads == []
    assert requests == []


@pytest.mark.parametrize("mode", ["remote", "sagemaker"])
def test_hosted_rejects_non_finite_targets_before_text_preprocessing(monkeypatch, mode):
    def reject_text_preprocessing(*args, **kwargs):
        pytest.fail("Invalid targets must be rejected before text preprocessing")

    monkeypatch.setattr(nori_client_module, "_widen_text_columns", reject_text_preprocessing)
    client, uploads, requests = _create_stubbed_hosted_target_client(monkeypatch, mode)
    with client, pytest.raises(ValueError, match="y_train must contain only finite numbers"):
        client.predict(
            pd.DataFrame({"text": ["a", "b"]}),
            [1.0, np.nan],
            pd.DataFrame({"text": ["c", "d"]}),
            text_columns=["text"],
        )

    assert uploads == []
    assert requests == []


@pytest.mark.parametrize("mode", ["remote", "sagemaker"])
@pytest.mark.parametrize("multi_target", [False, True], ids=["vector", "matrix"])
def test_hosted_accepts_finite_targets_with_missing_features(monkeypatch, mode, multi_target):
    y_train = [[1.0, 2.0], [3.0, 4.0]] if multi_target else [1.0, 2.0]
    client, uploads, requests = _create_stubbed_hosted_target_client(monkeypatch, mode, multi_target=multi_target)
    with client:
        predictions = client.predict([[0.0, np.nan], [1.0, 2.0]], y_train, [[2.0, None], [3.0, 4.0]])

    assert predictions == ([[10.0, 11.0], [20.0, 21.0]] if multi_target else [10.0, 20.0])
    assert len(requests) == 1
    assert len(uploads) == (1 if mode == "sagemaker" else 0)


@pytest.mark.parametrize("missing", [None, np.nan, np.inf, -np.inf], ids=["null", "nan", "inf", "-inf"])
def test_local_delegates_non_finite_target_validation_to_runtime(monkeypatch, missing):
    captured = {}

    def predict_locally(X_train, y_train, X_test, *, task, model):
        captured["targets"] = y_train
        return [10.0, 20.0]

    monkeypatch.setattr(nori_client_module, "_load_local_predict", lambda: predict_locally)
    with SynthefyNoriClient(mode="local", model="nori-30m") as client:
        assert client.predict([[0.0], [1.0]], [1.0, missing], [[2.0], [3.0]]) == [10.0, 20.0]

    np.testing.assert_equal(captured["targets"], np.asarray([1.0, missing], dtype=float))


def test_custom_http_endpoint_still_requires_and_sends_model():
    capture: Dict = {}
    client = SynthefyNoriClient(
        api_key="test-key",
        base_url="https://example.invalid",
        endpoint="/predict",
        model="customer/custom-nori",
    )
    _attach_mock(client, _ok_handler([1.0], capture))

    preds = client.predict(X_train=[[0.0], [1.0]], y_train=[0.0, 1.0], X_test=[[2.0]])

    assert preds == [1.0]
    assert capture["path"] == "/predict"
    assert capture["body"]["model"] == "customer/custom-nori"


def test_no_dedicated_url_constants_are_exported():
    """The client addresses Nori only by gateway slug, never by a per-model host.

    A hardcoded ``model-<id>.api.baseten.co`` constant cannot stay correct: each variant is
    its own Baseten model with its own id, and an id does not survive a model being deleted
    and re-created. Callers use ``model=`` and let the gateway resolve it.
    """
    import synthefy.nori_client as nc

    for name in ("DEDICATED_BASE_URL", "DEDICATED_ENDPOINT"):
        assert not hasattr(nc, name), f"{name} must not exist; address Nori by gateway slug"
    # The gateway host is `inference.baseten.co`; a per-model host is `model-<id>.api.baseten.co`,
    # so any `api.baseten.co` occurrence means an id-addressed host came back.
    source = Path(nc.__file__).read_text()
    assert "api.baseten.co" not in source, (
        "a per-model Baseten host is hardcoded in nori_client.py; use the gateway slug"
    )


# --------------------------------------------------------------------------- #
# Mode selection / authentication / configuration
# --------------------------------------------------------------------------- #


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("SYNTHEFY_NORI_API_KEY", "env-key")
    client = SynthefyNoriClient(model="nori-30m")
    assert client.api_key == "env-key"
    assert client.mode == "remote"


def test_baseten_env_is_not_read(monkeypatch):
    """BASETEN_API_KEY was the pre-5.0 name and is no longer honoured."""
    monkeypatch.delenv("SYNTHEFY_NORI_API_KEY", raising=False)
    monkeypatch.setenv("BASETEN_API_KEY", "legacy-key")
    with pytest.raises(ValueError, match="SYNTHEFY_NORI_API_KEY"):
        SynthefyNoriClient(model="nori-30m")


def test_explicit_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("SYNTHEFY_NORI_API_KEY", "from-env")
    assert SynthefyNoriClient(api_key="explicit", model="nori-30m").api_key == "explicit"


def test_missing_api_key_raises_in_remote_mode(monkeypatch):
    monkeypatch.delenv("SYNTHEFY_NORI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SYNTHEFY_NORI_API_KEY"):
        SynthefyNoriClient(model="nori-30m")


def test_local_mode_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("SYNTHEFY_NORI_API_KEY", raising=False)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    assert client.mode == "local"
    assert client.client is None


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        SynthefyNoriClient(api_key="test-key", mode="nope")


def test_default_auth_scheme_is_bearer():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    assert client.auth_scheme == "Bearer"


def test_auth_scheme_override_sets_authorization_header():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", auth_scheme="Api-Key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])

    assert capture["headers"]["authorization"] == "Api-Key test-key"


def test_invalid_auth_scheme_raises():
    with pytest.raises(ValueError, match="auth_scheme must be one of"):
        SynthefyNoriClient(api_key="test-key", auth_scheme="Token")


def test_auto_mode_is_rejected_before_credentials_model_or_transport(monkeypatch):
    def unexpected_transport(*args, **kwargs):
        raise AssertionError("transport setup must not run for mode='auto'")

    monkeypatch.delenv("SYNTHEFY_NORI_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "Client", unexpected_transport)
    monkeypatch.setattr(
        "synthefy.nori_client.create_aws_clients",
        unexpected_transport,
    )
    monkeypatch.setattr("synthefy.nori_client._load_local_predict", unexpected_transport)
    monkeypatch.setattr("synthefy.nori_client._load_local_regressor", unexpected_transport)

    with pytest.raises(
        ValueError,
        match=r"mode='auto'.*removed.*'remote'.*'sagemaker'.*'local'",
    ):
        SynthefyNoriClient(mode="auto")


def test_context_manager_closes_client():
    with SynthefyNoriClient(api_key="test-key", model="nori-30m") as client:
        assert isinstance(client, SynthefyNoriClient)
    assert client.client.is_closed


# --------------------------------------------------------------------------- #
# Shape validation (runs before any network call or local import)
# --------------------------------------------------------------------------- #


@pytest.fixture
def client() -> SynthefyNoriClient:
    # No transport is attached; valid inputs would fail, but these tests assert
    # that validation raises *before* any request is attempted.
    return SynthefyNoriClient(api_key="test-key", model="nori-30m")


def test_mismatched_train_rows_raises(client):
    with pytest.raises(ValueError, match="they must match"):
        client.predict(X_train=[[1.0], [2.0]], y_train=[1.0], X_test=[[3.0]])


def test_feature_count_mismatch_raises(client):
    with pytest.raises(ValueError, match="features"):
        client.predict(X_train=[[1.0, 2.0]], y_train=[1.0], X_test=[[3.0, 4.0, 5.0]])


def test_non_2d_x_train_raises(client):
    with pytest.raises(ValueError, match="X_train must be 2D"):
        client.predict(X_train=[1.0, 2.0, 3.0], y_train=[1.0], X_test=[[3.0]])


def test_empty_x_train_raises(client):
    # A 2D array with zero rows reaches the row-count guard.
    with pytest.raises(ValueError, match="at least one context row"):
        client.predict(X_train=np.empty((0, 2)), y_train=[], X_test=[[3.0, 4.0]])


def test_flat_empty_x_train_raises_dimensionality(client):
    # A flat empty list is 1D, so it fails the 2D guard instead.
    with pytest.raises(ValueError, match="X_train must be 2D"):
        client.predict(X_train=[], y_train=[], X_test=[[3.0]])


def test_ragged_x_train_raises(client):
    with pytest.raises(ValueError, match="X_train"):
        client.predict(X_train=[[1.0, 2.0], [3.0]], y_train=[1.0, 2.0], X_test=[[3.0, 4.0]])


# --------------------------------------------------------------------------- #
# Remote mode -- error mapping
# --------------------------------------------------------------------------- #


def test_http_400_maps_to_bad_request_error_with_server_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "missing field: y_train"})

    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, handler)

    with pytest.raises(BadRequestError) as exc_info:
        client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])

    assert "missing field: y_train" in str(exc_info.value)
    assert exc_info.value.status_code == 400


def test_http_401_maps_to_authentication_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    client = SynthefyNoriClient(api_key="bad-key", model="nori-30m")
    _attach_mock(client, handler)

    with pytest.raises(AuthenticationError) as exc_info:
        client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])

    assert exc_info.value.status_code == 401


def test_retries_on_server_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("synthefy.nori_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "temporarily down"})
        return httpx.Response(200, json={"task": "regression", "predictions": [7.0]})

    client = SynthefyNoriClient(api_key="test-key", max_retries=2, model="nori-30m")
    _attach_mock(client, handler)

    preds = client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])

    assert preds == [7.0]
    assert calls["n"] == 2


def test_exhausted_retries_raise_internal_server_error(monkeypatch):
    monkeypatch.setattr("synthefy.nori_client.time.sleep", lambda _s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = SynthefyNoriClient(api_key="test-key", max_retries=1, model="nori-30m")
    _attach_mock(client, handler)

    with pytest.raises(InternalServerError):
        client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])


def test_final_error_wins_over_stale_earlier_attempt(monkeypatch):
    # A transient connection error on attempt 0 followed by a retryable 5xx on the
    # final attempt must surface the FINAL error (InternalServerError), not the
    # stale APIConnectionError from the earlier attempt.
    monkeypatch.setattr("synthefy.nori_client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("transient")
        return httpx.Response(503, json={"error": "down"})

    client = SynthefyNoriClient(api_key="test-key", max_retries=1, model="nori-30m")
    _attach_mock(client, handler)

    with pytest.raises(InternalServerError):
        client.predict(X_train=[[1.0]], y_train=[1.0], X_test=[[2.0]])
    assert calls["n"] == 2


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan"), True])
def test_constructor_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        SynthefyNoriClient(api_key="test-key", model="nori-30m", timeout=timeout)


@pytest.mark.parametrize("max_retries", [-1, 1.5, True])
def test_constructor_rejects_invalid_max_retries(max_retries):
    with pytest.raises(ValueError, match="max_retries"):
        SynthefyNoriClient(
            api_key="test-key",
            model="nori-30m",
            max_retries=max_retries,
        )


def test_per_call_zero_timeout_is_forwarded_exactly():
    capture: Dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        capture["timeout"] = request.extensions["timeout"]
        return httpx.Response(200, json={"task": "regression", "predictions": [7.0]})

    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, handler)

    assert client.predict([[1.0]], [1.0], [[2.0]], timeout=0.0) == [7.0]
    assert set(capture["timeout"].values()) == {0.0}


def test_negative_retry_after_is_clamped_to_zero():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    response = httpx.Response(429, headers={"Retry-After": "-5"})

    assert client._compute_backoff(0, response) == 0.0


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #


def test_request_model_roundtrip():
    req = NoriPredictRequest(X_train=[[1.0, 2.0]], y_train=[3.0], X_test=[[4.0, 5.0]])
    assert req.model_dump() == {
        "X_train": [[1.0, 2.0]],
        "y_train": [3.0],
        "X_test": [[4.0, 5.0]],
        "task": "regression",
        # Optional serving-memory policy. None by default, and _predict_remote excludes it
        # from the payload when unset, so an existing caller's request is unchanged on the
        # wire -- see test_a_request_without_memory_does_not_send_the_field.
        "memory_policy": None,
        "output_type": None,
        "quantiles": None,
        "discretize": None,
        "categorical_levels": None,
        "large_context_policy": None,
        "large_context_threshold": None,
        "large_context_seed": None,
        "large_context_holdout": None,
        "multi_target_prediction_strategy": None,
        "multi_target_prediction_policy": None,
    }


def test_request_model_omits_unset_distribution_fields_on_the_wire():
    # The canonical wire serializer must stay
    # exactly what earlier client versions sent, so adding these fields cannot
    # change any existing request.
    req = NoriPredictRequest(X_train=[[1.0, 2.0]], y_train=[3.0], X_test=[[4.0, 5.0]])
    assert req.to_wire() == {
        "X_train": [[1.0, 2.0]],
        "y_train": [3.0],
        "X_test": [[4.0, 5.0]],
        "task": "regression",
    }


def test_remote_multi_target_samples_are_one_call_and_shape_checked():
    capture: Dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        capture["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "task": "regression",
                "model": "nori-30m",
                "predictions": [[1.0, 2.0], [3.0, 4.0]],
                "samples": [
                    [[0.5, 1.5], [1.5, 2.5]],
                    [[2.5, 3.5], [3.5, 4.5]],
                ],
                "output_type": "samples",
                "multi_target_prediction_strategy": "independent",
            },
        )

    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, handler)
    result = client.predict(
        [[0.0], [1.0], [2.0]],
        [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]],
        [[3.0], [4.0]],
        output_type="samples",
        multi_target_prediction_strategy="independent",
        multi_target_prediction_policy={"n_draws": 2, "random_state": 42},
    )

    assert np.asarray(result).shape == (2, 2, 2)
    assert capture["body"]["y_train"] == [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]
    assert capture["body"]["multi_target_prediction_policy"] == {"n_draws": 2, "random_state": 42}


def test_remote_multi_target_samples_reject_wrong_draw_count():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "task": "regression",
                "model": "nori-30m",
                "predictions": [[1.0, 2.0]],
                "samples": [[[0.5, 1.5]]],
                "output_type": "samples",
                "multi_target_prediction_strategy": "independent",
            },
        )

    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, handler)

    with pytest.raises(ValueError, match=r"expected \(1, 2, 2\)"):
        client.predict(
            [[0.0], [1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[2.0]],
            output_type="samples",
            multi_target_prediction_strategy="independent",
            multi_target_prediction_policy={"n_draws": 2},
        )


def test_multi_target_samples_reject_as_pandas_before_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be called")

    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, handler)

    with pytest.raises(ValueError, match="as_pandas=True is not supported"):
        client.predict(
            [[0.0], [1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[2.0]],
            output_type="samples",
            as_pandas=True,
        )


def test_multi_target_work_bounds_apply_only_to_hosted_modes():
    controls = {
        "y_train": [[0.0, 1.0], [1.0, 0.0]],
        "output_type": "samples",
        "strategy": "autoregressive",
        "policy": {
            "n_draws": 1_001,
            "copula_cv": 21,
            "copula_pit_jitter": 0.2,
            "autoregressive_n_orders": 9,
        },
        "large_context_policy": None,
        "discretizing": False,
        "is_thinking": False,
    }

    assert _validate_multi_target_controls(**controls, mode="local") is True
    with pytest.raises(ValueError, match="n_draws must be <= 1000 for hosted inference"):
        _validate_multi_target_controls(**controls, mode="remote")


def test_remote_autoregressive_orders_are_sent_and_exposed():
    capture: Dict = {}
    orders = [[1, 0], [0, 1]]

    def handler(request: httpx.Request) -> httpx.Response:
        capture["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "task": "regression",
                "model": "nori-30m",
                "predictions": [[1.0, 2.0]],
                "multi_target_prediction_strategy": "autoregressive",
                "target_orders": orders,
            },
        )

    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, handler)
    client.predict(
        [[0.0], [1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[2.0]],
        multi_target_prediction_strategy="autoregressive",
        multi_target_prediction_policy={"autoregressive_orders": orders},
    )

    assert capture["body"]["multi_target_prediction_policy"] == {"autoregressive_orders": orders}
    assert client.last_target_orders == orders


def test_remote_multi_target_memory_reports_are_exposed():
    reports = [{**_REPORT, "strategy": "independent", "target": target, "order": None} for target in range(2)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "task": "regression",
                "model": "nori-30m",
                "predictions": [[1.0, 2.0]],
                "multi_target_prediction_strategy": "independent",
                "multi_target_memory_reports": reports,
            },
        )

    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, handler)
    client.predict(
        [[0.0], [1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[2.0]],
        memory_policy="exact",
        multi_target_prediction_strategy="independent",
    )

    assert client.last_memory_report is None
    assert client.last_multi_target_memory_reports == reports


def test_response_model_parses_predictions():
    resp = NoriPredictResponse(**{"task": "regression", "predictions": [1.0, 2.0, 3.0]})
    assert resp.predictions == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# Local mode
# --------------------------------------------------------------------------- #


def test_local_predict_raises_helpful_error_without_package(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "synthefy_nori" or name.startswith("synthefy_nori."):
            raise ModuleNotFoundError("No module named 'synthefy_nori'", name="synthefy_nori")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client = SynthefyNoriClient(mode="local", model="nori-30m")
    with pytest.raises(ImportError, match=r"synthefy-nori"):
        client.predict(X_train=[[1.0, 2.0]], y_train=[3.0], X_test=[[4.0, 5.0]])


def test_local_loaders_do_not_mask_transitive_import_failures(monkeypatch):
    from synthefy import nori_client as module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "synthefy_nori" or name.startswith("synthefy_nori."):
            raise ModuleNotFoundError(
                "No module named 'sentinel_transitive_dep'",
                name="sentinel_transitive_dep",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for loader in (module._load_local_predict, module._load_local_regressor):
        with pytest.raises(ModuleNotFoundError, match="sentinel_transitive_dep"):
            loader()


def test_local_predict_validates_shapes_before_import():
    # Shape validation happens before the optional dependency is imported, so
    # this raises ValueError regardless of whether synthefy-nori is present.
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    with pytest.raises(ValueError, match="they must match"):
        client.predict(X_train=[[1.0], [2.0]], y_train=[1.0], X_test=[[3.0]])


@pytest.mark.slow
def test_local_predict_real_inference():
    pytest.importorskip("synthefy_nori")

    client = SynthefyNoriClient(mode="local", model="nori-30m")
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 3))
    y_train = X_train[:, 0] * 2.0 + 1.0
    X_test = rng.normal(size=(5, 3))

    preds = client.predict(X_train, y_train, X_test)

    assert isinstance(preds, list)
    assert len(preds) == 5
    assert all(isinstance(p, float) for p in preds)


# --------------------------------------------------------------------------- #
# Model-variant selector
# --------------------------------------------------------------------------- #

_XTR = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
_YTR = [1.0, 1.0, 2.0]
_XTE = [[2.0, 2.0]]


def test_model_variant_resolves_gateway_and_local():
    c30 = SynthefyNoriClient(api_key="k", model="nori-30m")
    assert c30.model == "synthefy/nori-30m"
    assert c30._local_variant == "nori-30m"

    # "nori-6m" is the ~6M base: its own gateway slug + explicit local variant
    c6 = SynthefyNoriClient(api_key="k", model="nori-6m")
    assert c6.model == "synthefy/nori-6m" and c6._local_variant == "nori-6m"

    # a bare "nori" is not a valid selector -- it names no size, so it is NOT in the registry
    # (it passes through as a raw gateway model, it does not resolve to a size)
    assert "nori" not in NORI_VARIANTS

    # a raw gateway slug passes through unchanged
    craw = SynthefyNoriClient(api_key="k", model="synthefy/custom")
    assert craw.model == "synthefy/custom" and craw._local_variant is None


def test_model_is_required_no_default():
    # There is no default model -- omitting model= raises (every request names a size).
    with pytest.raises(ValueError, match="model is required"):
        SynthefyNoriClient(api_key="k")
    with pytest.raises(ValueError, match="model is required"):
        SynthefyNoriClient(api_key="k", model=None)
    # An explicit size resolves to its size-suffixed gateway slug + local checkpoint.
    c = SynthefyNoriClient(api_key="k", model="nori-30m")
    assert c.model == "synthefy/nori-30m"
    assert c._local_variant == "nori-30m"


def test_no_bare_nori_slug():
    # The ambiguous bare selectors are gone -- only size-explicit names/slugs resolve.
    assert "nori" not in NORI_VARIANTS
    assert "synthefy/nori" not in NORI_VARIANTS
    assert set(NORI_VARIANTS) >= {
        "nori-6m",
        "nori-30m",
        "nori-100m",
        "synthefy/nori-6m",
        "synthefy/nori-30m",
        "synthefy/nori-100m",
    }


def test_nori_100m_resolves_like_the_other_sizes():
    # The ~98.3M variant is a plain size selector: friendly name and raw slug both map to the
    # nori-100m gateway slug and the nori-100m local checkpoint (no Thinking, no special case).
    c100 = SynthefyNoriClient(api_key="k", model="nori-100m")
    assert c100.model == "synthefy/nori-100m"
    assert c100._local_variant == "nori-100m"

    craw = SynthefyNoriClient(api_key="k", model="synthefy/nori-100m")
    assert craw.model == "synthefy/nori-100m"
    assert craw._local_variant == "nori-100m"

    assert not _is_thinking_model("nori-100m")


def test_nori_100m_is_offered_everywhere_the_size_list_is_derived():
    # NORI_VARIANTS, _MODEL_NAMES and SAGEMAKER_VARIANTS all derive from RELEASED_MODEL_NAMES, so a
    # new size must appear in each without a second edit.
    assert "nori-100m" in _MODEL_NAMES
    assert "nori-100m" in SAGEMAKER_VARIANTS
    assert "nori-30m-thinking-medium" in _MODEL_NAMES
    assert all("/" not in name for name in _MODEL_NAMES)
    # Thinking has no local checkpoint; nori-100m does. Assert the behaviour directly rather
    # than through a derived name list.
    assert _resolve_local_variant("nori-100m") == "nori-100m"
    with pytest.raises(ValueError, match="no local checkpoint"):
        _resolve_local_variant("nori-30m-thinking-medium")


def test_local_mode_passes_nori_100m_to_predict(monkeypatch):
    seen: Dict = {}

    def fake_predict(X_train, y_train, X_test, *, task=None, model="__unset__"):
        seen["model"] = model
        return [0.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: fake_predict)
    client = SynthefyNoriClient(mode="local", model="nori-100m")
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)
    assert seen["model"] == "nori-100m"


def test_remote_body_uses_nori_100m_gateway_slug():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-100m")
    _attach_mock(client, _ok_handler([1.0], capture))
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)
    assert capture["body"]["model"] == "synthefy/nori-100m"


def test_remote_body_uses_variant_gateway_slug():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)
    assert capture["body"]["model"] == "synthefy/nori-30m"


def test_local_mode_passes_variant_to_predict(monkeypatch):
    seen: Dict = {}

    def fake_predict(X_train, y_train, X_test, *, task=None, model="__unset__"):
        seen["model"] = model
        return [0.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: fake_predict)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)
    assert seen["model"] == "nori-30m"


def test_local_mode_nori_6m_forces_base_variant(monkeypatch):
    seen: Dict = {}

    def fake_predict(X_train, y_train, X_test, *, task=None, model="__unset__"):
        seen["model"] = model
        return [0.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: fake_predict)
    client = SynthefyNoriClient(mode="local", model="nori-6m")  # base 6M
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)
    # nori-6m forwards its variant explicitly so local loads the base, not the package's 30M default
    assert seen["model"] == "nori-6m"


def test_local_variant_needs_model_selector_on_old_synthefy_nori(monkeypatch):
    def old_predict(X_train, y_train, X_test, *, task=None):  # no model= param
        return [0.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: old_predict)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    with pytest.raises(ImportError, match="model= selector"):
        client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)


def test_gateway_slug_resolves_to_local_variant():
    # Size-explicit gateway slugs map to the right local checkpoint, not a raw-repo lookup, so slug
    # users get the intended weights locally. There is no bare synthefy/nori.
    cbase = SynthefyNoriClient(api_key="k", model="synthefy/nori-6m")
    assert cbase.model == "synthefy/nori-6m" and cbase._local_variant == "nori-6m"
    c30 = SynthefyNoriClient(api_key="k", model="synthefy/nori-30m")
    assert c30.model == "synthefy/nori-30m" and c30._local_variant == "nori-30m"


# --------------------------------------------------------------------------- #
# Nori Thinking is hosted-API only; no silent fallback to the base model
# --------------------------------------------------------------------------- #


def test_is_thinking_model_matches_released_medium_selector():
    for name in (
        "synthefy/nori-30m-thinking-medium",
        "nori-30m-thinking-medium",
    ):
        assert _is_thinking_model(name)
    for name in ("nori", "nori-6m", "nori-30m", "synthefy/nori", "synthefy/custom", None):
        assert not _is_thinking_model(name)


@pytest.mark.parametrize(
    "model",
    [
        "synthefy/nori-30m-thinking-medium",  # raw gateway slug
        "nori-30m-thinking-medium",
    ],
)
def test_thinking_model_raises_in_local_mode(model):
    # Thinking has no local checkpoint: constructing for local inference must raise a clear
    # error (pointing at mode="remote"), never silently run the base ~6M model.
    with pytest.raises(ValueError, match=r"Thinking.*hosted Synthefy API"):
        SynthefyNoriClient(mode="local", model=model)


def test_thinking_friendly_name_resolves_to_gateway_slug_remote():
    # In remote mode the friendly Thinking name maps to its gateway slug (uniform with nori-30m),
    # so callers never need the raw "synthefy/" prefix.
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m-thinking-medium")  # remote (default)
    assert client.model == "synthefy/nori-30m-thinking-medium"
    _attach_mock(client, _ok_handler([1.0], capture))
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)
    assert capture["body"]["model"] == "synthefy/nori-30m-thinking-medium"


def test_unknown_model_raises_in_local_mode_no_base_fallback():
    # A selector with no local checkpoint (a custom deployment slug) must raise in local mode
    # rather than silently substituting the base model.
    with pytest.raises(ValueError, match=r"no local checkpoint"):
        SynthefyNoriClient(mode="local", model="synthefy/custom")


def test_unknown_model_still_passes_through_in_remote_mode():
    # The local guard is scoped to local mode; a custom slug is still a valid remote gateway id.
    client = SynthefyNoriClient(api_key="k", model="synthefy/custom")
    assert client.model == "synthefy/custom"


# --------------------------------------------------------------------------- #
# Categorical-target discretization (discretize= / categorical_levels=)
# --------------------------------------------------------------------------- #


def _discretized_handler(predictions, strategy, capture):
    """A deployment that discretized: levels plus the echo naming what ran."""

    def handler(request: httpx.Request) -> httpx.Response:
        capture["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "task": "regression",
                "predictions": predictions,
                "discretize": strategy,
            },
        )

    return handler


def test_remote_returns_the_levels_the_deployment_chose():
    """The client passes the server's answer through rather than re-deriving it.

    It used to snap the returned means itself, which only ever worked for snap-mean.
    """
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _discretized_handler([1.0, 2.0, 2.0], "snap-mean", capture))
    preds = client.predict(
        X_train=[[0.0], [1.0], [2.0]],
        y_train=[1.0, 1.0, 2.0],
        X_test=[[0.5], [1.5], [2.5]],
        discretize="snap-mean",
    )
    assert preds == [1.0, 2.0, 2.0]
    assert capture["body"]["discretize"] == "snap-mean"


def test_remote_discretization_survives_the_pandas_wrapper():
    """The as_pandas path must not reshape or re-round what came back."""
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _discretized_handler([1.0, 2.0], "snap-mean", capture))
    preds = client.predict(
        X_train=pd.DataFrame({"a": [0.0, 1.0, 2.0]}),
        y_train=pd.Series([1.0, 1.0, 2.0], name="rating"),
        X_test=pd.DataFrame({"a": [0.5, 1.5]}, index=[10, 20]),
        as_pandas=True,
        discretize="snap-mean",
    )
    assert isinstance(preds, pd.Series)
    assert preds.name == "rating"
    assert list(preds.index) == [10, 20]
    assert preds.tolist() == [1.0, 2.0]


def test_a_deployment_that_ignored_discretize_raises_instead_of_returning_means():
    """The echo is the only evidence a level was ever chosen.

    A deployment predating the field drops it and answers with continuous predictions,
    which are numerically valid and look like any other result. Returning those as if
    they were levels is the failure the echo exists to prevent.
    """
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _ok_handler([0.9, 1.6], {}))  # no discretize key
    with pytest.raises(ValueError, match="predates the capability"):
        client.predict(
            X_train=[[0.0], [1.0]],
            y_train=[1.0, 2.0],
            X_test=[[0.5], [1.5]],
            discretize="snap-mean",
        )


def test_a_normalized_echo_is_not_mistaken_for_a_different_strategy():
    """The server lowercases the name and echoes the resolved one, as it does for
    output_type. Comparing against the raw string rejected a correct response."""
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _discretized_handler([1.0, 2.0], "map-cell", {}))
    preds = client.predict(
        X_train=[[0.0], [1.0]],
        y_train=[1.0, 2.0],
        X_test=[[0.5], [1.5]],
        discretize="MAP-CELL",
    )
    assert preds == [1.0, 2.0]


def test_a_deployment_that_ran_another_strategy_raises():
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _discretized_handler([1.0, 2.0], "map-cell", {}))
    with pytest.raises(ValueError, match="ran 'map-cell' instead"):
        client.predict(
            X_train=[[0.0], [1.0]],
            y_train=[1.0, 2.0],
            X_test=[[0.5], [1.5]],
            discretize="snap-mean",
        )


def test_levels_alone_accept_whichever_default_the_deployment_resolved():
    """The caller named no strategy, so any echoed one is the server's to choose."""
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _discretized_handler([1.0, 2.0], "map-cell", {}))
    preds = client.predict(
        X_train=[[0.0], [1.0]],
        y_train=[1.0, 2.0],
        X_test=[[0.5], [1.5]],
        categorical_levels=[1, 2],
    )
    assert preds == [1.0, 2.0]


def test_remote_sends_the_strategy_rather_than_snapping_locally():
    """The server owns the arithmetic, so the client's job is to forward, not compute.

    Every strategy needs the model's own predictive distribution, which only the
    deployment has without the caller installing torch. The client used to refuse all
    but ``snap-mean`` and snap the returned means itself; now it sends the request.
    """
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _discretized_handler([1.0], "map-cell", capture))

    client.predict(
        X_train=_XTR,
        y_train=_YTR,
        X_test=_XTE,
        discretize="map-cell",
        categorical_levels=[3, 1, 2],
    )

    assert capture["body"]["discretize"] == "map-cell"
    assert capture["body"]["categorical_levels"] == [3.0, 1.0, 2.0]


def test_remote_sends_levels_without_a_strategy():
    """Levels alone are meaningful: the server resolves them to its default strategy.

    Refusing here would re-impose a restriction the server does not have, and the
    response's ``discretize`` echo is what tells the caller which default ran.
    """
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _discretized_handler([1.0], "map-cell", capture))

    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE, categorical_levels=[1, 2])

    assert "discretize" not in capture["body"]
    assert capture["body"]["categorical_levels"] == [1.0, 2.0]


def test_a_request_that_does_not_discretize_carries_neither_field():
    """An unused optional field must not appear, or older deployments see a new key."""
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))

    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)

    assert "discretize" not in capture["body"]
    assert "categorical_levels" not in capture["body"]


@pytest.mark.parametrize(
    "levels, phrase",
    [
        ([1.0, float("nan")], "finite"),
        (["a", "b"], "must be numbers"),
        ([], "empty"),
    ],
)
def test_unsendable_levels_name_the_argument_they_came_from(levels, phrase):
    """The deployment owns the policy; the client owns being able to serialize the call.

    Without this, a NaN level reached the payload guard and was reported as a problem with
    `y_train`, and a text level surfaced as a raw numpy conversion error.
    """
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))
    with pytest.raises(ValueError, match=phrase) as exc:
        client.predict(
            X_train=_XTR,
            y_train=_YTR,
            X_test=_XTE,
            discretize="snap-mean",
            categorical_levels=levels,
        )
    assert "categorical_levels" in str(exc.value)
    assert "body" not in capture


def test_discretize_still_refuses_a_distribution_output_type_before_the_request():
    """The one check that stays client-side, because it needs no server to decide.

    The strategy chooses the summary, so pairing it with output_type is a contradiction.
    The server rejects it too; failing here saves a paid round-trip.
    """
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))
    with pytest.raises(ValueError, match="discretize"):
        client.predict(
            X_train=_XTR,
            y_train=_YTR,
            X_test=_XTE,
            discretize="map-cell",
            output_type="quantiles",
            quantiles=[0.1, 0.9],
        )
    assert "body" not in capture


def test_local_mode_passes_discretize_and_levels(monkeypatch):
    seen: Dict = {}

    def fake_predict(X_train, y_train, X_test, *, task=None, model=None, **kwargs):
        assert model == "nori-30m"
        seen.update(kwargs)
        return [1.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: fake_predict)
    monkeypatch.setattr("synthefy.nori_client._local_discretize_available", lambda: True)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    client.predict(
        X_train=_XTR,
        y_train=_YTR,
        X_test=_XTE,
        discretize="median-cell",
        categorical_levels=[1, 2],
    )
    assert seen["discretize"] == "median-cell"
    assert seen["categorical_levels"] == [1, 2]


def test_local_mode_levels_alone_activate_package_default(monkeypatch):
    seen: Dict = {}

    def fake_predict(X_train, y_train, X_test, *, task=None, model=None, **kwargs):
        assert model == "nori-30m"
        seen.update(kwargs)
        return [1.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: fake_predict)
    monkeypatch.setattr("synthefy.nori_client._local_discretize_available", lambda: True)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE, categorical_levels=[1, 2])
    assert "discretize" not in seen  # package picks its own default (map-cell)
    assert seen["categorical_levels"] == [1, 2]


def test_local_mode_without_discretize_sends_only_required_model(monkeypatch):
    def strict_predict(X_train, y_train, X_test, *, task=None, model=None):
        assert model == "nori-30m"
        return [1.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: strict_predict)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    assert client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE) == [1.0]


def test_local_mode_preserves_degradation_warning_and_message(monkeypatch):
    class LocalDegradationWarning(UserWarning):
        pass

    message = "Nori: SVD fit failed -> passthrough of all 400 raw columns"

    def degraded_predict(X_train, y_train, X_test, *, task=None, model=None):
        warnings.warn(message, LocalDegradationWarning)
        return [1.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: degraded_predict)
    client = SynthefyNoriClient(mode="local", model="nori-30m")

    with pytest.warns(LocalDegradationWarning, match="SVD fit failed") as caught:
        assert client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE) == [1.0]

    assert str(caught[0].message) == message


def test_local_mode_preserves_strict_degradation_error_and_message(monkeypatch):
    class LocalDegradationWarning(UserWarning):
        pass

    message = "Nori: SVD transform failed -> a single all-zero column"

    def degraded_predict(X_train, y_train, X_test, *, task=None, model=None):
        warnings.warn(message, LocalDegradationWarning)
        return [1.0]

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: degraded_predict)
    client = SynthefyNoriClient(mode="local", model="nori-30m")

    with warnings.catch_warnings():
        # synthefy_nori.strict_pipeline(SvdFallbackWarning) uses this same standard-library
        # filter mechanism. The client must not catch, wrap, or rewrite the resulting exception.
        warnings.simplefilter("error", LocalDegradationWarning)
        with pytest.raises(LocalDegradationWarning, match="SVD transform failed") as caught:
            client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE)

    assert str(caught.value) == message


def test_local_discretize_needs_newer_synthefy_nori(monkeypatch):
    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: lambda *a, **k: [1.0])
    monkeypatch.setattr("synthefy.nori_client._local_discretize_available", lambda: False)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    with pytest.raises(ImportError, match=r"synthefy-nori"):
        client.predict(X_train=_XTR, y_train=_YTR, X_test=_XTE, discretize="map-cell")


@pytest.mark.slow
def test_local_discretize_real_inference():
    pytest.importorskip("synthefy_nori.discretize")

    client = SynthefyNoriClient(mode="local", model="nori-30m")
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 3))
    y_train = np.clip(np.round(X_train[:, 0] * 2.0 + 3.0), 1, 5)
    X_test = rng.normal(size=(5, 3))

    preds = client.predict(X_train, y_train, X_test, discretize="map-cell")
    levels = set(np.unique(y_train).tolist())
    assert all(p in levels for p in preds)


# --------------------------------------------------------------------------- #
# Text features -- client-side embedding (text_columns=...)
# --------------------------------------------------------------------------- #


def _fake_torch(*, cuda_available=False, mps_available=False):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps_available)),
    )


def test_text_device_auto_prefers_cuda_over_mps(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(cuda_available=True, mps_available=True),
    )

    assert _resolve_text_device("auto") == "cuda"


def test_text_device_auto_uses_mps_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=True))

    assert _resolve_text_device(None) == "mps"


def test_text_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())

    assert _resolve_text_device("auto") == "cpu"


def test_text_device_explicit_override_skips_auto_detection(monkeypatch):
    def unexpected_probe():
        pytest.fail("explicit text_device should not probe accelerator availability")

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=unexpected_probe),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=unexpected_probe)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _resolve_text_device("cuda:1") == "cuda:1"


@pytest.mark.parametrize("device", ["", "  ", 1])
def test_text_device_rejects_invalid_explicit_override(device):
    with pytest.raises(ValueError, match="text_device"):
        _resolve_text_device(device)


@requires_text_runtime
def test_text_device_is_forwarded_to_multimodal_preprocessor(monkeypatch):
    capture = {}

    class FakePreprocessor:
        def __init__(self, text_columns, **kwargs):
            capture["text_columns"] = text_columns
            capture.update(kwargs)

        def fit_transform(self, frame):
            return np.zeros((len(frame), 1), dtype=np.float32)

        def transform(self, frame):
            return np.zeros((len(frame), 1), dtype=np.float32)

    monkeypatch.setattr("synthefy.text_features.MultimodalPreprocessor", FakePreprocessor)
    train = pd.DataFrame({"review": ["good", "bad"]})
    test = pd.DataFrame({"review": ["fine"]})

    _widen_text_columns(
        train,
        test,
        ["review"],
        8,
        "minilm",
        100,
        "ordinal",
        "cuda:1",
    )

    assert capture["text_columns"] == ["review"]
    assert capture["device"] == "cuda:1"


def _fake_embed(texts):
    """Deterministic 8-d embedding, so tests need no sentence-transformers/model."""
    import hashlib

    out = []
    for t in texts:
        h = hashlib.sha1(t.encode("utf-8")).digest()
        out.append(np.frombuffer(h[:8], dtype=np.uint8).astype(np.float32) / 255.0)
    return np.stack(out)


class _FakePreloadedEncoder:
    def encode(self, texts, **kwargs):
        return _fake_embed(texts)


@pytest.mark.parametrize(
    "embedder",
    [_fake_embed, _FakePreloadedEncoder()],
    ids=["callable", "preloaded"],
)
@requires_text_runtime
def test_text_device_is_ignored_for_custom_encoder(embedder):
    train = pd.DataFrame({"review": ["good", "bad", "great", "awful"]})
    test = pd.DataFrame({"review": ["fine"]})

    train_features, test_features = _widen_text_columns(
        train,
        test,
        ["review"],
        2,
        embedder,
        100,
        "ordinal",
        "",
    )

    assert train_features.shape == (4, 2)
    assert test_features.shape == (1, 2)


@requires_text_runtime
def test_text_columns_embeds_client_side_and_sends_numeric():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0, 2.0], capture))

    df_train = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "brand": ["a", "b", "a", "b", "a", "b"],  # categorical -> 1 col
            "review": ["good", "bad", "ok", "great", "poor", "fine"],  # text -> SVD
        }
    )
    df_test = pd.DataFrame({"x1": [1.5, 5.5], "brand": ["a", "b"], "review": ["nice", "awful"]})

    preds = client.predict(
        df_train, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], df_test, text_columns=["review"], svd_dim=4, embedder=_fake_embed
    )

    assert preds == [1.0, 2.0]
    # x1 (numeric) + brand (1 categorical col) + 4 SVD text cols = 6 numeric features
    assert len(capture["body"]["X_train"][0]) == 6
    assert len(capture["body"]["X_test"][0]) == 6
    # the payload is fully numeric (text was embedded away client-side)
    assert all(isinstance(v, (int, float)) for row in capture["body"]["X_test"] for v in row)


@requires_text_runtime
def test_text_columns_accepts_a_pandas_index_declaration():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))
    train = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "review": ["good", "bad", "fine"]})
    test = pd.DataFrame({"review": ["new"], "amount": [4.0]})

    client.predict(
        train,
        [1.0, 2.0, 3.0],
        test,
        text_columns=pd.Index(["review"]),
        svd_dim=2,
        embedder=_fake_embed,
    )

    assert len(capture["body"]["X_train"][0]) == 3


@requires_text_runtime
def test_text_columns_respect_onehot_categorical_encoding():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0, 2.0], capture))

    df_train = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "brand": ["a", "b", "a", "b", "a", "b"],
            "review": ["good", "bad", "ok", "great", "poor", "fine"],
        }
    )
    df_test = pd.DataFrame(
        {
            "review": ["nice", "awful"],
            "brand": ["a", "new"],
            "x1": [1.5, 5.5],
        }
    )

    client.predict(
        df_train,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        df_test,
        text_columns=["review"],
        svd_dim=4,
        embedder=_fake_embed,
        categorical_encoding="onehot",
    )

    # x1 + brand_a + brand_b + four text-SVD features.
    assert len(capture["body"]["X_train"][0]) == 7
    assert capture["body"]["X_test"][0][:3] == [1.5, 1.0, 0.0]
    assert capture["body"]["X_test"][1][:3] == [5.5, 0.0, 0.0]


@requires_text_runtime
def test_text_columns_require_a_choice_for_high_cardinality_tabular_columns():
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], capture))
    df_train = pd.DataFrame(
        {
            "row_id": [f"id-{index}" for index in range(6)],
            "review": ["good", "bad", "ok", "great", "poor", "fine"],
        }
    )
    df_test = pd.DataFrame({"row_id": ["new-id"], "review": ["nice"]})

    with pytest.raises(ValueError, match="Automatic handling is ambiguous"):
        client.predict(
            df_train,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            df_test,
            text_columns=["review"],
            svd_dim=4,
            embedder=_fake_embed,
            max_categorical_cardinality=2,
        )


def test_text_columns_requires_dataframe():
    client = SynthefyNoriClient(api_key="test-key", model="nori-30m")
    _attach_mock(client, _ok_handler([1.0], {}))
    with pytest.raises(ValueError):
        client.predict([[1.0, 2.0]], [1.0], [[1.0, 2.0]], text_columns=["review"])


# ------------------------------------------------------------------ memory policy
# `memory_policy=` is the serving-memory policy, at parity with the local package. The wire half is
# what these cover: that it reaches the request only when asked for, that what the server
# reports comes back to the caller, and that a deployment which IGNORES it is treated as a
# failure rather than as success.
def _memory_handler(capture: Dict, report: Optional[Dict] = None) -> Handler:
    """Mock endpoint that echoes a memory_report, as a supporting deployment does."""

    def handler(request: httpx.Request) -> httpx.Response:
        capture["body"] = json.loads(request.content)
        body: Dict[str, object] = {"task": "regression", "predictions": [1.0, 2.0]}
        if report is not None:
            body["memory_report"] = report
        return httpx.Response(200, json=body)

    return handler


# --------------------------------------------------------------------------- #
# Distribution output (output_type= / quantiles=)
# --------------------------------------------------------------------------- #

_LEVELS = [0.1, 0.5, 0.9]


def _dist_handler(
    capture: Dict,
    *,
    predictions: List[float],
    quantiles=None,
    taus=None,
    output_type: str = "quantiles",
    echo_output_type: bool = True,
    memory_report=None,
) -> Handler:
    """Mock a deployment that serves distribution output.

    ``echo_output_type=False`` simulates a deployment that predates distribution
    output: it ignores the request's ``output_type`` and answers with means.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        capture["body"] = json.loads(request.content)
        body: Dict = {"task": "regression", "predictions": predictions}
        if echo_output_type:
            body["output_type"] = output_type
        if quantiles is not None:
            body["quantiles"] = quantiles
        if taus is not None:
            body["taus"] = taus
        if memory_report is not None:
            body["memory_report"] = memory_report
        return httpx.Response(200, json=body)

    return handler


_REPORT = {
    "rung": "resident_int8",
    "est_cache_gb": 0.0122,
    "resident_gb": 0.0065,
    "query_chunk": 256,
    "dropped_context_rows": 0,
    "attempt_history": [
        {
            "pipeline_ids": [0],
            "path": "cached",
            "rung": "resident_int8",
            "cache_dtype": "int8",
            "offload_to_host": False,
            "context_row_chunk": None,
            "query_chunk": None,
            "outcome": "success",
            "reason": "resolved",
            "dropped_context_rows": 0,
        }
    ],
    "clamped": [],
    "notes": [],
}

_X_TRAIN = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
_Y_TRAIN = [0.0, 1.0, 0.5]
_X_TEST = [[0.2, 0.8], [0.9, 0.1]]


def _client_with(handler: Handler) -> SynthefyNoriClient:
    client = SynthefyNoriClient(api_key="k", model="nori-30m", mode="remote")
    _attach_mock(client, handler)
    return client


def test_a_request_without_memory_does_not_send_the_field():
    """The default request must stay byte-for-byte what it was before this feature.

    Not merely "memory is null": the hosted schema declares the field as a preset name or an
    object and forbids unknown properties, so an explicit null is a different request.
    """
    capture: Dict = {}
    client = _client_with(_memory_handler(capture))
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    assert "memory_policy" not in capture["body"]
    assert set(capture["body"]) == {"X_train", "y_train", "X_test", "task", "model"}
    assert client.last_memory_report is None


@pytest.mark.parametrize(
    "policy",
    [
        "exact",
        "max_context",
        "off",
        {"cache_dtype": "int8"},
        {"elements_budget": 4000},
        {"stream_context": True},
    ],
    ids=["exact", "max_context", "off", "dict", "elements_budget", "stream_context"],
)
def test_a_policy_is_sent_verbatim(policy):
    capture: Dict = {}
    client = _client_with(_memory_handler(capture, _REPORT))
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy=policy)
    assert capture["body"]["memory_policy"] == policy


def test_the_servers_report_reaches_the_caller():
    """The rung depends on the replica's free VRAM, so the response is the only source."""
    client = _client_with(_memory_handler({}, _REPORT))
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy={"cache_dtype": "int8"})
    # Validated through MemoryReport, exposed as a dict (as the library does).
    assert client.last_memory_report["rung"] == "resident_int8"
    assert client.last_memory_report["est_cache_gb"] == _REPORT["est_cache_gb"]


def test_a_deployment_that_ignores_the_policy_is_an_error_not_a_success():
    """The capability handshake, and the reason the server echoes at all.

    A deployment predating `memory` drops the field and returns default-memory predictions
    that are numerically valid — nothing in `predictions` reveals the policy was ignored. So
    a missing report has to be surfaced, or the caller believes something took effect that
    did not.
    """
    client = _client_with(_memory_handler({}))  # no memory_report in the response
    with pytest.raises(ValueError, match="did not report back"):
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy="exact")


def test_the_report_is_cleared_between_calls():
    """A stale report must not be readable as belonging to the call that just ran."""
    capture: Dict = {}
    client = SynthefyNoriClient(api_key="k", model="nori-30m", mode="remote")
    _attach_mock(client, _memory_handler(capture, _REPORT))
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy="exact")
    assert client.last_memory_report is not None

    _attach_mock(client, _memory_handler(capture))  # a call that sets no policy
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    assert client.last_memory_report is None, "the previous call's report leaked"


def test_the_server_rejection_message_is_surfaced_unchanged():
    """For rules the client does NOT copy, the server's own message must come through.

    Field-level mistakes (unknown name, bad type, out of bounds) are now caught locally by
    MemoryPolicy. What stays server-side is which COMBINATIONS are incoherent -- deliberately
    not duplicated, because a second copy of behaviour would drift in a way a schema comparison
    cannot see. This exercises one of those: cache=False with a cache-only field.
    """
    detail = (
        "Invalid 'memory_policy': 1 validation error for MemoryPolicy\n  Value error, "
        "cache=False cannot be combined with cache_dtype"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"detail": detail}})

    client = _client_with(handler)
    with pytest.raises(BadRequestError) as excinfo:
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy={"cache": False, "cache_dtype": "int8"})
    assert "cannot be combined" in str(excinfo.value)


def test_an_unknown_field_is_now_caught_locally_without_a_round_trip():
    """The client's model forbids extras, so a typo does not cost a request."""
    client = _client_with(_memory_handler({}, _REPORT))
    with pytest.raises(ValueError, match="int8"):
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy={"int8": True})


def test_the_request_model_accepts_both_shapes():
    from synthefy import MemoryPolicy, NoriPredictRequest

    assert (
        NoriPredictRequest(X_train=_X_TRAIN, y_train=_Y_TRAIN, X_test=_X_TEST, memory_policy="exact").memory_policy
        == "exact"
    )
    # A dict is COERCED into the typed model by pydantic, which is the point of the field
    # being MemoryPolicy: bounds, enums and unknown-field rejection happen before any request.
    coerced = NoriPredictRequest(
        X_train=_X_TRAIN,
        y_train=_Y_TRAIN,
        X_test=_X_TEST,
        memory_policy={"cache_dtype": "int8"},
    ).memory_policy
    assert isinstance(coerced, MemoryPolicy)
    assert coerced.cache_dtype == "int8"
    # ...and only the field that was set is carried, so the server's defaults still apply.
    assert coerced.model_dump(exclude_unset=True) == {"cache_dtype": "int8"}
    assert NoriPredictRequest(
        X_train=_X_TRAIN,
        y_train=_Y_TRAIN,
        X_test=_X_TEST,
        memory_policy={"cache_dtype": "int8"},
    ).to_wire()["memory_policy"] == {"cache_dtype": "int8"}
    # Unset by default, so the field cannot change an existing caller's payload.
    assert NoriPredictRequest(X_train=_X_TRAIN, y_train=_Y_TRAIN, X_test=_X_TEST).memory_policy is None


def test_local_mode_refuses_memory_on_an_old_synthefy_nori(monkeypatch):
    """An opaque TypeError from deep inside the library is not an acceptable answer."""
    from synthefy import nori_client as module

    class OldRegressor:
        def __init__(self, model=None):
            self.model = model

        def predict(self, X, *, output_type="mean"):
            return [0.0] * len(X)

    monkeypatch.setattr(module, "_local_memory_policy_available", lambda: False)
    monkeypatch.setattr(module, "_load_local_regressor", lambda: OldRegressor)
    client = SynthefyNoriClient(model="nori-30m", mode="local")
    with pytest.raises(ImportError, match="0.13.0"):
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy="exact")


@pytest.mark.parametrize("enabled", [True, False])
def test_local_mode_refuses_stream_field_when_installed_policy_lacks_it(monkeypatch, enabled):
    """Even explicit False is forwarded, so an older policy would reject either value."""
    from synthefy import nori_client as module

    initialized = []

    class PreStreamingRegressor:
        def __init__(self, model=None, memory_policy=None):
            initialized.append((model, memory_policy))

        def predict(self, X, *, output_type="mean"):
            return [0.0] * len(X)

    monkeypatch.setattr(module, "_local_memory_policy_available", lambda: True)
    monkeypatch.setattr(module, "_local_stream_context_available", lambda: False)
    monkeypatch.setattr(module, "_load_local_regressor", lambda: PreStreamingRegressor)

    client = SynthefyNoriClient(model="nori-30m", mode="local")
    with pytest.raises(ImportError, match="MemoryPolicy does not include.*stream_context"):
        client.predict(
            _X_TRAIN,
            _Y_TRAIN,
            _X_TEST,
            memory_policy={"stream_context": enabled},
        )
    assert initialized == [], "capability guard must run before estimator construction"


def test_local_mode_forwards_stream_context_when_installed_policy_supports_it(monkeypatch):
    from synthefy import nori_client as module

    seen: Dict = {}

    class StreamingRegressor:
        def __init__(self, model=None, memory_policy=None):
            seen["memory_policy"] = memory_policy
            self.memory_policy = memory_policy
            self.memory_report_ = None

        def fit(self, X, y):
            return self

        def predict(self, X, *, output_type="mean", quantiles=None):
            self.memory_report_ = {
                **_REPORT,
                "rung": "stream_bf16",
                "stream_context": True,
                "context_row_chunk": 2048,
                "reuse_context_cache": False,
            }
            return [0.1] * len(X)

    monkeypatch.setattr(module, "_local_memory_policy_available", lambda: True)
    monkeypatch.setattr(module, "_local_stream_context_available", lambda: True)
    monkeypatch.setattr(module, "_load_local_regressor", lambda: StreamingRegressor)

    client = SynthefyNoriClient(model="nori-30m", mode="local")
    client.predict(
        _X_TRAIN,
        _Y_TRAIN,
        _X_TEST,
        memory_policy={"stream_context": True},
    )

    assert seen["memory_policy"] == {"stream_context": True}
    assert client.last_memory_report["rung"] == "stream_bf16"
    assert client.last_memory_report["context_row_chunk"] == 2048


def test_local_mode_uses_estimator_and_surfaces_memory_report(monkeypatch):
    from synthefy import nori_client as module

    seen: Dict = {}

    class FakeRegressor:
        def __init__(self, model=None, memory_policy=None):
            seen["model"] = model
            seen["memory_policy"] = memory_policy
            self.memory_policy = memory_policy
            self.memory_report_ = None

        def fit(self, X, y):
            seen["fit"] = (X, y)
            return self

        def predict(self, X, *, output_type="mean", quantiles=None, **_kwargs):
            self.memory_report_ = dict(_REPORT)
            return [0.1, 0.2]

    monkeypatch.setattr(module, "_local_memory_policy_available", lambda: True)
    monkeypatch.setattr(module, "_load_local_regressor", lambda: FakeRegressor)
    monkeypatch.setattr(
        module,
        "_load_local_predict",
        lambda: (_ for _ in ()).throw(AssertionError("functional path used")),
    )
    client = SynthefyNoriClient(model="nori-30m", mode="local")
    predictions = client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy={"cache_dtype": "int8"})

    assert predictions == [0.1, 0.2]
    assert seen["model"] == "nori-30m"
    assert seen["memory_policy"] == {"cache_dtype": "int8"}
    assert client.last_memory_report["rung"] == "resident_int8"
    assert client.last_memory_report["attempt_history"][0]["outcome"] == "success"


# --------------------------------- the model in synthefy-nori IS the schema
class _FakePolicy:
    """Stands in for ANOTHER package's MemoryPolicy — same name, different class.

    pydantic refuses a foreign class outright, with a message that is misleading to someone
    holding exactly what it says it wants, so the annotation carries a BeforeValidator that
    dumps anything model_dump-shaped. Duck-typed here for the same reason it is there: so the
    client needs no dependency on the model package.

    Duck-typed on purpose — importing the real model would make a thin API client depend on
    the model package (and its torch tree) just to name a shape.
    """

    def __init__(self, dumped):
        self._dumped = dumped

    def model_dump(self, **kwargs):
        # The client asks for exclude_unset: carry only what the caller actually set, so the
        # server's defaults apply to everything else.
        if kwargs.get("exclude_unset"):
            return {k: v for k, v in self._dumped.items() if v is not None}
        return dict(self._dumped)


def test_a_policy_object_is_serialised_for_the_wire():
    capture: Dict = {}
    client = _client_with(_memory_handler(capture, _REPORT))
    # As a real MemoryPolicy looks: inputs set, resolve()'s outputs still None.
    policy = _FakePolicy(
        {
            "cache": True,
            "cache_dtype": "int8",
            "gpu_budget_absolute_gb": None,
            "rung": None,
            "est_cache_gb": None,
        }
    )
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy=policy)
    sent = capture["body"]["memory_policy"]
    assert sent == {"cache": True, "cache_dtype": "int8"}
    # The decided-output fields must not be sent: the server rejects a policy carrying a rung,
    # because re-using resolved values as configuration skips every coherence check.
    assert "rung" not in sent and "est_cache_gb" not in sent


def test_feeding_a_resolved_report_back_in_is_caught_locally():
    """The decided fields live on MemoryReport, not MemoryPolicy, so this fails before the call.

    The server rejects it too — re-using decided outputs as configuration would skip every
    coherence check — but the client's model has no `rung` field and forbids extras, so the
    mistake costs no round trip.
    """
    client = _client_with(_memory_handler({}, _REPORT))
    with pytest.raises(ValueError, match="rung"):
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy={"cache": True, "rung": "resident_bf16"})


def test_a_nonsense_memory_policy_type_is_rejected_locally():
    """One error type for every bad policy, from pydantic, before any request is sent."""
    client = _client_with(_memory_handler({}, _REPORT))
    with pytest.raises(ValueError):  # pydantic ValidationError subclasses ValueError
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy=object())


def test_an_unknown_preset_is_caught_locally():
    """The preset is a Literal, not a bare str, so a typo never reaches the network.

    It used to: the field accepted any string, so "aggressive" was sent and came back a 400.
    """
    client = _client_with(_memory_handler({}, _REPORT))
    with pytest.raises(ValueError, match="aggressive"):
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy="aggressive")


def test_out_of_range_values_are_caught_locally():
    client = _client_with(_memory_handler({}, _REPORT))
    with pytest.raises(ValueError):
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy={"gpu_budget_frac": 1.5})


@pytest.mark.skipif(
    importlib.util.find_spec("synthefy_nori") is None
    or importlib.util.find_spec("synthefy_nori.inference.memory_policy") is None,
    reason="needs synthefy-nori >= 0.13.0, which is the version that has MemoryPolicy",
)
def test_the_real_memory_policy_round_trips_through_the_client():
    """Drift guard: the actual model must survive the client's normalisation.

    Skips without synthefy-nori (an optional extra), so it runs wherever the two are installed
    together — which is where a divergence would matter.
    """
    from synthefy_nori.inference.memory_policy import MemoryPolicy

    capture: Dict = {}
    client = _client_with(_memory_handler(capture, _REPORT))
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, memory_policy=MemoryPolicy(cache_dtype="int8", gpu_budget_frac=0.6))
    sent = capture["body"]["memory_policy"]
    assert sent["cache_dtype"] == "int8" and sent["gpu_budget_frac"] == 0.6
    assert "rung" not in sent, "an unresolved policy must not carry decided outputs"
    # And what we send back must be something the model itself accepts, i.e. a real round trip.
    assert MemoryPolicy(**sent).cache_dtype == "int8"


# -------------------------------------------------------------- large context
_LARGE_CONTEXT_REPORT = {
    "applied": True,
    "policy": "cluster_route",
    "threshold": 1,
    "seed": 7,
    "reason": None,
    "window": 2,
    "n_train": 3,
    "n_test": 2,
    "shards_available": 1,
    "nori_calls": 2,
    "full_context": False,
    "reused_train_state": False,
    "gate_winner": None,
}


def _large_context_handler(capture: Dict, report: Optional[Dict] = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        capture["body"] = json.loads(request.content)
        body = {"task": "regression", "predictions": [0.1, 0.2]}
        if report is not None:
            body["large_context_report"] = report
        return httpx.Response(200, json=body)

    return handler


def test_large_context_wire_and_report_round_trip():
    capture: Dict = {}
    client = _client_with(_large_context_handler(capture, _LARGE_CONTEXT_REPORT))
    predictions = client.predict(
        _X_TRAIN,
        _Y_TRAIN,
        _X_TEST,
        large_context_policy="cluster_route",
        large_context_threshold=1,
        large_context_seed=7,
    )
    assert predictions == [0.1, 0.2]
    assert capture["body"]["large_context_policy"] == "cluster_route"
    assert capture["body"]["large_context_threshold"] == 1
    assert capture["body"]["large_context_seed"] == 7
    assert client.last_large_context_report == _LARGE_CONTEXT_REPORT


def test_large_context_defaults_are_sent_only_when_a_policy_is_requested():
    capture: Dict = {}
    report = {
        **_LARGE_CONTEXT_REPORT,
        "threshold": 50_000,
        "seed": 0,
    }
    client = _client_with(_large_context_handler(capture, report))
    client.predict(
        _X_TRAIN,
        _Y_TRAIN,
        _X_TEST,
        large_context_policy="cluster_route",
    )
    assert capture["body"]["large_context_threshold"] == 50_000
    assert capture["body"]["large_context_seed"] == 0

    capture.clear()
    _attach_mock(client, _large_context_handler(capture))
    client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST)
    assert not {
        "large_context_policy",
        "large_context_threshold",
        "large_context_seed",
    } & set(capture["body"])
    assert client.last_large_context_report is None


def test_missing_large_context_report_fails_the_capability_handshake():
    client = _client_with(_large_context_handler({}))
    with pytest.raises(ValueError, match="omitted large_context_report"):
        client.predict(
            _X_TRAIN,
            _Y_TRAIN,
            _X_TEST,
            large_context_policy="cluster_route",
            large_context_threshold=1,
            large_context_seed=7,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threshold", 2),
        ("seed", 8),
        ("policy", "random"),
    ],
)
def test_mismatched_large_context_report_fails_closed(field, value):
    report = dict(_LARGE_CONTEXT_REPORT)
    report[field] = value
    client = _client_with(_large_context_handler({}, report))
    with pytest.raises(ValueError, match="mismatched large_context_report"):
        client.predict(
            _X_TRAIN,
            _Y_TRAIN,
            _X_TEST,
            large_context_policy="cluster_route",
            large_context_threshold=1,
            large_context_seed=7,
        )


@pytest.mark.parametrize(
    "policy",
    [
        "boost",
        "safeboost",
        "cluster_route[groups=16]",
    ],
)
def test_client_forwards_large_context_policy_strings_unchanged(policy):
    capture: Dict = {}
    report = {
        **_LARGE_CONTEXT_REPORT,
        "policy": policy,
        "threshold": 50_000,
        "seed": 0,
    }
    client = _client_with(_large_context_handler(capture, report))
    client.predict(
        _X_TRAIN,
        _Y_TRAIN,
        _X_TEST,
        large_context_policy=policy,
    )
    assert capture["body"]["large_context_policy"] == policy
    assert client.last_large_context_report["policy"] == policy


def test_remote_client_rejects_callable_large_context_policy_before_network():
    calls = {"count": 0}

    def custom_policy(_problem, _rng):
        pass

    def handler(_request):
        calls["count"] += 1
        return httpx.Response(500)

    client = _client_with(handler)
    with pytest.raises(ValueError, match="policy name string"):
        client.predict(
            _X_TRAIN,
            _Y_TRAIN,
            _X_TEST,
            large_context_policy=custom_policy,
        )
    assert calls["count"] == 0


def test_remote_client_forwards_a_tail_gated_policy_list_and_verifies_report():
    capture: Dict = {}
    policies = ["target_rank[cap=32768]", "target_rank[cap=65536]"]
    report = {
        **_LARGE_CONTEXT_REPORT,
        "policy": "gate[target_rank[cap=32768],target_rank[cap=65536]]",
        "threshold": 50_000,
        "seed": 0,
        "holdout_strategy": "tail",
    }
    client = _client_with(_large_context_handler(capture, report))
    client.predict(
        _X_TRAIN,
        _Y_TRAIN,
        _X_TEST,
        large_context_policy=policies,
        large_context_holdout="tail",
    )
    assert capture["body"]["large_context_policy"] == policies
    assert capture["body"]["large_context_holdout"] == "tail"
    assert client.last_large_context_report == report


@pytest.mark.parametrize("policy", [[], [""], ["random", 1], ["random"] * 9])
def test_remote_client_rejects_invalid_policy_lists_before_network(policy):
    client = _client_with(lambda _request: pytest.fail("network called"))
    with pytest.raises(ValueError, match="large_context_policy"):
        client.predict(_X_TRAIN, _Y_TRAIN, _X_TEST, large_context_policy=policy)


@pytest.mark.parametrize("output_type", ["quantiles", "full"])
def test_client_rejects_large_context_distribution_before_network(output_type):
    client = _client_with(lambda _request: httpx.Response(500))
    with pytest.raises(ValueError, match="no combined predictive distribution"):
        client.predict(
            _X_TRAIN,
            _Y_TRAIN,
            _X_TEST,
            output_type=output_type,
            quantiles=[0.1, 0.9] if output_type == "quantiles" else None,
            large_context_policy="cluster_route",
        )


def test_client_rejects_large_context_on_thinking_before_network():
    client = SynthefyNoriClient(
        api_key="k",
        model="nori-30m-thinking-medium",
        mode="remote",
    )
    _attach_mock(client, lambda _request: httpx.Response(500))
    with pytest.raises(ValueError, match="Thinking"):
        client.predict(
            _X_TRAIN,
            _Y_TRAIN,
            _X_TEST,
            large_context_policy="cluster_route",
        )


def test_large_context_report_is_cleared_even_when_the_next_call_is_rejected():
    client = _client_with(_large_context_handler({}, _LARGE_CONTEXT_REPORT))
    client.predict(
        _X_TRAIN,
        _Y_TRAIN,
        _X_TEST,
        large_context_policy="cluster_route",
        large_context_threshold=1,
        large_context_seed=7,
    )
    assert client.last_large_context_report is not None
    with pytest.raises(ValueError):
        client.predict(
            _X_TRAIN,
            _Y_TRAIN,
            _X_TEST,
            large_context_policy=lambda *_: None,
        )
    assert client.last_large_context_report is None


def test_large_context_request_model_is_bounded_and_omits_unset_fields():
    base = {
        "X_train": _X_TRAIN,
        "y_train": _Y_TRAIN,
        "X_test": _X_TEST,
    }
    request = NoriPredictRequest(
        **base,
        large_context_policy="cluster_route_g4",
        large_context_threshold=123,
        large_context_seed=42,
    )
    assert request.to_wire()["large_context_policy"] == "cluster_route_g4"
    assert "large_context_policy" not in NoriPredictRequest(**base).to_wire()
    with pytest.raises(ValueError):
        NoriPredictRequest(**base, large_context_threshold=123)
    for policy in ("boost", "safeboost", "cluster_route[groups=16]"):
        assert NoriPredictRequest(**base, large_context_policy=policy).to_wire()["large_context_policy"] == policy
    gated = NoriPredictRequest(
        **base,
        large_context_policy=["random", "target_rank[cap=32768]"],
        large_context_holdout="tail",
    )
    assert gated.to_wire()["large_context_policy"] == ["random", "target_rank[cap=32768]"]
    assert gated.to_wire()["large_context_holdout"] == "tail"
    with pytest.raises(ValueError, match="valid only with.*list"):
        NoriPredictRequest(**base, large_context_policy="random", large_context_holdout="tail")
    with pytest.raises(ValueError):
        NoriPredictRequest(
            **base,
            large_context_policy="random",
            large_context_seed=2**32,
        )


def test_local_skipped_report_uses_the_callable_name():
    from synthefy import nori_client as module

    def custom_policy(_problem, _rng):
        pass

    request = NoriPredictRequest(
        X_train=_X_TRAIN,
        y_train=_Y_TRAIN,
        X_test=_X_TEST,
        large_context_policy=custom_policy,
    )
    report = module._normalized_large_context_report(request, None)

    assert report["applied"] is False
    assert report["policy"] == "custom_policy"


def test_local_large_context_uses_the_estimator_and_copies_its_report(monkeypatch):
    """Mirrors test_local_mode_forwards_the_policy_when_supported: the capability
    check probes the REAL installed synthefy-nori (find_spec), not this fake
    regressor's constructor, so "supported" has to be mocked directly -- an
    environment testing just the base synthefy install (no synthefy-nori at all)
    would otherwise report unsupported regardless of what FakeRegressor accepts.
    """
    from synthefy import nori_client as module

    seen: Dict = {}

    def custom_policy(_problem, _rng):
        pass

    class FakeRegressor:
        def __init__(
            self,
            model=None,
            memory_policy=None,
            large_context_policy=None,
            large_context_threshold=50_000,
            large_context_seed=0,
            large_context_cache_entries=1,
        ):
            seen["init"] = {
                "model": model,
                "large_context_policy": large_context_policy,
                "large_context_threshold": large_context_threshold,
                "large_context_seed": large_context_seed,
                "large_context_cache_entries": large_context_cache_entries,
            }
            self.memory_policy = memory_policy
            self.large_context_policy = large_context_policy
            self.large_context_threshold = large_context_threshold
            self.large_context_seed = large_context_seed
            self.large_context_cache_entries = large_context_cache_entries
            self.large_context_report_ = None

        def fit(self, X, y):
            seen["fit"] = (X, y)
            return self

        def predict(
            self,
            X,
            *,
            output_type="mean",
            quantiles=None,
            discretize=None,
            categorical_levels=None,
        ):
            seen["predict"] = {
                "output_type": output_type,
                "quantiles": quantiles,
                "discretize": discretize,
                "categorical_levels": categorical_levels,
            }
            self.large_context_report_ = {
                "policy": getattr(self.large_context_policy, "__name__", self.large_context_policy),
                "window": 2,
                "n_train": 3,
                "n_test": 2,
                "shards_available": 1,
                "nori_calls": 2,
                "full_context": False,
                "reused_train_state": False,
            }
            return [0.4, 0.5]

    monkeypatch.setattr(module, "_load_local_regressor", lambda: FakeRegressor)
    monkeypatch.setattr(module, "_local_large_context_available", lambda: True)
    client = SynthefyNoriClient(model="nori-30m", mode="local")
    predictions = client.predict(
        _X_TRAIN,
        _Y_TRAIN,
        _X_TEST,
        large_context_policy=custom_policy,
        large_context_threshold=1,
        large_context_seed=7,
    )
    assert predictions == [0.4, 0.5]
    assert seen["init"] == {
        "model": "nori-30m",
        "large_context_policy": custom_policy,
        "large_context_threshold": 1,
        "large_context_seed": 7,
        "large_context_cache_entries": 1,
    }
    assert client.last_large_context_report == {
        **_LARGE_CONTEXT_REPORT,
        "policy": "custom_policy",
    }


def test_local_large_context_has_a_clear_old_package_guard(monkeypatch):
    """An opaque TypeError from deep inside the library is not an acceptable answer.

    Mirrors test_local_mode_refuses_memory_on_an_old_synthefy_nori: the capability
    check is a module-presence probe (find_spec against the REAL installed
    synthefy-nori), not a signature probe on whatever regressor class
    _load_local_regressor returns, so simulating "an old package" means patching the
    availability check itself -- an unrelated mocked regressor class proves nothing
    about which synthefy-nori is actually installed.
    """
    from synthefy import nori_client as module

    class OldRegressor:
        def __init__(self, model=None):
            self.model = model

        def fit(self, X, y):
            return self

        def predict(self, X, *, output_type="mean", quantiles=None):
            return [0.0] * len(X)

    monkeypatch.setattr(module, "_load_local_regressor", lambda: OldRegressor)
    monkeypatch.setattr(module, "_local_large_context_available", lambda: False)
    client = SynthefyNoriClient(model="nori-30m", mode="local")
    with pytest.raises(ImportError, match="newer synthefy-nori"):
        client.predict(
            _X_TRAIN,
            _Y_TRAIN,
            _X_TEST,
            large_context_policy="cluster_route",
        )


def _remote_client() -> SynthefyNoriClient:
    return SynthefyNoriClient(api_key="k", model="nori-30m")


# ----------------------------------------------------- argument validation


@pytest.mark.parametrize("output_type", ["mode", "p50"])
def test_unknown_output_type_raises_listing_the_valid_names(output_type):
    client = _remote_client()
    _attach_mock(client, _ok_handler([1.0], {}))
    with pytest.raises(ValueError, match="output_type must be one of"):
        client.predict(_XTR, _YTR, _XTE, output_type=output_type)


def test_quantiles_output_type_requires_levels():
    client = _remote_client()
    _attach_mock(client, _ok_handler([1.0], {}))
    with pytest.raises(ValueError, match=r"requires quantiles=\[\.\.\.\]"):
        client.predict(_XTR, _YTR, _XTE, output_type="quantiles")


def test_empty_quantiles_raises():
    client = _remote_client()
    _attach_mock(client, _ok_handler([1.0], {}))
    with pytest.raises(ValueError, match="empty sequence"):
        client.predict(_XTR, _YTR, _XTE, output_type="quantiles", quantiles=[])


@pytest.mark.parametrize("bad", [[0.0], [1.0], [1.5], [-0.1], [0.5, 1.0], [float("nan")]])
def test_quantiles_must_lie_strictly_inside_the_unit_interval(bad):
    client = _remote_client()
    _attach_mock(client, _ok_handler([1.0], {}))
    with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
        client.predict(_XTR, _YTR, _XTE, output_type="quantiles", quantiles=bad)


def test_quantiles_without_the_quantiles_output_type_raises():
    client = _remote_client()
    _attach_mock(client, _ok_handler([1.0], {}))
    with pytest.raises(ValueError, match="only valid with output_type='quantiles'"):
        client.predict(_XTR, _YTR, _XTE, quantiles=_LEVELS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_type": "quantiles", "quantiles": _LEVELS, "discretize": "snap-mean"},
        {"output_type": "median", "discretize": "snap-mean"},
        {"output_type": "median", "categorical_levels": [1, 2]},
    ],
)
def test_distribution_output_cannot_combine_with_discretization(kwargs):
    # Mirrors NoriRegressor.predict: discrete labels and a distribution summary
    # are different answers, so asking for both is a bug, not a merge.
    client = _remote_client()
    _attach_mock(client, _ok_handler([1.0], {}))
    with pytest.raises(ValueError, match="combines only with the default"):
        client.predict(_XTR, _YTR, _XTE, **kwargs)


def test_output_type_is_validated_before_any_expensive_work():
    # text_columns would normally raise (non-DataFrame inputs) after loading an
    # embedder; a bad output_type must be reported first, so no encoder, no
    # checkpoint and no network round-trip is paid for an unusable request.
    client = _remote_client()
    _attach_mock(client, _ok_handler([1.0], {}))
    with pytest.raises(ValueError, match="output_type must be one of"):
        client.predict(_XTR, _YTR, _XTE, output_type="nope", text_columns=["review"])


# ----------------------------------------------------- remote: wire format


def test_default_request_body_is_unchanged_by_the_new_parameters():
    # The byte-compatibility guarantee: an ordinary predict() must not start
    # sending output_type/quantiles to deployments that don't know them.
    capture: Dict = {}
    client = _remote_client()
    _attach_mock(client, _ok_handler([1.0], capture))
    client.predict(_XTR, _YTR, _XTE)
    assert set(capture["body"]) == {"X_train", "y_train", "X_test", "task", "model"}


def test_remote_quantiles_sends_the_levels_and_returns_level_major():
    capture: Dict = {}
    client = _remote_client()
    # Wire is row-major: one row per query row, one column per level.
    _attach_mock(
        client,
        _dist_handler(
            capture,
            predictions=[1.0, 2.0],
            quantiles=[[0.5, 1.0, 1.5], [1.5, 2.0, 2.5]],
            taus=_LEVELS,
        ),
    )

    lo, mid, hi = client.predict(
        _XTR,
        _YTR,
        [[2.0, 2.0], [3.0, 3.0]],
        output_type="quantiles",
        quantiles=_LEVELS,
    )

    assert capture["body"]["output_type"] == "quantiles"
    assert capture["body"]["quantiles"] == _LEVELS
    # Level-major (n_levels, n_query), so lo/mid/hi unpack straight out.
    assert lo == [0.5, 1.5]
    assert mid == [1.0, 2.0]
    assert hi == [1.5, 2.5]


def test_remote_quantiles_preserve_memory_policy_and_report():
    capture: Dict = {}
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            capture,
            predictions=[1.0],
            quantiles=[[0.5, 1.0, 1.5]],
            taus=_LEVELS,
            memory_report=_REPORT,
        ),
    )

    client.predict(
        _XTR,
        _YTR,
        [[2.0, 2.0]],
        output_type="quantiles",
        quantiles=_LEVELS,
        memory_policy={"cache_dtype": "int8"},
    )

    assert capture["body"]["memory_policy"] == {"cache_dtype": "int8"}
    assert client.last_memory_report["rung"] == "resident_int8"


def test_remote_quantiles_preserve_the_requested_level_order():
    # The rows follow the caller's order, not sorted order (same as NoriRegressor).
    capture: Dict = {}
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            capture,
            predictions=[1.0],
            quantiles=[[9.0, 1.0]],
            taus=[0.9, 0.1],
        ),
    )
    hi, lo = client.predict(_XTR, _YTR, [[2.0, 2.0]], output_type="quantiles", quantiles=[0.9, 0.1])
    assert capture["body"]["quantiles"] == [0.9, 0.1]
    assert hi == [9.0] and lo == [1.0]


def test_remote_quantiles_as_pandas_is_a_frame_keyed_by_level():
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            {},
            predictions=[1.0, 2.0],
            quantiles=[[0.5, 1.0, 1.5], [1.5, 2.0, 2.5]],
            taus=_LEVELS,
        ),
    )
    X_test = pd.DataFrame({"a": [2.0, 3.0], "b": [2.0, 3.0]}, index=["r1", "r2"])
    y_train = pd.Series([1.0, 1.0, 2.0], name="price")

    out = client.predict(
        pd.DataFrame(_XTR, columns=["a", "b"]),
        y_train,
        X_test,
        output_type="quantiles",
        quantiles=_LEVELS,
        as_pandas=True,
    )

    assert isinstance(out, pd.DataFrame)
    # Row-major under pandas, indexed by X_test so the bands join back.
    assert list(out.index) == ["r1", "r2"]
    assert list(out.columns) == ["price[0.1]", "price[0.5]", "price[0.9]"]
    assert out["price[0.5]"].tolist() == [1.0, 2.0]


def test_remote_full_returns_the_whole_bank():
    capture: Dict = {}
    client = _remote_client()
    taus = [0.25, 0.5, 0.75]
    _attach_mock(
        client,
        _dist_handler(
            capture,
            predictions=[1.0, 2.0],
            quantiles=[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
            taus=taus,
            output_type="full",
        ),
    )

    out = client.predict(_XTR, _YTR, [[2.0, 2.0], [3.0, 3.0]], output_type="full")

    assert capture["body"]["output_type"] == "full"
    assert "quantiles" not in capture["body"]  # no levels needed for "full"
    assert set(out) == {"quantiles", "taus", "mean"}
    assert out["quantiles"] == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]  # row-major
    assert out["taus"] == taus
    assert out["mean"] == [1.0, 2.0]


def test_remote_full_as_pandas_gives_a_frame_and_a_series():
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            {},
            predictions=[1.0, 2.0],
            quantiles=[[0.0, 2.0], [1.0, 3.0]],
            taus=[0.25, 0.75],
            output_type="full",
        ),
    )
    X_test = pd.DataFrame({"a": [2.0, 3.0], "b": [2.0, 3.0]}, index=[7, 8])

    out = client.predict(
        pd.DataFrame(_XTR, columns=["a", "b"]),
        pd.Series(_YTR, name="y"),
        X_test,
        output_type="full",
        as_pandas=True,
    )

    assert list(out["quantiles"].columns) == ["y[0.25]", "y[0.75]"]
    assert list(out["quantiles"].index) == [7, 8]
    assert isinstance(out["mean"], pd.Series)
    assert out["mean"].name == "y" and list(out["mean"].index) == [7, 8]
    assert out["taus"] == [0.25, 0.75]  # plain list, not a pandas object


def test_remote_median_returns_one_value_per_row():
    capture: Dict = {}
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            capture,
            predictions=[3.0, 4.0],
            output_type="median",
        ),
    )
    preds = client.predict(_XTR, _YTR, [[2.0, 2.0], [3.0, 3.0]], output_type="median")
    assert capture["body"]["output_type"] == "median"
    assert preds == [3.0, 4.0]


# ----------------------------------------------------- remote: capability gate


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_type": "median"},
        {"output_type": "quantiles", "quantiles": _LEVELS},
        {"output_type": "full"},
    ],
)
def test_remote_deployment_that_ignores_output_type_raises(kwargs):
    # THE important case: an old deployment silently answers with means. Means
    # are indistinguishable from a real "median" result, so without the echo the
    # client would hand back a confidently wrong answer.
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            {},
            predictions=[1.0],
            echo_output_type=False,
        ),
    )
    with pytest.raises(ValueError, match="predates distribution output"):
        client.predict(_XTR, _YTR, [[2.0, 2.0]], **kwargs)


def test_remote_echoing_a_different_output_type_raises():
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            {},
            predictions=[1.0],
            output_type="mean",
        ),
    )
    with pytest.raises(ValueError, match="honored output_type='mean' instead"):
        client.predict(_XTR, _YTR, [[2.0, 2.0]], output_type="median")


def test_remote_echo_without_a_quantile_block_raises():
    client = _remote_client()
    _attach_mock(client, _dist_handler({}, predictions=[1.0]))  # no quantiles/taus
    with pytest.raises(ValueError, match="returned no quantile block"):
        client.predict(_XTR, _YTR, [[2.0, 2.0]], output_type="quantiles", quantiles=_LEVELS)


def test_remote_default_output_type_needs_no_echo():
    # Ordinary calls must keep working against deployments that never send the
    # field, so the gate applies only to a non-default output_type.
    client = _remote_client()
    _attach_mock(client, _ok_handler([1.0, 2.0], {}))
    assert client.predict(_XTR, _YTR, [[2.0, 2.0], [3.0, 3.0]]) == [1.0, 2.0]


def test_remote_malformed_quantile_block_shape_raises():
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            {},
            predictions=[1.0, 2.0],
            quantiles=[[0.5, 1.0, 1.5]],  # 1 row for 2 query rows
            taus=_LEVELS,
        ),
    )
    with pytest.raises(ValueError, match="quantile block of shape"):
        client.predict(
            _XTR,
            _YTR,
            [[2.0, 2.0], [3.0, 3.0]],
            output_type="quantiles",
            quantiles=_LEVELS,
        )


def test_remote_level_count_mismatch_raises():
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            {},
            predictions=[1.0],
            quantiles=[[0.5, 1.5]],
            taus=[0.1, 0.9],
        ),
    )
    with pytest.raises(ValueError, match="server returned 2"):
        client.predict(_XTR, _YTR, [[2.0, 2.0]], output_type="quantiles", quantiles=_LEVELS)


def test_remote_null_quantiles_come_back_as_nan():
    # JSON has no NaN, so the server nulls non-finite values; they must land as
    # NaN rather than None objects inside a numeric result.
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            {},
            predictions=[1.0],
            quantiles=[[0.5, None, 1.5]],
            taus=_LEVELS,
        ),
    )
    lo, mid, hi = client.predict(_XTR, _YTR, [[2.0, 2.0]], output_type="quantiles", quantiles=_LEVELS)
    assert lo == [0.5] and hi == [1.5]
    assert math.isnan(mid[0])


def test_response_model_parses_the_distribution_fields():
    resp = NoriPredictResponse(
        **{
            "task": "regression",
            "predictions": [1.0],
            "output_type": "quantiles",
            "quantiles": [[0.5, 1.0, 1.5]],
            "taus": _LEVELS,
        }
    )
    assert resp.output_type == "quantiles"
    assert resp.quantiles == [[0.5, 1.0, 1.5]]
    assert resp.taus == _LEVELS


# ----------------------------------------------------- local mode


def _fake_regressor_class(seen: Dict, *, with_model=True, with_output_type=True):
    """Build a NoriRegressor stand-in recording what the client forwarded.

    ``with_model``/``with_output_type`` drop the corresponding parameter to
    simulate a synthefy-nori too old for it (the client probes the signatures).
    """
    if with_model:

        def __init__(self, model=None, memory_policy=None):
            seen["init_count"] = seen.get("init_count", 0) + 1
            seen["init_model"] = model
            seen["init_memory_policy"] = memory_policy
            self.memory_policy = memory_policy
            self.memory_report_ = None
    else:

        def __init__(self):  # noqa: E306 - old build: no model= selector
            seen["init_count"] = seen.get("init_count", 0) + 1
            seen["init_model"] = "<absent>"
            self.memory_report_ = None

    def fit(self, X, y):
        seen["fit"] = (X, y)
        return self

    if with_output_type:

        def predict(self, X, *, output_type="mean", quantiles=None):
            seen["predict"] = {
                "X": X,
                "output_type": output_type,
                "quantiles": quantiles,
            }
            if getattr(self, "memory_policy", None) is not None:
                self.memory_report_ = dict(_REPORT)
            n = len(X)
            if output_type == "quantiles":
                # (n_levels, n_query), the shape NoriRegressor returns.
                return np.array(
                    [[10.0 * k + i for i in range(n)] for k in range(len(quantiles))],
                    dtype=float,
                )
            if output_type == "full":
                K = 3
                Q = np.array([[float(i) + k for k in range(K)] for i in range(n)], dtype=float)
                taus = (np.arange(K, dtype=float) + 1.0) / (K + 1.0)
                return {"quantiles": Q, "taus": taus, "mean": Q.mean(axis=1)}
            return np.arange(n, dtype=float)
    else:

        def predict(self, X):  # noqa: E306 - old build: mean only
            return np.arange(len(X), dtype=float)

    return type(
        "_FakeNoriRegressor",
        (),
        {"__init__": __init__, "fit": fit, "predict": predict},
    )


def _local_client_with_fake(monkeypatch, seen: Dict, **kwargs) -> SynthefyNoriClient:
    monkeypatch.setattr(
        "synthefy.nori_client._load_local_regressor",
        lambda: _fake_regressor_class(seen, **kwargs),
    )
    return SynthefyNoriClient(mode="local", model="nori-30m")


def test_local_quantiles_route_through_the_estimator_api(monkeypatch):
    # The functional synthefy_nori.predict cannot express output_type (it forwards
    # **kwargs to the constructor), so local distribution output must fit/predict
    # a NoriRegressor directly.
    seen: Dict = {}
    client = _local_client_with_fake(monkeypatch, seen)

    lo, mid, hi = client.predict(
        _XTR,
        _YTR,
        [[2.0, 2.0], [3.0, 3.0]],
        output_type="quantiles",
        quantiles=_LEVELS,
    )

    assert seen["fit"][0] == _XTR and seen["fit"][1] == _YTR
    assert seen["predict"]["output_type"] == "quantiles"
    assert seen["predict"]["quantiles"] == _LEVELS
    assert seen["init_model"] == "nori-30m"  # variant selector forwarded
    assert lo == [0.0, 1.0]
    assert mid == [10.0, 11.0]
    assert hi == [20.0, 21.0]


def test_local_client_reuses_one_estimator_until_closed(monkeypatch):
    seen: Dict = {}
    client = _local_client_with_fake(monkeypatch, seen)

    for _ in range(2):
        client.predict(
            _XTR,
            _YTR,
            _XTE,
            output_type="quantiles",
            quantiles=_LEVELS,
        )

    assert seen["init_count"] == 1

    client.close()
    client.predict(
        _XTR,
        _YTR,
        _XTE,
        output_type="quantiles",
        quantiles=_LEVELS,
    )
    assert seen["init_count"] == 2


def test_local_quantiles_forward_memory_policy_to_the_estimator(monkeypatch):
    seen: Dict = {}
    monkeypatch.setattr("synthefy.nori_client._local_memory_policy_available", lambda: True)
    client = _local_client_with_fake(monkeypatch, seen)

    client.predict(
        _XTR,
        _YTR,
        _XTE,
        output_type="quantiles",
        quantiles=_LEVELS,
        memory_policy={"cache_dtype": "int8"},
    )

    assert seen["init_memory_policy"] == {"cache_dtype": "int8"}


def test_local_full_returns_the_bank_dict(monkeypatch):
    seen: Dict = {}
    client = _local_client_with_fake(monkeypatch, seen)

    out = client.predict(_XTR, _YTR, [[2.0, 2.0], [3.0, 3.0]], output_type="full")

    assert seen["predict"]["output_type"] == "full"
    assert seen["predict"]["quantiles"] is None
    assert set(out) == {"quantiles", "taus", "mean"}
    assert out["quantiles"] == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
    assert out["taus"] == [0.25, 0.5, 0.75]
    assert out["mean"] == [1.0, 2.0]


def test_local_median_also_uses_the_estimator(monkeypatch):
    seen: Dict = {}
    client = _local_client_with_fake(monkeypatch, seen)
    preds = client.predict(_XTR, _YTR, [[2.0, 2.0], [3.0, 3.0]], output_type="median")
    assert seen["predict"]["output_type"] == "median"
    assert preds == [0.0, 1.0]


def test_local_mean_still_uses_the_functional_predict(monkeypatch):
    # The default path must not change: it keeps calling synthefy_nori.predict, so
    # existing local behavior is byte-for-byte what it was.
    def fake_predict(X_train, y_train, X_test, *, task=None, model=None, **kwargs):
        assert model == "nori-30m"
        return [7.0]

    def _boom():
        raise AssertionError("output_type='mean' must not load NoriRegressor")

    monkeypatch.setattr("synthefy.nori_client._load_local_predict", lambda: fake_predict)
    monkeypatch.setattr("synthefy.nori_client._load_local_regressor", _boom)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    assert client.predict(_XTR, _YTR, _XTE) == [7.0]


def test_local_quantiles_as_pandas_frame(monkeypatch):
    seen: Dict = {}
    client = _local_client_with_fake(monkeypatch, seen)
    X_test = pd.DataFrame({"a": [2.0, 3.0], "b": [2.0, 3.0]}, index=["r1", "r2"])

    out = client.predict(
        pd.DataFrame(_XTR, columns=["a", "b"]),
        pd.Series(_YTR, name="score"),
        X_test,
        output_type="quantiles",
        quantiles=[0.1, 0.9],
        as_pandas=True,
    )

    assert list(out.columns) == ["score[0.1]", "score[0.9]"]
    assert list(out.index) == ["r1", "r2"]
    assert out["score[0.9]"].tolist() == [10.0, 11.0]


def test_local_output_type_needs_a_newer_synthefy_nori(monkeypatch):
    seen: Dict = {}
    client = _local_client_with_fake(monkeypatch, seen, with_output_type=False)
    with pytest.raises(ImportError, match=r"output_type=.*added in 0\.6\.0"):
        client.predict(_XTR, _YTR, _XTE, output_type="quantiles", quantiles=_LEVELS)


def test_local_variant_selector_needs_a_newer_synthefy_nori(monkeypatch):
    seen: Dict = {}
    client = _local_client_with_fake(monkeypatch, seen, with_model=False)
    with pytest.raises(ImportError, match="model= selector"):
        client.predict(_XTR, _YTR, _XTE, output_type="quantiles", quantiles=_LEVELS)


def test_local_quantiles_missing_package_raises_install_hint(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "synthefy_nori" or name.startswith("synthefy_nori."):
            raise ModuleNotFoundError("No module named 'synthefy_nori'", name="synthefy_nori")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    with pytest.raises(ImportError, match=r"synthefy-nori"):
        client.predict(_XTR, _YTR, _XTE, output_type="quantiles", quantiles=_LEVELS)


def test_local_unexpected_quantile_shape_raises(monkeypatch):
    # Guards against a silent shape change in synthefy-nori: better a clear error
    # here than a confusing column mismatch downstream.
    class _Wrong:
        def __init__(self, model=None):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X, *, output_type="mean", quantiles=None):
            return np.zeros((len(quantiles) + 1, len(X)))

    monkeypatch.setattr("synthefy.nori_client._load_local_regressor", lambda: _Wrong)
    client = SynthefyNoriClient(mode="local", model="nori-30m")
    with pytest.raises(ValueError, match="quantile array of shape"):
        client.predict(_XTR, _YTR, _XTE, output_type="quantiles", quantiles=_LEVELS)


@pytest.mark.slow
def test_local_quantiles_real_inference():
    """Real checkpoint: the bands must be monotone and bracket the point prediction."""
    pytest.importorskip("synthefy_nori")

    client = SynthefyNoriClient(mode="local", model="nori-6m")
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(60, 3))
    y_train = X_train[:, 0] * 2.0 + rng.normal(scale=0.3, size=60)
    X_test = rng.normal(size=(8, 3))

    lo, mid, hi = client.predict(X_train, y_train, X_test, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    assert len(lo) == len(mid) == len(hi) == 8
    # A valid quantile function is non-decreasing in tau.
    assert all(a <= b <= c for a, b, c in zip(lo, mid, hi))

    mean = client.predict(X_train, y_train, X_test)
    assert all(a <= m <= c for a, m, c in zip(lo, mean, hi))

    full = client.predict(X_train, y_train, X_test, output_type="full")
    Q = np.asarray(full["quantiles"])
    taus = np.asarray(full["taus"])
    assert Q.shape == (8, taus.shape[0])
    assert np.all(np.diff(Q, axis=1) >= -1e-9)  # ascending per row
    assert np.all((taus > 0.0) & (taus < 1.0))
    # "full"'s mean is the same quantity output_type="mean" collapses to.
    assert np.allclose(full["mean"], np.asarray(mean), atol=0.15)


def test_remote_drifted_quantile_levels_raise():
    # Columns are labeled from the request, so levels that came back different
    # would mean data at one tau labeled with another.
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            {},
            predictions=[1.0],
            quantiles=[[0.5, 1.0, 1.5]],
            taus=[0.1, 0.5, 0.95],  # 0.9 -> 0.95
        ),
    )
    with pytest.raises(ValueError, match="server returned"):
        client.predict(_XTR, _YTR, [[2.0, 2.0]], output_type="quantiles", quantiles=_LEVELS)


def test_remote_ragged_quantile_block_raises_a_clear_error():
    client = _remote_client()
    _attach_mock(
        client,
        _dist_handler(
            {},
            predictions=[1.0, 2.0],
            quantiles=[[0.5, 1.0, 1.5], [1.5, 2.0]],  # second row short
            taus=_LEVELS,
        ),
    )
    with pytest.raises(ValueError, match="rows of unequal length"):
        client.predict(
            _XTR,
            _YTR,
            [[2.0, 2.0], [3.0, 3.0]],
            output_type="quantiles",
            quantiles=_LEVELS,
        )
