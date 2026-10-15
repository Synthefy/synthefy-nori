"""The AWS fakes and builders the fine-tuner's tests share; no test reaches AWS.

Describe, list, stop and delete calls answer from a hand-made SageMaker fake whose answers are queued per resource,
so a test states the sequence of states a job, package or endpoint goes through. Every ``create_*`` call goes to a
real botocore SageMaker client under a ``Stubber`` with ``expected_params``, so the request the fine-tuner sends is
pinned in full. Every call, stubbed or faked, SageMaker, CloudWatch Logs or S3, is checked against the installed
botocore's service model, so a parameter botocore would refuse before signing fails here too. Time never passes in
:func:`install_fake_aws`: the modules' clock is a fake whose ``sleep`` only advances it, and the fine-tuner's
``_utcnow`` is fixed so every generated name is exact.
"""

import contextlib
import copy
import functools
import io
import json
import tarfile
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import boto3
import botocore.session
from botocore import xform_name
from botocore.stub import Stubber
from botocore.validate import ParamValidator

from synthefy import _sagemaker_deploy as deploy_module
from synthefy import nori_fine_tune as fine_tune_module
from synthefy.nori_data_models import FineTuneHyperparameters, FineTuneJobRequest, serialize_hyperparameters
from synthefy.nori_fine_tune import SynthefyNoriFineTuner

REGION = "us-east-1"
ACCOUNT = "111122223333"
ALGORITHM = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:algorithm/nori-fine-tuning"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/SageMakerExecution"
MODEL = "ft-acme-churn-v1"
NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
JOB = "nori-ft-acme-churn-v1-20261001-120000"
JOB_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/{JOB}"
BUCKET = "customer-bucket"
TRAIN = f"s3://{BUCKET}/nori/train/"
OUTPUT = f"s3://{BUCKET}/nori/output/"
ARTIFACT_KEY = f"nori/output/{JOB}/output/model.tar.gz"
ARTIFACT = f"s3://{BUCKET}/{ARTIFACT_KEY}"
ASYNC_ROOT = f"s3://{BUCKET}/nori/output/{MODEL}/async"
PACKAGE_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package/{JOB}"
CONFIG = f"{MODEL}-20261001-120000"
SERVED_TYPES = ["ml.g5.xlarge", "ml.g5.2xlarge", "ml.g6.xlarge"]
STREAM = f"{JOB}/algo-1-1759320000"
#: Bytes that are not a report: the report reader must never read them.
OTHER_MEMBER = b"\x80\x02not a report"


# --------------------------------------------------------------------------- #
# botocore's own parameter check
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=None)
def _load_operation_models(service_name: str) -> Dict[str, Any]:
    """The installed botocore's operation models of one service, by boto3 method name."""
    service_model = botocore.session.get_session().get_service_model(service_name)
    return {xform_name(name): service_model.operation_model(name) for name in service_model.operation_names}


def assert_botocore_accepts(service_name: str, method_name: str, params: Dict[str, Any]) -> None:
    """Fail when the installed botocore would refuse these parameters before signing the request.

    A fake answers whatever it is sent, so without this it would accept a call the real client refuses.
    """
    operation = _load_operation_models(service_name)[method_name]
    report = ParamValidator().validate(params, operation.input_shape)
    assert not report.has_errors(), f"{service_name}.{method_name}: {report.generate_report()}"


# --------------------------------------------------------------------------- #
# Errors and fakes
# --------------------------------------------------------------------------- #


class AWSError(Exception):
    """A botocore ClientError's shape: the fine-tuner classifies AWS errors by ``response`` alone."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"An error occurred ({code}): {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


def build_missing_resource_error(kind: str, name: str = MODEL) -> AWSError:
    """SageMaker's answer for a missing endpoint, endpoint configuration or model."""
    return AWSError("ValidationException", f'Could not find {kind} "arn:aws:sagemaker:{REGION}:{ACCOUNT}:{name}".')


def build_missing_package_error() -> AWSError:
    return AWSError("ValidationException", f"ModelPackage {PACKAGE_ARN} does not exist.")


class FakeSageMaker:
    """SageMaker as the fine-tuner sees it.

    Each describe answers from its queue in order and repeats the last answer; an exception in a queue is raised.
    Stop and delete calls fail only when ``failures`` names them. An endpoint config exists once the stubbed
    create succeeded; ``hidden_endpoint_config_reads`` makes that many DescribeEndpointConfig calls miss it first,
    as SageMaker's eventually consistent read can. Every call is recorded with its parameters and checked against
    botocore's SageMaker model.
    """

    def __init__(self, stubbed: Any) -> None:
        self._stubbed = stubbed
        self.meta = stubbed.meta
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.training_jobs: List[Any] = []
        self.endpoints: List[Any] = [build_missing_resource_error("endpoint")]
        self.model_packages: List[Any] = [build_missing_package_error()]
        self.models: List[Any] = [build_missing_resource_error("model")]
        self.endpoint_config_names: List[str] = []
        #: The SageMaker model each endpoint config serves, for describe_endpoint_config.
        self.endpoint_config_models: Dict[str, str] = {}
        self.hidden_endpoint_config_reads = 0
        self.failures: Dict[str, Exception] = {}
        self.closed = False

    def _check_and_record(self, operation: str, params: Dict[str, Any]) -> None:
        assert_botocore_accepts("sagemaker", operation, params)
        self.calls.append((operation, params))

    def _answer_from_queue(self, operation: str, queue: List[Any], params: Dict[str, Any]) -> Dict[str, Any]:
        self._check_and_record(operation, params)
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return copy.deepcopy(answer)

    def _record_call(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._check_and_record(operation, params)
        failure = self.failures.get(operation)
        if failure is not None:
            raise failure
        return {}

    def _create_through_stubber(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((operation, params))
        return getattr(self._stubbed, operation)(**params)

    def describe_training_job(self, **params: Any) -> Dict[str, Any]:
        return self._answer_from_queue("describe_training_job", self.training_jobs, params)

    def describe_endpoint(self, **params: Any) -> Dict[str, Any]:
        return self._answer_from_queue("describe_endpoint", self.endpoints, params)

    def describe_model_package(self, **params: Any) -> Dict[str, Any]:
        return self._answer_from_queue("describe_model_package", self.model_packages, params)

    def describe_model(self, **params: Any) -> Dict[str, Any]:
        return self._answer_from_queue("describe_model", self.models, params)

    def describe_endpoint_config(self, **params: Any) -> Dict[str, Any]:
        self._check_and_record("describe_endpoint_config", params)
        name = params["EndpointConfigName"]
        if self.hidden_endpoint_config_reads > 0 or name not in self.endpoint_config_models:
            self.hidden_endpoint_config_reads = max(0, self.hidden_endpoint_config_reads - 1)
            raise build_missing_resource_error("endpoint configuration", name)
        return {
            "EndpointConfigName": name,
            "EndpointConfigArn": f"arn:config/{name}",
            "CreationTime": NOW,
            "ProductionVariants": [{"VariantName": "AllTraffic", "ModelName": self.endpoint_config_models[name]}],
        }

    def list_endpoint_configs(self, **params: Any) -> Dict[str, Any]:
        self._check_and_record("list_endpoint_configs", params)
        return {
            "EndpointConfigs": [
                {"EndpointConfigName": name, "EndpointConfigArn": f"arn:config/{name}", "CreationTime": NOW}
                for name in self.endpoint_config_names
                if params["NameContains"] in name
            ]
        }

    def stop_training_job(self, **params: Any) -> Dict[str, Any]:
        return self._record_call("stop_training_job", params)

    def delete_endpoint(self, **params: Any) -> Dict[str, Any]:
        return self._record_call("delete_endpoint", params)

    def delete_endpoint_config(self, **params: Any) -> Dict[str, Any]:
        return self._record_call("delete_endpoint_config", params)

    def delete_model(self, **params: Any) -> Dict[str, Any]:
        return self._record_call("delete_model", params)

    def create_training_job(self, **params: Any) -> Dict[str, Any]:
        return self._create_through_stubber("create_training_job", params)

    def create_model_package(self, **params: Any) -> Dict[str, Any]:
        return self._create_through_stubber("create_model_package", params)

    def create_model(self, **params: Any) -> Dict[str, Any]:
        return self._create_through_stubber("create_model", params)

    def create_endpoint_config(self, **params: Any) -> Dict[str, Any]:
        answer = self._create_through_stubber("create_endpoint_config", params)
        self.endpoint_config_models[params["EndpointConfigName"]] = params["ProductionVariants"][0]["ModelName"]
        return answer

    def create_endpoint(self, **params: Any) -> Dict[str, Any]:
        return self._create_through_stubber("create_endpoint", params)

    def close(self) -> None:
        self.closed = True

    def operations(self) -> List[str]:
        return [operation for operation, _ in self.calls]


class FakeLogs:
    """CloudWatch Logs paging as the service does: two events per page, and an exhausted stream hands back the
    forward token it was given.

    ``empty_pages`` makes a stream open with that many empty pages whose forward token still advances, which
    GetLogEvents is documented to return.
    """

    page_size = 2

    def __init__(self) -> None:
        self.streams: Dict[str, List[str]] = {}
        self.empty_pages: Dict[str, int] = {}
        self.group_exists = True
        self.closed = False

    def describe_log_streams(self, **params: Any) -> Dict[str, Any]:
        assert_botocore_accepts("logs", "describe_log_streams", params)
        assert params["logGroupName"] == "/aws/sagemaker/TrainingJobs"
        if not self.group_exists:
            raise AWSError("ResourceNotFoundException", "The specified log group does not exist.")
        names = sorted(name for name in self.streams if name.startswith(params["logStreamNamePrefix"]))
        return {"logStreams": [{"logStreamName": name} for name in names]}

    def get_log_events(self, **params: Any) -> Dict[str, Any]:
        assert_botocore_accepts("logs", "get_log_events", params)
        assert params["logGroupName"] == "/aws/sagemaker/TrainingJobs" and params["startFromHead"] is True
        name = params["logStreamName"]
        token = params.get("nextToken")
        empty_served = int(token.split("/")[1]) if token and token.startswith("e/") else 0
        if (token is None or token.startswith("e/")) and empty_served < self.empty_pages.get(name, 0):
            return {"events": [], "nextForwardToken": f"e/{empty_served + 1}", "nextBackwardToken": "b/0"}
        start = int(token.split("/")[1]) if token and token.startswith("f/") else 0
        page = self.streams[name][start : start + self.page_size]
        return {
            "events": [{"timestamp": 0, "message": message} for message in page],
            "nextForwardToken": f"f/{start + len(page)}",
            "nextBackwardToken": "b/0",
        }

    def close(self) -> None:
        self.closed = True


class FakeS3:
    def __init__(self) -> None:
        self.objects: Dict[Tuple[str, str], bytes] = {}
        self.calls: List[Tuple[str, str]] = []
        self.closed = False

    def head_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        assert_botocore_accepts("s3", "head_object", {"Bucket": Bucket, "Key": Key})
        self.calls.append(("head_object", Key))
        if (Bucket, Key) not in self.objects:
            raise AWSError("404", "Not Found")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def get_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        assert_botocore_accepts("s3", "get_object", {"Bucket": Bucket, "Key": Key})
        self.calls.append(("get_object", Key))
        if (Bucket, Key) not in self.objects:
            raise AWSError("NoSuchKey", "The specified key does not exist.")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, *, Bucket: str, Prefix: str, MaxKeys: int) -> Dict[str, Any]:
        assert_botocore_accepts("s3", "list_objects_v2", {"Bucket": Bucket, "Prefix": Prefix, "MaxKeys": MaxKeys})
        self.calls.append(("list_objects_v2", Prefix))
        keys = sorted(key for bucket, key in self.objects if bucket == Bucket and key.startswith(Prefix))
        return {"Contents": [{"Key": key} for key in keys[:MaxKeys]], "IsTruncated": len(keys) > MaxKeys}

    def close(self) -> None:
        self.closed = True


class Clock:
    """A module's ``time``: ``sleep`` advances ``monotonic`` at once and can run a side effect."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: List[float] = []
        self.on_sleep: Optional[Callable[[float], None]] = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(seconds)


def create_stubbed_sagemaker() -> Tuple[Any, Any]:
    """A real botocore SageMaker client and its Stubber."""
    client = boto3.client(
        "sagemaker", region_name=REGION, aws_access_key_id="unit-test", aws_secret_access_key="unit-test"
    )
    return client, Stubber(client)


@contextlib.contextmanager
def install_fake_aws(monkeypatch: Any) -> Iterator[SimpleNamespace]:
    """A fine-tuner on the fakes, with a fake clock and a fixed UTC time; every stubbed create must be used."""
    stubbed, stubber = create_stubbed_sagemaker()
    fakes = SimpleNamespace(sm=FakeSageMaker(stubbed), logs=FakeLogs(), s3=FakeS3(), stubber=stubber, clock=Clock())
    monkeypatch.setattr(
        fine_tune_module, "create_aws_clients", lambda _service_names, **_kwargs: (fakes.sm, fakes.logs, fakes.s3)
    )
    monkeypatch.setattr(fine_tune_module, "_utcnow", lambda: NOW)
    monkeypatch.setattr(fine_tune_module, "time", fakes.clock)
    monkeypatch.setattr(deploy_module, "time", fakes.clock)
    with stubber:
        fakes.tuner = SynthefyNoriFineTuner()
        yield fakes
    stubber.assert_no_pending_responses()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def build_hyperparameters(**overrides: Any) -> FineTuneHyperparameters:
    return FineTuneHyperparameters(**{"target_column": "churn", "model_name": MODEL, **overrides})


def build_request(**overrides: Any) -> FineTuneJobRequest:
    fields: Dict[str, Any] = {
        "algorithm": ALGORITHM,
        "training_data": TRAIN,
        "output_path": OUTPUT,
        "role_arn": ROLE,
        "hyperparameters": build_hyperparameters(),
    }
    fields.update(overrides)
    return FineTuneJobRequest(**fields)


def build_description(status: str, **overrides: Any) -> Dict[str, Any]:
    """One DescribeTrainingJob response; ModelArtifacts is absent unless given."""
    description: Dict[str, Any] = {
        "TrainingJobName": JOB,
        "TrainingJobArn": JOB_ARN,
        "TrainingJobStatus": status,
        "SecondaryStatus": status,
        "SecondaryStatusTransitions": [
            {"Status": "Starting", "StartTime": NOW, "StatusMessage": "Preparing the instances for training"}
        ],
        "StoppingCondition": {"MaxRuntimeInSeconds": 12600},
        "CreationTime": NOW,
        "AlgorithmSpecification": {"AlgorithmName": ALGORITHM, "TrainingInputMode": "File"},
        "RoleArn": ROLE,
        "OutputDataConfig": {"S3OutputPath": OUTPUT},
        "HyperParameters": serialize_hyperparameters(build_hyperparameters()),
    }
    description.update(overrides)
    return description


def build_completed(**overrides: Any) -> Dict[str, Any]:
    return build_description(
        "Completed", ModelArtifacts={"S3ModelArtifacts": ARTIFACT}, TrainingEndTime=NOW, **overrides
    )


def build_package(status: str, *, artifact: str = ARTIFACT, reasons: Sequence[str] = ()) -> Dict[str, Any]:
    return {
        "ModelPackageName": JOB,
        "ModelPackageArn": PACKAGE_ARN,
        "ModelPackageStatus": status,
        "ModelPackageStatusDetails": {
            "ValidationStatuses": [{"Name": JOB, "Status": "Failed", "FailureReason": reason} for reason in reasons]
        },
        "SourceAlgorithmSpecification": {"SourceAlgorithms": [{"AlgorithmName": ALGORITHM, "ModelDataUrl": artifact}]},
        "InferenceSpecification": {"SupportedRealtimeInferenceInstanceTypes": list(SERVED_TYPES)},
    }


def build_endpoint(status: str, reason: Optional[str] = None) -> Dict[str, Any]:
    described: Dict[str, Any] = {"EndpointName": MODEL, "EndpointStatus": status}
    if reason is not None:
        described["FailureReason"] = reason
    return described


def build_curve(child: int, *, interrupted: bool = False) -> Dict[str, Any]:
    return {
        "child": child,
        "steps": [1, 2, 3],
        "train_loss": [0.9, None, 0.7],
        "val_mse": [0.5, 0.45, 0.4],
        "best_step": 3,
        "val_mse_best": 0.4,
        "interrupted": interrupted,
        "joined_average": True,
    }


def build_report(**overrides: Any) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "model_name": MODEL,
        "model_version": "sha256:" + "a" * 64,
        "base_model": "nori-6m",
        "base_model_version": "sha256:" + "b" * 64,
        "n_rows": 5000,
        "n_rows_used": 5000,
        "n_features": 4,
        "children": 2,
        "steps_run": [3, 3],
        "validation_source": "channel",
        "n_val_rows": 500,
        "val_mse_before": 0.6,
        "val_mse_after": 0.41,
        "child_curves": [build_curve(0), build_curve(1)],
        "cut_short": False,
        "seconds": 42.5,
        "instance_type": "ml.g5.xlarge",
    }
    report.update(overrides)
    return report


def build_artifact(report: Optional[Union[Dict[str, Any], bytes]], *, member_prefix: str = "./") -> bytes:
    """A model.tar.gz as SageMaker tars a job's model directory: other members first, the report last.

    ``report`` is the report's fields, or the member's raw bytes for a report that does not parse.
    """
    members = [
        (f"{member_prefix}model/weights.bin", OTHER_MEMBER),
        (f"{member_prefix}metadata.json", b"{}"),
    ]
    if report is not None:
        data = report if isinstance(report, bytes) else json.dumps(report).encode()
        members.append((f"{member_prefix}training_report.json", data))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def store_artifact(s3: FakeS3, data: bytes) -> None:
    s3.objects[(BUCKET, ARTIFACT_KEY)] = data


def build_channel(name: str, uri: str) -> Dict[str, Any]:
    return {
        "ChannelName": name,
        "ContentType": "text/csv",
        "InputMode": "File",
        "DataSource": {
            "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated"}
        },
    }


def build_expected_training_job(**overrides: Any) -> Dict[str, Any]:
    """CreateTrainingJob's parameters for ``build_request()`` with every default."""
    params: Dict[str, Any] = {
        "TrainingJobName": JOB,
        "AlgorithmSpecification": {"AlgorithmName": ALGORITHM, "TrainingInputMode": "File"},
        "RoleArn": ROLE,
        "HyperParameters": {
            "target_column": "churn",
            "model_name": MODEL,
            "base_model": "nori-6m",
            "task": "regression",
            "bag": "8",
            "steps": "100",
            "lr": "1e-05",
            "patience": "25",
            "budget_s": "900.0",
            "seed": "0",
        },
        "InputDataConfig": [build_channel("train", TRAIN)],
        "OutputDataConfig": {"S3OutputPath": OUTPUT},
        "ResourceConfig": {"InstanceType": "ml.g5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 30},
        "StoppingCondition": {"MaxRuntimeInSeconds": 12600},
        "EnableNetworkIsolation": True,
    }
    params.update(overrides)
    return params


def expect_create_model_package(stubber: Any) -> None:
    stubber.add_response(
        "create_model_package",
        {"ModelPackageArn": PACKAGE_ARN},
        {
            "ModelPackageName": JOB,
            "ModelPackageDescription": f"Fine-tuned Nori model {MODEL} from training job {JOB}",
            "SourceAlgorithmSpecification": {
                "SourceAlgorithms": [{"AlgorithmName": ALGORITHM, "ModelDataUrl": ARTIFACT}]
            },
        },
    )


def expect_create_model(stubber: Any, *, role_arn: str = ROLE) -> None:
    stubber.add_response(
        "create_model",
        {"ModelArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model/{MODEL}"},
        {
            "ModelName": MODEL,
            "PrimaryContainer": {"ModelPackageName": PACKAGE_ARN},
            "ExecutionRoleArn": role_arn,
            "EnableNetworkIsolation": True,
        },
    )


def expect_create_endpoint_config(stubber: Any, *, instance_type: str = "ml.g5.xlarge") -> None:
    stubber.add_response(
        "create_endpoint_config",
        {"EndpointConfigArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:endpoint-config/{CONFIG}"},
        {
            "EndpointConfigName": CONFIG,
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": MODEL,
                    "InitialInstanceCount": 1,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
            "AsyncInferenceConfig": {
                "OutputConfig": {
                    "S3OutputPath": f"{ASYNC_ROOT}/outputs/",
                    "S3FailurePath": f"{ASYNC_ROOT}/failures/",
                },
                "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1},
            },
        },
    )


def expect_create_endpoint(stubber: Any) -> None:
    stubber.add_response(
        "create_endpoint",
        {"EndpointArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:endpoint/{MODEL}"},
        {"EndpointName": MODEL, "EndpointConfigName": CONFIG},
    )


def expect_every_deploy_create(stubber: Any) -> None:
    expect_create_model_package(stubber)
    expect_create_model(stubber)
    expect_create_endpoint_config(stubber)
    expect_create_endpoint(stubber)
