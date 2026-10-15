"""The fine-tuner's job lifecycle (``synthefy.nori_fine_tune``): fine_tune, status, wait, stop, logs, curves and
the report through the tuner, and the AWS clients it builds. No test reaches AWS; the fakes are in
``_fine_tune_fakes``."""

import os
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Dict

import pytest
from botocore.exceptions import ReadTimeoutError
from synthefy import _aws as aws_module
from synthefy.errors import APITimeoutError, FineTuningError
from synthefy.nori_data_models import FineTuneJob
from synthefy.nori_fine_tune import SynthefyNoriFineTuner
from synthefy._training_jobs import build_job_name
from synthefy._training_logs import CURVE_KEYS

from ._fine_tune_fakes import (
    ALGORITHM,
    ARTIFACT,
    ARTIFACT_KEY,
    BUCKET,
    JOB,
    JOB_ARN,
    MODEL,
    NOW,
    REGION,
    ACCOUNT,
    STREAM,
    TRAIN,
    OUTPUT,
    AWSError,
    build_artifact,
    build_channel,
    build_completed,
    build_description,
    build_expected_training_job,
    build_hyperparameters,
    build_report,
    build_request,
    install_fake_aws,
    store_artifact,
)


@pytest.fixture
def aws(monkeypatch):
    with install_fake_aws(monkeypatch) as fakes:
        yield fakes


def test_importing_the_fine_tuner_loads_neither_boto3_nor_torch():
    code = (
        "import sys, synthefy.nori_fine_tune\n"
        "print(sorted(name for name in ('boto3', 'botocore', 'torch') if name in sys.modules))\n"
    )
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=environment, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_default_job_name_fits_sagemakers_63_characters_for_the_longest_model_name():
    name = build_job_name("ft-" + "a" * 60, NOW)
    assert name == "nori-ft-" + "a" * 39 + "-20261001-120000"
    assert len(name) == 63
    assert build_request(job_name=name).job_name == name


def test_the_fakes_refuse_a_parameter_the_installed_botocore_does_not_know(aws):
    with pytest.raises(AssertionError, match='Unknown parameter in input: "TrainingJobNames"'):
        aws.sm.describe_training_job(TrainingJobNames=JOB)
    with pytest.raises(AssertionError, match='Missing required parameter in input: "logStreamName"'):
        aws.logs.get_log_events(logGroupName="/aws/sagemaker/TrainingJobs", startFromHead=True)


# --------------------------------------------------------------------------- #
# fine_tune
# --------------------------------------------------------------------------- #


def test_fine_tune_sends_the_whole_training_job_request_and_returns_the_described_job(aws):
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, build_expected_training_job())
    aws.sm.training_jobs = [build_description("InProgress", SecondaryStatus="Starting")]

    job = aws.tuner.fine_tune(build_request())

    assert aws.sm.operations() == ["create_training_job", "describe_training_job"]
    assert (job.job_name, job.job_arn, job.status, job.secondary_status) == (JOB, JOB_ARN, "InProgress", "Starting")
    assert (job.algorithm, job.model_name, job.artifact_uri) == (ALGORITHM, MODEL, None)


def test_fine_tune_sends_both_channels_the_kms_key_and_every_override(aws):
    validation = f"s3://{BUCKET}/nori/validation/holdout.csv"
    hyperparameters = build_hyperparameters(
        base_model="nori-100m", feature_columns=["tenure", "plan"], bag=2, budget_s=60.0
    )
    expected = build_expected_training_job(
        TrainingJobName="churn-run-7",
        HyperParameters={
            **build_expected_training_job()["HyperParameters"],
            "base_model": "nori-100m",
            "feature_columns": '["tenure", "plan"]',
            "bag": "2",
            "budget_s": "60.0",
        },
        InputDataConfig=[build_channel("train", TRAIN), build_channel("validation", validation)],
        OutputDataConfig={"S3OutputPath": OUTPUT, "KmsKeyId": "alias/nori"},
        ResourceConfig={"InstanceType": "ml.g6e.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 50},
        StoppingCondition={"MaxRuntimeInSeconds": 1_980},
    )
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, expected)
    aws.sm.training_jobs = [build_description("InProgress")]

    aws.tuner.fine_tune(
        build_request(
            validation_data=validation,
            hyperparameters=hyperparameters,
            job_name="churn-run-7",
            volume_size_gb=50,
            kms_key_id="alias/nori",
        )
    )


def test_fine_tune_refuses_an_algorithm_from_another_region_naming_both_before_any_call(aws):
    foreign = f"arn:aws:sagemaker:eu-west-1:{ACCOUNT}:algorithm/nori-fine-tuning"
    with pytest.raises(ValueError, match="is in eu-west-1, but this fine-tuner calls SageMaker in us-east-1"):
        aws.tuner.fine_tune(build_request(algorithm=foreign))
    assert aws.sm.calls == []


def test_fine_tune_names_the_training_quota_when_sagemaker_refuses_the_instance(aws):
    aws.stubber.add_client_error(
        "create_training_job",
        service_error_code="ResourceLimitExceeded",
        service_message="The account-level service limit 'ml.g5.xlarge for training job usage' is 0 Instances.",
        expected_params=build_expected_training_job(),
    )
    with pytest.raises(FineTuningError) as raised:
        aws.tuner.fine_tune(build_request())
    message = str(raised.value)
    assert "'ml.g5.xlarge for training job usage' quota in Service Quotas" in message
    assert raised.value.job_name == JOB
    assert "is 0 Instances" in raised.value.failure_reason
    assert aws.sm.operations() == ["create_training_job"]


def test_fine_tune_keeps_the_job_name_and_arn_when_the_first_status_read_fails_after_the_job_started(aws):
    timeout = ReadTimeoutError(endpoint_url=f"https://api.sagemaker.{REGION}.amazonaws.com/")
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, build_expected_training_job())
    aws.sm.training_jobs = [timeout]

    with pytest.raises(FineTuningError) as raised:
        aws.tuner.fine_tune(build_request())

    assert (raised.value.job_name, raised.value.job_arn) == (JOB, JOB_ARN)
    assert raised.value.__cause__ is timeout
    message = str(raised.value)
    assert f"SageMaker started training job {JOB} ({JOB_ARN})" in message
    assert f"Call get_status('{JOB}') to reconnect to it" in message
    assert "calling fine_tune() again would start a second job" in message
    assert aws.sm.operations() == ["create_training_job", "describe_training_job"]


def test_fine_tune_refuses_a_one_csv_channel_whose_key_other_objects_start_with_naming_them(aws):
    training = f"s3://{BUCKET}/nori/data/train.csv"
    for key in ("nori/data/train.csv", "nori/data/train.csv.backup.csv", "nori/data/train.csv.old/part.csv"):
        aws.s3.objects[(BUCKET, key)] = b"x,y\n1,2\n"

    with pytest.raises(ValueError) as raised:
        aws.tuner.fine_tune(build_request(training_data=training))

    message = str(raised.value)
    assert f"training_data='{training}' names one .csv object, but SageMaker reads every key" in message
    assert f"s3://{BUCKET}/nori/data/train.csv.backup.csv, s3://{BUCKET}/nori/data/train.csv.old/part.csv" in message
    assert aws.s3.calls == [("list_objects_v2", "nori/data/train.csv")]
    assert aws.sm.calls == []


def test_fine_tune_accepts_a_one_csv_channel_that_no_other_key_starts_with(aws):
    training = f"s3://{BUCKET}/nori/data/train.csv"
    aws.s3.objects[(BUCKET, "nori/data/train.csv")] = b"x,y\n1,2\n"
    aws.s3.objects[(BUCKET, "nori/data/train-old.csv")] = b"x,y\n1,2\n"
    expected = build_expected_training_job(InputDataConfig=[build_channel("train", training)])
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, expected)
    aws.sm.training_jobs = [build_description("InProgress")]

    assert aws.tuner.fine_tune(build_request(training_data=training)).job_name == JOB


def test_fine_tune_submits_the_hyperparameters_as_built_after_the_callers_object_changes(aws):
    hyperparameters = build_hyperparameters()
    request = build_request(hyperparameters=hyperparameters)
    hyperparameters.base_model = "nori-100m"
    hyperparameters.budget_s = 100_000.0
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, build_expected_training_job())
    aws.sm.training_jobs = [build_description("InProgress")]

    assert aws.tuner.fine_tune(request).job_name == JOB


@pytest.mark.parametrize(
    "field, value, problem",
    [
        ("base_model", "nori-100m", "needs a 48 GB GPU, which 'ml.g5.xlarge' does not have"),
        ("budget_s", 100_000.0, "max_runtime_s=12600 is below bag x budget_s=800000"),
    ],
    ids=["base-needs-a-larger-gpu", "budget-outgrows-the-runtime"],
)
def test_fine_tune_refuses_a_request_whose_hyperparameters_changed_after_it_was_built(aws, field, value, problem):
    request = build_request()
    setattr(request.hyperparameters, field, value)

    with pytest.raises(ValueError, match="hyperparameters were changed after it was built") as raised:
        aws.tuner.fine_tune(request)

    assert problem in str(raised.value)
    assert aws.sm.calls == [] and aws.s3.calls == []


# --------------------------------------------------------------------------- #
# status, wait, stop
# --------------------------------------------------------------------------- #


def test_status_passes_every_field_through_including_a_status_aws_adds_later(aws):
    aws.sm.training_jobs = [
        build_description(
            "Paused",
            SecondaryStatus="Hibernating",
            SecondaryStatusTransitions=[
                {"Status": "Starting", "StartTime": NOW, "StatusMessage": "Preparing the instances for training"},
                {"Status": "Training", "StartTime": NOW, "StatusMessage": "Training in progress."},
            ],
            FinalMetricDataList=[
                {"MetricName": "val_mse", "Value": 0.25, "Timestamp": NOW},
                {"MetricName": "step", "Value": 17.0, "Timestamp": NOW},
            ],
            ModelArtifacts={"S3ModelArtifacts": ARTIFACT},
            BillableTimeInSeconds=321,
            TrainingEndTime=NOW,
            FailureReason="none yet",
        )
    ]
    job = aws.tuner.get_status(JOB)
    assert job == FineTuneJob(
        job_name=JOB,
        job_arn=JOB_ARN,
        status="Paused",
        secondary_status="Hibernating",
        status_message="Training in progress.",
        created_at=NOW,
        ended_at=NOW,
        artifact_uri=ARTIFACT,
        failure_reason="none yet",
        latest_metrics={"val_mse": 0.25, "step": 17.0},
        algorithm=ALGORITHM,
        model_name=MODEL,
        billable_seconds=321,
    )


def test_wait_returns_a_failed_job_with_its_capacity_reason_instead_of_raising(aws):
    reason = "CapacityError: Unable to provision requested ML compute capacity. Please retry later."
    aws.sm.training_jobs = [
        build_description("InProgress", SecondaryStatus="Starting"),
        build_description("Failed", FailureReason=reason),
    ]
    job = aws.tuner.wait(JOB)
    assert (job.status, job.failure_reason) == ("Failed", reason)
    assert aws.clock.sleeps == [60.0]


def test_wait_logs_each_status_change_once_with_sagemakers_message(aws, caplog):
    starting = [{"Status": "Starting", "StartTime": NOW, "StatusMessage": "Preparing the instances for training"}]
    training = starting + [{"Status": "Training", "StartTime": NOW, "StatusMessage": "Training in progress."}]
    aws.sm.training_jobs = [
        build_description("InProgress", SecondaryStatus="Starting", SecondaryStatusTransitions=starting),
        build_description("InProgress", SecondaryStatus="Starting", SecondaryStatusTransitions=starting),
        build_description("InProgress", SecondaryStatus="Training", SecondaryStatusTransitions=training),
        build_completed(SecondaryStatusTransitions=training),
    ]
    with caplog.at_level("INFO", logger="synthefy.fine_tune"):
        aws.tuner.wait(JOB, poll_s=10)
    messages = [record.getMessage() for record in caplog.records if record.name == "synthefy.fine_tune"]
    assert messages == [
        f"training job {JOB}: InProgress / Starting: Preparing the instances for training",
        f"training job {JOB}: InProgress / Training: Training in progress.",
        f"training job {JOB}: Completed / Completed: Training in progress.",
    ]
    assert aws.clock.sleeps == [10.0, 10.0, 10.0]


def test_wait_default_deadline_is_the_max_runtime_plus_two_hours_and_the_job_keeps_running(aws):
    aws.sm.training_jobs = [build_description("InProgress", StoppingCondition={"MaxRuntimeInSeconds": 100})]
    with pytest.raises(APITimeoutError, match="still InProgress after 7300 s; it keeps running") as raised:
        aws.tuner.wait(JOB)
    assert f"get_status('{JOB}')" in str(raised.value)
    assert f"stop('{JOB}')" in str(raised.value)
    assert aws.clock.now == 7_300
    assert "stop_training_job" not in aws.sm.operations()


def test_stop_requests_the_stop_and_returns_the_job_as_described_right_after(aws):
    aws.sm.training_jobs = [build_description("Stopping")]
    job = aws.tuner.stop(JOB)
    assert aws.sm.calls == [
        ("stop_training_job", {"TrainingJobName": JOB}),
        ("describe_training_job", {"TrainingJobName": JOB}),
    ]
    assert job.status == "Stopping"


def test_stop_of_a_job_that_already_ended_returns_it(aws):
    aws.sm.failures["stop_training_job"] = AWSError(
        "ValidationException", "The request was rejected because the training job is in status Completed."
    )
    aws.sm.training_jobs = [build_completed()]
    assert aws.tuner.stop(JOB).status == "Completed"


# --------------------------------------------------------------------------- #
# logs and curves through the tuner
# --------------------------------------------------------------------------- #


def test_logs_follow_reads_new_lines_until_the_job_ends_then_reads_once_more(aws):
    aws.logs.streams = {STREAM: ["starting"]}
    aws.sm.training_jobs = [build_description("InProgress"), build_description("InProgress"), build_completed()]
    appended = iter(["step line", "final line"])
    aws.clock.on_sleep = lambda _seconds: aws.logs.streams[STREAM].append(next(appended))

    lines = list(aws.tuner.get_logs(JOB, follow=True, poll_s=1))

    assert lines == ["starting", "step line", "final line"]
    assert aws.clock.sleeps == [5.0, 5.0]


def test_logs_without_follow_return_the_lines_so_far_without_reading_the_job(aws):
    aws.logs.streams = {STREAM: ["a1", "a2", "a3"]}
    assert list(aws.tuner.get_logs(JOB)) == ["a1", "a2", "a3"]
    assert aws.sm.calls == []


def test_curves_of_a_running_job_come_from_its_log_without_reading_the_job_or_s3(aws):
    aws.logs.streams = {STREAM: ["nori-train: starting", "child=0 step=1 train_loss=0.9 val_mse=0.5"]}
    assert aws.tuner.get_training_curves(JOB) == {"child": [0], "step": [1], "train_loss": [0.9], "val_mse": [0.5]}
    assert aws.sm.calls == [] and aws.s3.calls == []


def test_curves_fall_back_to_the_report_when_the_log_stream_is_gone(aws):
    aws.sm.training_jobs = [build_completed()]
    store_artifact(aws.s3, build_artifact(build_report()))
    assert aws.tuner.get_training_curves(JOB) == {
        "child": [0, 0, 0, 1, 1, 1],
        "step": [1, 2, 3, 1, 2, 3],
        "train_loss": [0.9, None, 0.7, 0.9, None, 0.7],
        "val_mse": [0.5, 0.45, 0.4, 0.5, 0.45, 0.4],
    }


def test_curves_of_a_job_without_log_lines_or_artifact_are_empty(aws):
    aws.logs.group_exists = False
    aws.sm.training_jobs = [build_description("InProgress", SecondaryStatus="Starting")]
    assert aws.tuner.get_training_curves(JOB) == {key: [] for key in CURVE_KEYS}


# --------------------------------------------------------------------------- #
# the report through the tuner
# --------------------------------------------------------------------------- #


def test_report_refuses_a_job_without_an_artifact_with_its_failure_reason(aws):
    aws.sm.training_jobs = [build_description("Failed", FailureReason="the training table has too few rows")]
    with pytest.raises(FineTuningError, match="has no model artifact") as raised:
        aws.tuner.get_training_report(JOB)
    assert (raised.value.status, raised.value.job_name) == ("Failed", JOB)
    assert "too few rows" in raised.value.failure_reason
    assert aws.s3.calls == []


def test_report_is_read_from_s3_once_per_artifact(aws):
    aws.sm.training_jobs = [build_completed()]
    store_artifact(aws.s3, build_artifact(build_report()))
    assert aws.tuner.get_training_report(JOB) == aws.tuner.get_training_report(JOB)
    assert aws.s3.calls == [("get_object", ARTIFACT_KEY)]


# --------------------------------------------------------------------------- #
# The AWS clients
# --------------------------------------------------------------------------- #


def test_fine_tuner_builds_three_clients_from_an_argument_free_session_and_closes_them_all(monkeypatch):
    capture: Dict[str, Any] = {}

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            capture["config"] = kwargs

    class FakeClient:
        def __init__(self, name: str, region_name: str) -> None:
            self.name = name
            self.meta = SimpleNamespace(region_name=region_name)

        def close(self) -> None:
            capture.setdefault("closed", []).append(self.name)

    class FakeSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            capture["session"] = (args, kwargs)

        def client(self, service_name: str, **kwargs: Any) -> FakeClient:
            capture.setdefault("clients", []).append((service_name, kwargs))
            return FakeClient(service_name, kwargs["region_name"])

    class FakeBoto3:
        Session = FakeSession

    monkeypatch.setattr(aws_module, "load_aws_sdk", lambda: (FakeBoto3, FakeConfig))

    with SynthefyNoriFineTuner(region_name="eu-west-1", timeout=75.0, max_retries=3, user_agent="acme") as tuner:
        assert tuner.region_name == "eu-west-1"

    assert capture["session"] == ((), {})
    assert [name for name, _ in capture["clients"]] == ["sagemaker", "logs", "s3"]
    assert all(kwargs["region_name"] == "eu-west-1" for _, kwargs in capture["clients"])
    assert capture["config"] == {
        "connect_timeout": 75.0,
        "read_timeout": 75.0,
        "retries": {"mode": "standard", "total_max_attempts": 4},
        "user_agent_extra": "acme",
    }
    assert capture["closed"] == ["sagemaker", "logs", "s3"]


def test_fine_tuner_without_boto3_asks_for_the_aws_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(ImportError, match=re.escape('pip install "synthefy[aws]"')):
        SynthefyNoriFineTuner(region_name=REGION)


@pytest.mark.parametrize("stream_count", [1, 2])
def test_curves_use_the_report_after_log_events_expire_but_streams_remain(aws, stream_count):
    aws.logs.streams = {f"{JOB}/algo-{index + 1}-1759320000": [] for index in range(stream_count)}
    aws.sm.training_jobs = [build_completed()]
    store_artifact(aws.s3, build_artifact(build_report()))

    assert aws.tuner.get_training_curves(JOB) == {
        "child": [0, 0, 0, 1, 1, 1],
        "step": [1, 2, 3, 1, 2, 3],
        "train_loss": [0.9, None, 0.7, 0.9, None, 0.7],
        "val_mse": [0.5, 0.45, 0.4, 0.5, 0.45, 0.4],
    }
    assert aws.s3.calls == [("get_object", ARTIFACT_KEY)]


def test_curves_of_an_empty_stream_without_an_artifact_remain_empty(aws):
    aws.logs.streams = {STREAM: []}
    aws.sm.training_jobs = [build_description("InProgress", SecondaryStatus="Starting")]

    assert aws.tuner.get_training_curves(JOB) == {key: [] for key in CURVE_KEYS}
    assert aws.s3.calls == []


def test_curves_keep_progress_only_logs_without_reading_a_report(aws):
    aws.logs.streams = {STREAM: ["nori-train: starting"]}

    assert aws.tuner.get_training_curves(JOB) == {key: [] for key in CURVE_KEYS}
    assert aws.sm.calls == [] and aws.s3.calls == []
