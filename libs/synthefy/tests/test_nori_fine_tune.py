"""The fine-tuner's job lifecycle (``synthefy.nori_fine_tune``): fine_tune, status, wait, stop, logs, curves and
the report through the tuner, and the AWS clients it builds. No test reaches AWS; the fakes are in
``_fine_tune_fakes``."""

import copy
import json
import math
import os
import re
import subprocess
import sys
import threading
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Dict

import pytest
from botocore.exceptions import NoCredentialsError, ReadTimeoutError
from synthefy import _aws as aws_module
from synthefy.errors import APITimeoutError, FineTuningError
from synthefy.nori_data_models import FineTuneHyperparameters, FineTuneJob, TrainingReport
from synthefy.nori_fine_tune import SynthefyNoriFineTuner
from synthefy import _training_jobs as training_jobs_module
from synthefy._training_jobs import HYPERPARAMETERS_EVERY_VERSION_READS, build_job_name, revalidate_request
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
    RESOURCE_NAME_TOKEN,
    TRAIN,
    OUTPUT,
    AWSError,
    build_artifact,
    build_channel,
    build_completed,
    build_description,
    build_expected_training_job,
    build_curve,
    build_hyperparameters,
    build_report,
    build_request,
    CURRENT_ALGORITHM_HYPERPARAMETERS,
    IMPLICIT_RESUME_ALGORITHM_HYPERPARAMETERS,
    LEGACY_ALGORITHM_HYPERPARAMETERS,
    build_algorithm_description,
    build_parquet_file,
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
    name = build_job_name("ft-" + "a" * 60, NOW, RESOURCE_NAME_TOKEN)
    assert name == "nori-ft-" + "a" * 32 + "-20261001-120000-" + RESOURCE_NAME_TOKEN
    assert len(name) == 63
    assert build_request(job_name=name).job_name == name


def test_default_job_names_carry_the_token_so_two_launches_in_one_second_differ():
    assert build_job_name(MODEL, NOW, "0a1b2c") != build_job_name(MODEL, NOW, "3d4e5f")


def test_a_derived_runtime_follows_the_hyperparameters_when_checked_again_and_a_set_one_stays():
    # The README's several-models example copies a request with another model's hyperparameters, and a buyer may raise
    # max_training_hours in a copy: a runtime derived from the old hours would stop the job first.
    short = build_request(hyperparameters=build_hyperparameters(bag=2, max_training_hours=1))
    assert short.max_runtime_s == 5400
    other = build_hyperparameters(model_name="ft-acme-ltv-v1")
    assert revalidate_request(short.model_copy(update={"hyperparameters": other})).max_runtime_s == 88200
    longer = short.hyperparameters.model_copy(update={"max_training_hours": 2})
    assert revalidate_request(short.model_copy(update={"hyperparameters": longer})).max_runtime_s == 9000
    pinned = build_request(hyperparameters=build_hyperparameters(max_training_hours=1), max_runtime_s=600)
    assert revalidate_request(pinned.model_copy(update={"hyperparameters": other})).max_runtime_s == 600


def test_the_fakes_refuse_a_parameter_the_installed_botocore_does_not_know(aws):
    with pytest.raises(AssertionError, match='Unknown parameter in input: "TrainingJobNames"'):
        aws.sm.describe_training_job(TrainingJobNames=JOB)
    with pytest.raises(AssertionError, match='Missing required parameter in input: "logStreamName"'):
        aws.logs.get_log_events(logGroupName="/aws/sagemaker/TrainingJobs", startFromHead=True)


# --------------------------------------------------------------------------- #
# fine_tune
# --------------------------------------------------------------------------- #


def test_fine_tune_sends_the_whole_training_job_request_and_returns_the_described_job(aws):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, build_expected_training_job())
    aws.sm.training_jobs = [build_description("InProgress", SecondaryStatus="Starting")]

    job = aws.tuner.fine_tune(build_request())

    assert aws.sm.operations() == ["describe_algorithm", "create_training_job", "describe_training_job"]
    assert (job.job_name, job.job_arn, job.status, job.secondary_status) == (JOB, JOB_ARN, "InProgress", "Starting")
    assert (job.algorithm, job.model_name, job.artifact_uri) == (ALGORITHM, MODEL, None)


def test_fine_tune_sends_both_channels_the_kms_key_and_every_override(aws):
    _put_table(aws, "nori/train/part-0.parquet", [5000], columns=["churn", "tenure", "plan"])
    validation = f"s3://{BUCKET}/nori/validation/holdout.parquet"
    checkpoints = f"s3://{BUCKET}/nori/checkpoints/churn/"
    hyperparameters = build_hyperparameters(
        base_model="nori-100m", feature_columns=["tenure", "plan"], bag=2, max_training_hours=0.05
    )
    expected = build_expected_training_job(
        TrainingJobName="churn-run-7",
        HyperParameters={
            **build_expected_training_job()["HyperParameters"],
            "base_model": "nori-100m",
            "feature_columns": '["tenure", "plan"]',
            "bag": "2",
            "max_training_hours": "0.05",
        },
        InputDataConfig=[build_channel("train", TRAIN), build_channel("validation", validation)],
        OutputDataConfig={"S3OutputPath": OUTPUT, "KmsKeyId": "alias/nori"},
        ResourceConfig={"InstanceType": "ml.g6e.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 50},
        StoppingCondition={"MaxRuntimeInSeconds": 1_980},
        # No LocalPath: the job finds the folder only at SageMaker's default.
        CheckpointConfig={"S3Uri": checkpoints},
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
            checkpoint_uri=checkpoints,
        )
    )


def test_fine_tune_refuses_an_algorithm_from_another_region_naming_both_before_any_call(aws):
    foreign = f"arn:aws:sagemaker:eu-west-1:{ACCOUNT}:algorithm/nori-fine-tuning"
    with pytest.raises(ValueError, match="is in eu-west-1, but this fine-tuner calls SageMaker in us-east-1"):
        aws.tuner.fine_tune(build_request(algorithm=foreign))
    assert aws.sm.calls == []


def test_fine_tune_refuses_a_max_runtime_shorter_than_the_cap_before_creating_the_job(aws):
    request = build_request(hyperparameters=build_hyperparameters(max_training_hours=1), max_runtime_s=600)
    with pytest.raises(ValueError, match="ends the job before max_training_hours 1 stops its models"):
        aws.tuner.fine_tune(request)
    assert aws.sm.operations() == ["describe_algorithm"]


def test_fine_tune_names_the_training_quota_when_sagemaker_refuses_the_instance(aws):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
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
    assert aws.sm.operations() == ["describe_algorithm", "create_training_job"]


def test_fine_tune_keeps_the_job_name_and_arn_when_the_first_status_read_fails_after_the_job_started(aws):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
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
    assert "calling fine_tune() or resume() again would start a second job" in message
    assert aws.sm.operations() == ["describe_algorithm", "create_training_job", "describe_training_job"]


@pytest.mark.parametrize("method", ["fine_tune", "get_training_plan"])
def test_fine_tune_refuses_a_one_object_channel_whose_key_other_objects_start_with_naming_them(aws, method):
    # A plan of the one object would count fewer rows than the job reads, so the plan refuses it too.
    training = f"s3://{BUCKET}/nori/data/train.parquet"
    for key in (
        "nori/data/train.parquet",
        "nori/data/train.parquet.backup.parquet",
        "nori/data/train.parquet.old/part.parquet",
    ):
        aws.s3.objects[(BUCKET, key)] = b"PAR1"

    with pytest.raises(ValueError) as raised:
        getattr(aws.tuner, method)(build_request(training_data=training))

    message = str(raised.value)
    assert f"training_data='{training}' names one object, but SageMaker reads every key" in message
    assert (
        f"s3://{BUCKET}/nori/data/train.parquet.backup.parquet, s3://{BUCKET}/nori/data/train.parquet.old/part.parquet"
        in message
    )
    assert aws.s3.calls == [("list_objects_v2", "nori/data/train.parquet")]
    assert aws.sm.calls == []


def test_fine_tune_refuses_a_csv_object_before_any_aws_call(aws):
    # model_copy skips validation, so the request reaches fine_tune() with the CSV object in it.
    training = f"s3://{BUCKET}/nori/data/train.csv"
    request = build_request().model_copy(update={"training_data": training})

    with pytest.raises(ValueError) as raised:
        aws.tuner.fine_tune(request)

    message = str(raised.value)
    assert f"training_data names a CSV object; got '{training}'" in message
    assert "pandas.DataFrame.to_parquet(path, index=False)" in message
    assert aws.s3.calls == [] and aws.sm.calls == []


def test_fine_tune_accepts_a_one_object_channel_that_no_other_key_starts_with(aws):
    training = f"s3://{BUCKET}/nori/data/train.parquet"
    aws.s3.objects[(BUCKET, "nori/data/train.parquet")] = build_parquet_file(_COLUMNS, [5000])
    aws.s3.objects[(BUCKET, "nori/data/train-old.parquet")] = b"PAR1"
    expected = build_expected_training_job(InputDataConfig=[build_channel("train", training)])
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, expected)
    aws.sm.training_jobs = [build_description("InProgress")]

    assert aws.tuner.fine_tune(build_request(training_data=training)).job_name == JOB


def test_fine_tune_submits_the_hyperparameters_as_built_after_the_callers_object_changes(aws):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    hyperparameters = build_hyperparameters()
    request = build_request(hyperparameters=hyperparameters)
    hyperparameters.base_model = "nori-100m"
    hyperparameters.max_training_hours = 100.0
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, build_expected_training_job())
    aws.sm.training_jobs = [build_description("InProgress")]

    assert aws.tuner.fine_tune(request).job_name == JOB


@pytest.mark.parametrize(
    "field, value, problem",
    [
        ("base_model", "nori-100m", "needs a 48 GB GPU, which 'ml.g5.xlarge' does not have"),
        ("feature_columns", ["churn"], "feature_columns includes the target column 'churn'"),
    ],
    ids=["base-needs-a-larger-gpu", "target-among-the-features"],
)
def test_fine_tune_refuses_a_request_whose_hyperparameters_changed_after_it_was_built(aws, field, value, problem):
    request = build_request()
    setattr(request.hyperparameters, field, value)

    with pytest.raises(ValueError, match="hyperparameters were changed after it was built") as raised:
        aws.tuner.fine_tune(request)

    assert problem in str(raised.value)
    assert aws.sm.calls == [] and aws.s3.calls == []


# --------------------------------------------------------------------------- #
# the training plan and the Algorithm version
# --------------------------------------------------------------------------- #

_COLUMNS = ["churn"] + [f"x{i}" for i in range(20)]


def _put_table(aws, key: str, row_groups, columns=_COLUMNS) -> None:
    aws.s3.objects[(BUCKET, key)] = build_parquet_file(columns, row_groups)


@pytest.mark.parametrize(
    "tail_bytes, gets_per_file",
    [
        (training_jobs_module._FOOTER_TAIL_BYTES, 1),  # the tail of each file, which holds its footer
        (64, 2),  # a tail shorter than each footer, then the footer
    ],
    ids=["footer-in-tail", "footer-past-tail"],
)
def test_training_plan_reads_every_page_of_a_folder_and_skips_its_markers(aws, monkeypatch, tail_bytes, gets_per_file):
    # A folder of more keys than one ListObjectsV2 page: a plan from the first page alone counts too few rows.
    monkeypatch.setattr(training_jobs_module, "_LIST_PAGE_KEYS", 2)
    monkeypatch.setattr(training_jobs_module, "_FOOTER_TAIL_BYTES", tail_bytes)
    for part in range(3):
        _put_table(aws, f"nori/train/part-{part}.parquet", [1000, 500])
    aws.s3.objects[(BUCKET, "nori/train/_SUCCESS")] = b""
    aws.s3.objects[(BUCKET, "nori/train/.part-0.parquet.crc")] = b"crc"

    plan = aws.tuner.get_training_plan(build_request(hyperparameters=build_hyperparameters(bag=2)))

    assert (plan.n_rows_in_table, plan.n_feature_columns, plan.bag) == (4500, 20, 2)
    assert plan.rows_per_step_is_estimate
    assert [call for call in aws.s3.calls if call[0] == "list_objects_v2"] == [("list_objects_v2", "nori/train/")] * 3
    assert sum(call[0] == "get_object" for call in aws.s3.calls) == gets_per_file * 3


def test_the_training_plan_reads_a_folders_footers_side_by_side(aws, monkeypatch):
    # Each read waits at the barrier for the other: reading the files one at a time breaks it after 10 seconds.
    for part in range(2):
        _put_table(aws, f"nori/train/part-{part}.parquet", [50_000])
    barrier = threading.Barrier(2, timeout=10)
    get_object = aws.s3.get_object

    def get_object_together(**kwargs):
        barrier.wait()
        return get_object(**kwargs)

    monkeypatch.setattr(aws.s3, "get_object", get_object_together)

    plan = aws.tuner.get_training_plan(build_request())

    assert plan.n_rows_in_table == 100_000


def test_a_footer_pyarrow_cannot_read_raises_the_documented_value_error(aws):
    body = b"\x15" * 20
    aws.s3.objects[(BUCKET, "nori/train/part-0.parquet")] = b"PAR1" + body + len(body).to_bytes(4, "little") + b"PAR1"

    with pytest.raises(ValueError, match=r"could not read s3://customer-bucket/nori/train/part-0\.parquet as Parquet"):
        aws.tuner.get_training_plan(build_request())


def test_fine_tune_refuses_a_file_whose_pandas_metadata_is_not_json_before_creating_the_job(aws):
    # The job cannot read the file as a frame and refuses it, so a job submitted without a plan would fail its start.
    pa, pq = aws_module.load_parquet_reader()
    table = pa.table({column: [0.0] * 5000 for column in _COLUMNS}).replace_schema_metadata({b"pandas": b"{not json"})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    aws.s3.objects[(BUCKET, "nori/train/part-0.parquet")] = sink.getvalue().to_pybytes()

    with pytest.raises(ValueError, match=r"could not read s3://customer-bucket/nori/train/part-0\.parquet as Parquet"):
        aws.tuner.fine_tune(build_request())

    assert aws.sm.operations() == ["describe_algorithm"]


def test_fine_tune_without_credentials_for_the_channels_submits_the_job_without_a_plan(aws, monkeypatch, caplog):
    def get_object(**kwargs):
        raise NoCredentialsError()

    monkeypatch.setattr(aws.s3, "get_object", get_object)
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, build_expected_training_job())
    aws.sm.training_jobs = [build_description("InProgress")]

    with caplog.at_level("WARNING", logger="synthefy.fine_tune"):
        job = aws.tuner.fine_tune(build_request())

    assert job.training_plan is None
    assert "so the job is submitted without a plan: Unable to locate credentials" in caplog.text


def test_fine_tune_with_a_footer_longer_than_the_plan_reads_submits_the_job_without_a_plan(aws, monkeypatch, caplog):
    # Only the plan's reader has the limit: the job reads the file whole.
    monkeypatch.setattr(training_jobs_module, "_FOOTER_TAIL_BYTES", 64)
    monkeypatch.setattr(training_jobs_module, "_MAX_FOOTER_BYTES", 64)
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, build_expected_training_job())
    aws.sm.training_jobs = [build_description("InProgress")]

    with caplog.at_level("WARNING", logger="synthefy.fine_tune"):
        job = aws.tuner.fine_tune(build_request())

    assert job.training_plan is None
    assert "bytes, which this reader does not read" in caplog.text


def test_fine_tune_with_a_plan_error_nobody_foresaw_submits_the_job_without_a_plan(aws, monkeypatch, caplog):
    # The plan is advice: only a table the job would refuse stops the request.
    def get_plan(*args, **kwargs):
        raise KeyError("a plan error nobody foresaw")

    monkeypatch.setattr("synthefy.nori_fine_tune.get_training_plan_from_footers", get_plan)
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, build_expected_training_job())
    aws.sm.training_jobs = [build_description("InProgress")]

    with caplog.at_level("WARNING", logger="synthefy.fine_tune"):
        job = aws.tuner.fine_tune(build_request())

    assert job.training_plan is None
    assert "could not work out the training plan, so the job is submitted without a plan" in caplog.text
    assert "KeyError: 'a plan error nobody foresaw'" in caplog.text


@pytest.mark.parametrize(
    "hyperparameters, problem",
    [
        ({"target_column": "label"}, "target_column 'label' is not a column of training_data"),
        ({"feature_columns": ["x0", "tenure"]}, "feature_columns not in training_data: ['tenure']"),
    ],
    ids=["target", "feature"],
)
def test_fine_tune_refuses_a_column_the_training_table_lacks_before_creating_the_job(aws, hyperparameters, problem):
    _put_table(aws, "nori/train/part-0.parquet", [5000])

    with pytest.raises(ValueError) as raised:
        aws.tuner.fine_tune(build_request(hyperparameters=build_hyperparameters(**hyperparameters)))

    assert problem in str(raised.value)
    assert "create_training_job" not in aws.sm.operations()


@pytest.mark.parametrize(
    "key, problem",
    [
        ("nori/train/_SUCCESS", "holds no .parquet or .pq file"),
        ("nori/train/data.csv", "holds data.csv, which is neither .parquet nor .pq"),
    ],
    ids=["no-parquet-file", "csv-file"],
)
def test_fine_tune_refuses_a_folder_the_job_would_refuse_before_creating_the_job(aws, key, problem):
    # The job skips names starting with _ and refuses a file that is not Parquet, so either folder fails its start.
    aws.s3.objects[(BUCKET, key)] = b"a,b\n1,2\n"

    with pytest.raises(ValueError, match=problem):
        aws.tuner.fine_tune(build_request())

    assert aws.sm.operations() == ["describe_algorithm"]


def test_fine_tune_attaches_the_plan_and_warns_when_it_outlasts_sagemakers_limit_without_a_checkpoint(aws, caplog):
    # 8 models on one GPU for 119.8 hours: within SageMaker's 120 hours, but not with the 30 minutes the job needs
    # besides training.
    _put_table(aws, "nori/train/part-0.parquet", [1_000_000] * 50)
    settings: Dict[str, Any] = {"max_training_hours": None, "validation_every_n_steps": 10}
    estimate = aws.tuner.get_training_plan(build_request(hyperparameters=build_hyperparameters(**settings)))
    rounds = math.ceil(estimate.bag / estimate.gpus)
    steps = math.floor(119.8 * 3600 / (rounds * estimate.seconds_per_step))
    hyperparameters = build_hyperparameters(n_steps_per_model=steps, **settings)
    expected = build_expected_training_job(StoppingCondition={"MaxRuntimeInSeconds": 432_000})
    expected["HyperParameters"].update(
        n_steps_per_model=str(steps), max_training_hours="none", validation_every_n_steps="10"
    )
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, expected)
    aws.sm.training_jobs = [build_description("InProgress")]

    with caplog.at_level("INFO", logger="synthefy.fine_tune"):
        with pytest.warns(UserWarning, match="passes SageMaker's limit of 120 hours.*set checkpoint_uri"):
            job = aws.tuner.fine_tune(build_request(hyperparameters=hyperparameters))

    assert job.training_plan is not None and 119.5 < job.training_plan.training_hours < 120
    assert job.training_plan.get_summary().splitlines()[0] in caplog.text


def test_fine_tune_on_the_fixed_steps_algorithm_sends_its_hyperparameters_and_describes_it_once(aws, caplog):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    aws.sm.algorithms = [build_algorithm_description(LEGACY_ALGORITHM_HYPERPARAMETERS)]
    hyperparameters = build_expected_training_job()["HyperParameters"]
    for name in ("row_partition_strategy", "max_training_hours"):
        del hyperparameters[name]
    # A caller who never chose a base trains the version's declared default, nori-6m, not the client's nori-1-5,
    # which that version does not offer.
    hyperparameters.update(steps="100", budget_s="900.0", base_model="nori-6m")
    # patience 25 goes as 25: at 100 steps that version scores every step.
    assert set(hyperparameters) == {name for name, _, _ in LEGACY_ALGORITHM_HYPERPARAMETERS} - {"feature_columns"}
    # Eight models on one GPU, 900 s each, with half again as margin, plus the read and the packaging.
    expected = build_expected_training_job(
        HyperParameters=hyperparameters, StoppingCondition={"MaxRuntimeInSeconds": 1350 * 8 + 1800}
    )
    for _ in range(2):
        aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, expected)
    aws.sm.training_jobs = [build_description("InProgress")]

    with caplog.at_level("WARNING", logger="synthefy.fine_tune"):
        for _ in range(2):
            aws.tuner.fine_tune(build_request())

    assert aws.sm.operations().count("describe_algorithm") == 1
    assert "trains each model for a fixed number of steps (100), not on the whole table" in caplog.text


def test_the_fixed_steps_algorithm_gets_patience_as_the_steps_between_that_many_of_its_scores():
    # That version scores every steps // 500 steps and counts patience in steps: 25 scores are 100 steps at 2,000.
    strings = training_jobs_module.get_serialized_legacy_hyperparameters(
        FineTuneHyperparameters(target_column="y", model_name="ft-acme-churn-v1", n_steps_per_model=2000, patience=25)
    )

    assert (strings["steps"], strings["patience"]) == ("2000", "100")


@pytest.mark.parametrize(
    "setting, problem",
    [
        ({"row_partition_strategy": "shard"}, "does not read row_partition_strategy; a newer version"),
        ({"n_passes_per_row": 2.0}, "does not read n_passes_per_row; a newer version"),
        ({"max_training_hours": 24.0}, "does not read max_training_hours; a newer version"),
        ({"validation_every_n_steps": 10}, "does not read validation_every_n_steps; a newer version"),
        ({"patience": None}, "patience=None turns it off only on a newer version"),
    ],
    ids=["strategy", "passes", "hours", "validation", "patience-none"],
)
def test_fine_tune_on_the_fixed_steps_algorithm_refuses_a_setting_only_a_newer_version_reads(aws, setting, problem):
    # A default equal to the value given is still refused: the caller asked for it.
    aws.sm.algorithms = [build_algorithm_description(LEGACY_ALGORITHM_HYPERPARAMETERS)]

    with pytest.raises(FineTuningError) as raised:
        aws.tuner.fine_tune(build_request(hyperparameters=build_hyperparameters(**setting)))

    assert problem in str(raised.value)
    assert aws.sm.operations() == ["describe_algorithm"]


def test_a_fixed_steps_version_is_sent_a_max_runtime_shorter_than_the_cap(aws):
    # That version reads no max_training_hours, so the runtime set is the job's only limit.
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    aws.sm.algorithms = [build_algorithm_description(LEGACY_ALGORITHM_HYPERPARAMETERS)]
    hyperparameters = build_expected_training_job()["HyperParameters"]
    for name in ("row_partition_strategy", "max_training_hours"):
        del hyperparameters[name]
    hyperparameters.update(steps="100", budget_s="900.0", base_model="nori-6m")
    expected = build_expected_training_job(
        HyperParameters=hyperparameters, StoppingCondition={"MaxRuntimeInSeconds": 600}
    )
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, expected)
    aws.sm.training_jobs = [build_description("InProgress")]

    aws.tuner.fine_tune(build_request(max_runtime_s=600))

    assert aws.sm.operations() == ["describe_algorithm", "create_training_job", "describe_training_job"]


@pytest.mark.parametrize("n_rows, refused", [(1250, False), (1100, True)])
def test_the_fixed_steps_algorithms_plan_needs_its_1000_training_rows_in_the_table_not_per_model(aws, n_rows, refused):
    # That version trains every model on the rows outside its fold: 1,250 rows at bag 8 leave 1,125 after the
    # hold-out, which it trains on, although no fold's share reaches 1,000; 1,100 leave 990, which it refuses.
    aws.sm.algorithms = [build_algorithm_description(LEGACY_ALGORITHM_HYPERPARAMETERS)]
    _put_table(aws, "nori/train/part-0.parquet", [n_rows])

    if refused:
        with pytest.raises(ValueError, match="990 after the report hold-out, fewer than the 1,000 training rows"):
            aws.tuner.get_training_plan(build_request())
    else:
        plan = aws.tuner.get_training_plan(build_request())
        assert (plan.bag, plan.row_partition_strategy, plan.n_training_rows) == (8, "folds", 1125)


def test_the_fixed_steps_algorithms_plan_stops_each_model_at_its_900_seconds(aws):
    # That version stops each model after 900 seconds: on 50M rows of 20 features, about 3 s a step, 298 of the
    # 2,000 steps asked for, and 2 hours for the 8 models on one GPU.
    aws.sm.algorithms = [build_algorithm_description(LEGACY_ALGORITHM_HYPERPARAMETERS)]
    _put_table(aws, "nori/train/part-0.parquet", [1_000_000] * 50)

    plan = aws.tuner.get_training_plan(build_request(hyperparameters=build_hyperparameters(n_steps_per_model=2000)))

    assert plan.steps_per_model == 298
    assert plan.training_hours == pytest.approx(2.0, rel=0.01)
    expected = "stops each model after 900 seconds, about 298 of its 2,000 steps"
    assert any(expected in warning for warning in plan.warnings)
    # This version refuses n_passes_per_row, and in 900 seconds no n_steps_per_model reads past 43.5% of this table.
    remedy = "of the training rows; a newer version of the Algorithm reads more"
    assert any(warning.endswith(remedy) for warning in plan.warnings)


@pytest.mark.parametrize(
    "description, interval, patience_steps",
    [
        # The whole-table version's default scores every 100 // 25 steps: 25 scores are 100 steps.
        (CURRENT_ALGORITHM_HYPERPARAMETERS, 4, 100),
        # The fixed-steps version scores every 5,000 // 500 steps: 25 scores are 250 steps.
        (LEGACY_ALGORITHM_HYPERPARAMETERS, 10, 250),
    ],
    ids=["whole-table", "fixed-steps"],
)
def test_the_plan_waits_patience_scores_at_each_versions_validation_interval(
    aws, description, interval, patience_steps
):
    aws.sm.algorithms = [build_algorithm_description(description)]
    _put_table(aws, "nori/train/part-0.parquet", [5000])

    plan = aws.tuner.get_training_plan(build_request(hyperparameters=build_hyperparameters(n_steps_per_model=5000)))

    assert (plan.validation_every_n_steps, plan.patience_steps) == (interval, patience_steps)
    assert f"patience (25 scores, {patience_steps} steps) can stop it sooner" in plan.get_summary()


@pytest.mark.parametrize("code, describes", [("AccessDeniedException", 1), ("ThrottlingException", 2)])
def test_an_algorithm_that_cannot_be_described_is_described_again_unless_denied_and_gets_the_common_settings(
    aws, caplog, code, describes
):
    # A denial lasts, so it is kept; a throttled call describes again on the next request. Either way the job gets
    # only the settings every version reads, and the runtime the request has.
    aws.sm.algorithms = [AWSError(code, "the call failed")]
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    hyperparameters = build_expected_training_job()["HyperParameters"]
    for name in ("row_partition_strategy", "max_training_hours"):
        del hyperparameters[name]
    assert set(hyperparameters) == HYPERPARAMETERS_EVERY_VERSION_READS - {"feature_columns"}
    expected = build_expected_training_job(HyperParameters=hyperparameters)
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, expected)
    aws.sm.training_jobs = [build_description("InProgress")]

    with caplog.at_level("WARNING", logger="synthefy.fine_tune"):
        plan = aws.tuner.get_training_plan(build_request())
        aws.tuner.fine_tune(build_request())

    assert aws.sm.operations().count("describe_algorithm") == describes
    assert sum("could not describe Algorithm" in record.getMessage() for record in caplog.records) == describes
    assert any("could not be described, so this plan is the newest version's" in warning for warning in plan.warnings)


def test_an_algorithm_version_that_does_not_offer_the_base_refuses_it_naming_the_bases_it_offers(aws):
    aws.sm.algorithms = [
        build_algorithm_description(CURRENT_ALGORITHM_HYPERPARAMETERS, offered_base_models=["nori-6m", "nori-100m"])
    ]

    with pytest.raises(FineTuningError, match="does not offer base_model='nori-1-5'") as raised:
        aws.tuner.get_training_plan(build_request(hyperparameters=build_hyperparameters(base_model="nori-1-5")))

    assert "it offers 'nori-6m', 'nori-100m'" in str(raised.value)
    assert aws.sm.operations() == ["describe_algorithm"]


def test_an_algorithm_that_cannot_be_described_refuses_version_specific_settings_by_name(aws):
    # The versions read these differently: sent, one version would refuse them; left out, the job would drop them
    # unnoticed.
    aws.sm.algorithms = [AWSError("AccessDeniedException", "not authorized to perform sagemaker:DescribeAlgorithm")]
    hyperparameters = build_hyperparameters(n_passes_per_row=2.0, row_partition_strategy="folds")

    with pytest.raises(FineTuningError, match="could not be described") as raised:
        aws.tuner.fine_tune(build_request(hyperparameters=hyperparameters))

    assert "row_partition_strategy, n_passes_per_row cannot be sent" in str(raised.value)
    assert aws.sm.operations() == ["describe_algorithm"]


def test_an_algorithm_that_cannot_be_described_refuses_n_steps_per_model_by_name(aws):
    # One version reads the setting as steps and refuses n_steps_per_model; the other refuses steps.
    aws.sm.algorithms = [AWSError("AccessDeniedException", "not authorized to perform sagemaker:DescribeAlgorithm")]
    _put_table(aws, "nori/train/part-0.parquet", [5000])

    with pytest.raises(FineTuningError, match="could not be described") as raised:
        aws.tuner.fine_tune(build_request(hyperparameters=build_hyperparameters(n_steps_per_model=2000)))

    assert "n_steps_per_model cannot be sent: the Algorithm's versions read it differently" in str(raised.value)
    assert aws.sm.operations() == ["describe_algorithm"]


def test_an_algorithm_that_cannot_be_described_refuses_a_max_runtime_shorter_than_the_default_cap(aws):
    # The newest version may be the one running, and it stops its models at the default max_training_hours, which
    # this request cannot lower.
    aws.sm.algorithms = [AWSError("AccessDeniedException", "not authorized to perform sagemaker:DescribeAlgorithm")]

    with pytest.raises(ValueError, match=r"leave max_runtime_s unset or set it to at least 88,200 \(.*\)$") as raised:
        aws.tuner.fine_tune(build_request(max_runtime_s=600))

    assert "max_runtime_s 600 ends the job before max_training_hours 24 stops its models" in str(raised.value)
    assert aws.sm.operations() == ["describe_algorithm"]


def test_the_clients_plan_steps_draw_the_largest_share_by_the_member_cycle():
    # Step s draws from member s % n_members, so steps that end on the member with fewer rows per step read less
    # than the mean rows per step say; this share is one row past what 121 steps draw.
    hp = FineTuneHyperparameters(
        target_column="y", model_name="ft-acme-churn-v1", row_partition_strategy="shard", bag=1
    )
    rows = [training_jobs_module.get_rows_and_targets_per_step(width, None)[0] for width in (40, 400)]
    share = 60 * sum(rows) + rows[1] + 1
    plan = training_jobs_module.get_training_plan_from_counts(
        hp,
        n_rows=share,
        n_validation_rows=700,
        n_features=3,
        largest_row_group=share,
        instance_type="ml.g5.xlarge",
        widths=[40, 400],
    )
    steps = plan.steps_per_model
    drawn = [sum(rows[s % 2] for s in range(1, n + 1)) for n in (steps - 1, steps)]
    assert drawn[0] < share <= drawn[1]
    assert plan.n_passes_per_row == drawn[1] / share
    # One step fewer ends on that member too; the plan's passes count the rows those steps draw.
    hp = FineTuneHyperparameters(
        target_column="y",
        model_name="ft-acme-churn-v1",
        row_partition_strategy="shard",
        bag=1,
        n_steps_per_model=steps - 1,
    )
    plan = training_jobs_module.get_training_plan_from_counts(
        hp,
        n_rows=share,
        n_validation_rows=700,
        n_features=3,
        largest_row_group=share,
        instance_type="ml.g5.xlarge",
        widths=[40, 400],
    )
    assert plan.n_passes_per_row == drawn[0] / share


# --------------------------------------------------------------------------- #
# the checkpoint folder: fine_tune(overwrite=...) and resume()
# --------------------------------------------------------------------------- #

_CHECKPOINTS = f"s3://{BUCKET}/nori/checkpoints/churn/"
_CHECKPOINT_PREFIX = "nori/checkpoints/churn/"
_EARLIER_JOB = f"nori-{MODEL}-20260930-080000-9f8e7d"
#: The record _fill_checkpoint_folder makes the folder's newest.
_NEWEST_RECORD_KEY = _CHECKPOINT_PREFIX + "child-2.step-500.json"


def _fill_checkpoint_folder(aws, writer: str = _EARLIER_JOB, *, extra=()) -> None:
    """Models 0 and 1 finished and model 2's state after step 500, its record the newest and naming ``writer``."""
    for name in ("child-0.pt", "child-1.pt", "child-2.step-500.pt", "child-2.step-500.best.pt", *extra):
        aws.s3.objects[(BUCKET, _CHECKPOINT_PREFIX + name)] = b"weights"
    for name in ("child-0.json", "child-1.json"):
        aws.s3.objects[(BUCKET, _CHECKPOINT_PREFIX + name)] = json.dumps({"training_job_name": "nori-older"}).encode()
    aws.s3.objects[(BUCKET, _NEWEST_RECORD_KEY)] = json.dumps({"training_job_name": writer, "saved_at": 1.0}).encode()
    # The earliest LastModified: the client ranks records by saved_at, never by a time S3 sets.
    aws.s3.modified[(BUCKET, _NEWEST_RECORD_KEY)] = NOW - timedelta(minutes=5)


def _checkpoint_keys(aws):
    return sorted(key for bucket, key in aws.s3.objects if key.startswith(_CHECKPOINT_PREFIX))


def test_fine_tune_refuses_a_checkpoint_folder_with_files_naming_its_models_step_and_job_before_creating(aws):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    _fill_checkpoint_folder(aws)
    kept = _checkpoint_keys(aws)

    with pytest.raises(FineTuningError) as raised:
        aws.tuner.fine_tune(build_request(checkpoint_uri=_CHECKPOINTS))

    message = str(raised.value)
    assert f"{_CHECKPOINTS} holds 2 finished models, the saved state of model 2 after step 500" in message
    assert f"its newest record was written by training job '{_EARLIER_JOB}'" in message
    assert (
        f"resume('{_EARLIER_JOB}')" in message and "overwrite=True" in message and "another checkpoint_uri" in message
    )
    assert "create_training_job" not in aws.sm.operations()
    assert _checkpoint_keys(aws) == kept


@pytest.mark.parametrize(
    "writer_answer",
    [
        build_description("Stopped", TrainingJobName=_EARLIER_JOB),
        AWSError("ValidationException", "Requested resource not found."),
    ],
    ids=["stopped writer", "writer SageMaker does not know"],
)
def test_fine_tune_with_overwrite_deletes_the_checkpoint_files_then_creates_the_job(aws, monkeypatch, writer_answer):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    _fill_checkpoint_folder(aws)
    aws.s3.objects[(BUCKET, "nori/checkpoints/churn-old/child-0.pt")] = b"weights"  # shares the text, not the folder
    aws.stubber.add_response(
        "create_training_job",
        {"TrainingJobArn": JOB_ARN},
        build_expected_training_job(CheckpointConfig={"S3Uri": _CHECKPOINTS}),
    )
    aws.sm.training_jobs = [writer_answer, build_description("InProgress")]
    keys_when_created = []
    create = aws.sm.create_training_job

    def create_and_record(**params):
        keys_when_created.append(_checkpoint_keys(aws))
        return create(**params)

    monkeypatch.setattr(aws.sm, "create_training_job", create_and_record)

    aws.tuner.fine_tune(build_request(checkpoint_uri=_CHECKPOINTS), overwrite=True)

    assert keys_when_created == [[]]
    assert (BUCKET, "nori/checkpoints/churn-old/child-0.pt") in aws.s3.objects
    assert aws.sm.calls[1] == ("describe_training_job", {"TrainingJobName": _EARLIER_JOB})


@pytest.mark.parametrize(
    "case, phrase",
    [
        ("foreign file", f"among them s3://{BUCKET}/{_CHECKPOINT_PREFIX}notes.txt; nothing was deleted"),
        ("running writer", f"training job {_EARLIER_JOB} is Stopping, so overwrite=True would delete"),
        ("cut-off listing", "holds more than 1,000 files"),
    ],
)
def test_fine_tune_with_overwrite_deletes_nothing_from_a_folder_it_cannot_empty_safely(aws, case, phrase):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    _fill_checkpoint_folder(aws, extra=("notes.txt",) if case == "foreign file" else ())
    if case == "cut-off listing":
        for k in range(1000):
            aws.s3.objects[(BUCKET, f"{_CHECKPOINT_PREFIX}child-{k + 10}.pt")] = b"weights"
    aws.sm.training_jobs = [
        build_description("Stopping" if case == "running writer" else "Stopped", TrainingJobName=_EARLIER_JOB)
    ]
    kept = _checkpoint_keys(aws)

    with pytest.raises(FineTuningError, match=re.escape(phrase)):
        aws.tuner.fine_tune(build_request(checkpoint_uri=_CHECKPOINTS), overwrite=True)

    assert _checkpoint_keys(aws) == kept
    assert "delete_objects" not in [operation for operation, _ in aws.s3.calls]
    assert "create_training_job" not in aws.sm.operations()


@pytest.mark.parametrize(
    "case, error, phrase",
    [
        ("no checkpoint_uri", ValueError, "overwrite=True deletes the files in the request's checkpoint_uri"),
        ("listing refused", FineTuningError, f"could not read the checkpoint folder {_CHECKPOINTS} (AccessDenied"),
        # A record S3 refuses is a failed read, not a record whose writer is unknown.
        ("record read refused", FineTuningError, f"could not read the checkpoint folder {_CHECKPOINTS} (AccessDenied"),
        ("delete refused", FineTuningError, f", the first s3://{BUCKET}/{_CHECKPOINT_PREFIX}child-0.json, failed (Acc"),
        ("key not deleted", FineTuningError, f"S3 refused to delete s3://{BUCKET}/{_NEWEST_RECORD_KEY} (AccessDenied"),
        ("job listing refused", FineTuningError, "could not list the training jobs that save to the checkpoint folder"),
    ],
)
def test_fine_tune_with_overwrite_creates_no_job_when_the_folder_cannot_be_read_or_emptied(aws, case, error, phrase):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    _fill_checkpoint_folder(aws)
    kept = _checkpoint_keys(aws)
    aws.sm.training_jobs = [build_description("Stopped", TrainingJobName=_EARLIER_JOB)]
    denied = AWSError("AccessDenied", "Access Denied")
    if case == "job listing refused":
        aws.sm.failures["list_training_jobs"] = AWSError("AccessDeniedException", "not authorized")
    if case == "listing refused":
        aws.s3.failures["list_objects_v2"] = denied
    if case == "record read refused":
        aws.s3.failures["get_object"] = denied
    if case == "delete refused":
        aws.s3.failures["delete_objects"] = denied
    if case == "key not deleted":
        aws.s3.delete_errors[_NEWEST_RECORD_KEY] = "AccessDenied"

    with pytest.raises(error, match=re.escape(phrase)) as raised:
        aws.tuner.fine_tune(
            build_request(checkpoint_uri=None if case == "no checkpoint_uri" else _CHECKPOINTS), overwrite=True
        )

    if error is FineTuningError:
        permission = {
            "listing refused": "s3:ListBucket",
            "record read refused": "s3:GetObject",
            "job listing refused": "sagemaker:ListTrainingJobs",
        }.get(case, "s3:DeleteObject")
        assert permission in str(raised.value) and "no job was created" in str(raised.value)
    if case != "key not deleted":
        assert _checkpoint_keys(aws) == kept
    assert "create_training_job" not in aws.sm.operations()


@pytest.mark.parametrize("case", ["folder listing", "delete", "key not deleted", "job listing"])
def test_fine_tune_names_a_permission_only_when_aws_denied_access(aws, case):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    _fill_checkpoint_folder(aws)
    aws.sm.training_jobs = [build_description("Stopped", TrainingJobName=_EARLIER_JOB)]
    busy = AWSError("SlowDown", "Please reduce your request rate.")
    if case == "folder listing":
        aws.s3.failures["list_objects_v2"] = busy
    if case == "delete":
        aws.s3.failures["delete_objects"] = busy
    if case == "key not deleted":
        aws.s3.delete_errors[_NEWEST_RECORD_KEY] = "SlowDown"
    if case == "job listing":
        aws.sm.failures["list_training_jobs"] = busy

    with pytest.raises(FineTuningError, match=re.escape("(SlowDown: ")) as raised:
        aws.tuner.fine_tune(build_request(checkpoint_uri=_CHECKPOINTS), overwrite=True)

    assert "let the identity" not in str(raised.value) and "no job was created" in str(raised.value)
    assert "create_training_job" not in aws.sm.operations()


@pytest.mark.parametrize("case", ["empty folder", "overwrite after a resume"])
def test_fine_tune_deletes_nothing_and_creates_no_job_while_a_job_that_saved_nothing_runs_on_the_folder(aws, case):
    """A running job that has saved no model yet is named by no record in the folder, only by SageMaker's listing."""
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    if case == "overwrite after a resume":
        _fill_checkpoint_folder(aws)
        aws.sm.training_jobs = [build_description("Stopped", TrainingJobName=_EARLIER_JOB)]
    # The first saves to a folder whose name starts the same; the second gave the folder without its last "/".
    aws.sm.running_training_jobs = [
        build_description(
            "InProgress", TrainingJobName="nori-elsewhere", CheckpointConfig={"S3Uri": _CHECKPOINTS[:-1] + "-old"}
        ),
        build_description("InProgress", TrainingJobName="nori-resumed", CheckpointConfig={"S3Uri": _CHECKPOINTS[:-1]}),
    ]
    kept = _checkpoint_keys(aws)

    with pytest.raises(FineTuningError) as raised:
        aws.tuner.fine_tune(build_request(checkpoint_uri=_CHECKPOINTS), overwrite=case == "overwrite after a resume")

    message = str(raised.value)
    assert "training job nori-resumed is InProgress and saves to checkpoint_uri" in message
    assert "nothing was deleted and no job was created" in message and "stop('nori-resumed')" in message
    assert raised.value.job_name == "nori-resumed"
    assert _checkpoint_keys(aws) == kept
    assert "delete_objects" not in [operation for operation, _ in aws.s3.calls]
    assert "create_training_job" not in aws.sm.operations()


@pytest.mark.parametrize("overwrite", [False, True])
def test_fine_tune_says_a_folder_holding_only_files_no_job_writes_is_not_a_checkpoint_folder(aws, overwrite):
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    for name in ("part-0.parquet", "part-1.parquet"):
        aws.s3.objects[(BUCKET, _CHECKPOINT_PREFIX + name)] = b"rows"

    with pytest.raises(FineTuningError) as raised:
        aws.tuner.fine_tune(build_request(checkpoint_uri=_CHECKPOINTS), overwrite=overwrite)

    message = str(raised.value)
    assert "is not a checkpoint folder: it holds 2 files, none of them one a fine-tuning job writes" in message
    assert f"the first s3://{BUCKET}/{_CHECKPOINT_PREFIX}part-0.parquet" in message and "resume(" not in message
    assert len(_checkpoint_keys(aws)) == 2
    assert "create_training_job" not in aws.sm.operations()


def _describe_created_job(params, status: str = "Stopped", **overrides):
    """DescribeTrainingJob's answer for a job created with ``params``."""
    description = {
        **copy.deepcopy(params),
        "TrainingJobArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:training-job/{params['TrainingJobName']}",
        "TrainingJobStatus": status,
        "SecondaryStatus": status,
        "CreationTime": NOW,
        "EnableManagedSpotTraining": False,
        "EnableInterContainerTrafficEncryption": False,
    }
    description.update(overrides)
    return description


@pytest.mark.parametrize(
    "version, trains_on_whole_table, sends_resume, overrides",
    [
        ("current", True, True, {"instance_type": "ml.g5.2xlarge", "max_training_hours": 2}),
        ("current", True, True, {}),
        ("implicit resume", True, False, {}),
        ("fixed steps", False, False, {"instance_type": "ml.g5.2xlarge"}),
    ],
)
def test_resume_creates_the_jobs_request_again_under_a_new_name_with_its_plan(
    aws, caplog, version, trains_on_whole_table, sends_resume, overrides
):
    hyperparameters = {
        "current": CURRENT_ALGORITHM_HYPERPARAMETERS,
        "implicit resume": IMPLICIT_RESUME_ALGORITHM_HYPERPARAMETERS,
        "fixed steps": LEGACY_ALGORITHM_HYPERPARAMETERS,
    }[version]
    aws.sm.algorithms = [build_algorithm_description(hyperparameters)]
    first = training_jobs_module.build_training_job_request(
        build_request(
            validation_data=f"s3://{BUCKET}/nori/validation/",
            hyperparameters=build_hyperparameters(
                feature_columns=["tenure", "plan"], bag=3, patience=None if trains_on_whole_table else 7
            ),
            volume_size_gb=50,
            kms_key_id="alias/nori",
            checkpoint_uri=_CHECKPOINTS,
        ),
        _EARLIER_JOB,
        trains_on_whole_table=trains_on_whole_table,
    )
    _fill_checkpoint_folder(aws)
    for channel in ("train", "validation"):
        _put_table(aws, f"nori/{channel}/part-0.parquet", [5000], columns=["churn", "tenure", "plan"])
    expected = copy.deepcopy(first)
    expected["TrainingJobName"] = JOB
    if sends_resume:
        expected["HyperParameters"]["resume"] = "true"
    if "instance_type" in overrides:
        expected["ResourceConfig"]["InstanceType"] = overrides["instance_type"]
    if "max_training_hours" in overrides:
        expected["HyperParameters"]["max_training_hours"] = "2.0"
        expected["StoppingCondition"]["MaxRuntimeInSeconds"] = 2 * 3600 + 1800
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, expected)
    aws.sm.training_jobs = [_describe_created_job(first), build_description("InProgress")]

    with caplog.at_level("INFO", logger="synthefy.fine_tune"):
        job = aws.tuner.resume(_EARLIER_JOB, **overrides)

    assert job.job_name == JOB
    # The plan of the request the new job runs, the override included, logged as the most the job takes.
    assert job.training_plan.instance_type == expected["ResourceConfig"]["InstanceType"]
    assert job.training_plan.bag == 3
    assert "training plan of the whole request, the most this job takes" in caplog.text
    assert job.training_plan.get_summary().splitlines()[0] in caplog.text


def _report_artifact(aws, **overrides) -> None:
    store_artifact(aws.s3, build_artifact(build_report(**overrides)))


@pytest.mark.parametrize(
    "case, phrase",
    [
        ("running", f"training job {_EARLIER_JOB} is InProgress, and a second job on its checkpoint folder"),
        ("no checkpoint folder", f"training job {_EARLIER_JOB} has no checkpoint_uri"),
        ("empty folder", "is empty, so there is nothing to continue; start a new job with fine_tune()"),
        ("running writer", "and training job nori-other is Stopping; wait for it to end"),
        ("completed whole", "its training report says no model was cut short, failed, or ended or left unstarted"),
        ("unknown version", "could not be described (An error occurred (AccessDeniedException)"),
        # The job an earlier resume() started has saved nothing yet: only SageMaker's listing names it.
        ("resumed already", "training job nori-resumed is InProgress and saves to the same checkpoint folder"),
        ("no checkpoint files", "holds no file a fine-tuning job writes, so there is nothing to continue"),
        (
            "settings a request lacks",
            "it was created with managed spot training (EnableManagedSpotTraining), a VPC (VpcConfig), a volume KMS "
            "key (ResourceConfig.VolumeKmsKeyId), which a job this client creates does not have",
        ),
        ("unreadable report", f"it completed, and its training report in {ARTIFACT} could not be read ("),
    ],
)
def test_resume_refuses_a_job_it_cannot_continue_naming_the_job_and_what_to_do(aws, case, phrase):
    first = training_jobs_module.build_training_job_request(build_request(checkpoint_uri=_CHECKPOINTS), _EARLIER_JOB)
    # The table is one the job trains on: no case is refused for it.
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    status = {"running": "InProgress", "completed whole": "Completed", "unreadable report": "Completed"}.get(
        case, "Stopped"
    )
    if case == "no checkpoint folder":
        del first["CheckpointConfig"]
    if case == "no checkpoint files":
        aws.s3.objects[(BUCKET, _CHECKPOINT_PREFIX + "part-0.parquet")] = b"rows"
    elif case != "empty folder":
        _fill_checkpoint_folder(aws, writer="nori-other" if case == "running writer" else _EARLIER_JOB)
    description = _describe_created_job(first, status)
    if case in ("completed whole", "unreadable report"):
        description["ModelArtifacts"] = {"S3ModelArtifacts": ARTIFACT}
    if case == "completed whole":
        _report_artifact(aws)
    if case == "settings a request lacks":
        description["EnableManagedSpotTraining"] = True
        description["VpcConfig"] = {"SecurityGroupIds": ["sg-0a1b2c3d"], "Subnets": ["subnet-0a1b2c3d"]}
        description["ResourceConfig"]["VolumeKmsKeyId"] = "alias/volume"
    if case == "resumed already":
        aws.sm.running_training_jobs = [
            build_description("InProgress", TrainingJobName="nori-resumed", CheckpointConfig={"S3Uri": _CHECKPOINTS})
        ]
    if case == "unknown version":
        denied = AWSError("AccessDeniedException", "not authorized to perform: sagemaker:DescribeAlgorithm")
        denied.args = ("An error occurred (AccessDeniedException) when calling DescribeAlgorithm",)
        aws.sm.algorithms = [denied]
    aws.sm.training_jobs = [description, build_description("Stopping", TrainingJobName="nori-other")]

    with pytest.raises(FineTuningError, match=re.escape(phrase)) as raised:
        aws.tuner.resume(_EARLIER_JOB)

    assert raised.value.job_name == _EARLIER_JOB and f"training job {_EARLIER_JOB}" in str(raised.value)
    assert "create_training_job" not in aws.sm.operations()


@pytest.mark.parametrize(
    "report",
    [
        {"cut_short": True},
        # The cap ends a model without cutting the job short: the job completes with a model left to train.
        {"child_curves": [build_curve(0), {**build_curve(1), "stop_reason": "max_training_hours"}]},
    ],
    ids=["cut-short", "ended-by-max-training-hours"],
)
def test_resume_continues_a_completed_job_whose_report_leaves_a_model_to_train(aws, report):
    first = training_jobs_module.build_training_job_request(build_request(checkpoint_uri=_CHECKPOINTS), _EARLIER_JOB)
    _fill_checkpoint_folder(aws)
    _put_table(aws, "nori/train/part-0.parquet", [5000])
    _report_artifact(aws, **report)
    expected = {**copy.deepcopy(first), "TrainingJobName": JOB}
    expected["HyperParameters"]["resume"] = "true"
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, expected)
    completed = _describe_created_job(first, "Completed", ModelArtifacts={"S3ModelArtifacts": ARTIFACT})
    aws.sm.training_jobs = [completed, build_description("InProgress")]

    aws.tuner.resume(_EARLIER_JOB)


def test_resume_without_credentials_for_the_tables_submits_the_job_without_a_plan(aws, monkeypatch, caplog):
    # The plan is advice in resume() too: a caller who cannot read the tables still continues the job.
    def get_plan(*args, **kwargs):
        raise NoCredentialsError()

    monkeypatch.setattr("synthefy.nori_fine_tune.get_training_plan_from_footers", get_plan)
    first = training_jobs_module.build_training_job_request(build_request(checkpoint_uri=_CHECKPOINTS), _EARLIER_JOB)
    _fill_checkpoint_folder(aws)
    expected = {**copy.deepcopy(first), "TrainingJobName": JOB}
    expected["HyperParameters"]["resume"] = "true"
    aws.stubber.add_response("create_training_job", {"TrainingJobArn": JOB_ARN}, expected)
    aws.sm.training_jobs = [_describe_created_job(first), build_description("InProgress")]

    with caplog.at_level("WARNING", logger="synthefy.fine_tune"):
        job = aws.tuner.resume(_EARLIER_JOB)

    assert job.job_name == JOB and job.training_plan is None
    assert "so the job is submitted without a plan: Unable to locate credentials" in caplog.text


def test_resume_refuses_a_table_the_job_would_refuse_before_creating_a_job(aws):
    # The training channel holds no Parquet file: the job would fail on it, after it was paid for.
    first = training_jobs_module.build_training_job_request(build_request(checkpoint_uri=_CHECKPOINTS), _EARLIER_JOB)
    _fill_checkpoint_folder(aws)
    aws.sm.training_jobs = [_describe_created_job(first)]

    with pytest.raises(ValueError, match=re.escape(f"training_data={TRAIN!r} holds no .parquet or .pq file")):
        aws.tuner.resume(_EARLIER_JOB)

    assert "create_training_job" not in aws.sm.operations()


@pytest.mark.parametrize(
    "argument, value", [("instance_type", "ml.g5.xlarge"), ("instance_type", ""), ("max_training_hours", 200)]
)
def test_resume_raises_value_error_naming_an_override_a_request_cannot_take(aws, argument, value):
    request = build_request(hyperparameters=build_hyperparameters(base_model="nori-100m"), checkpoint_uri=_CHECKPOINTS)
    first = training_jobs_module.build_training_job_request(request, _EARLIER_JOB)
    _fill_checkpoint_folder(aws)
    aws.sm.training_jobs = [_describe_created_job(first)]

    with pytest.raises(ValueError, match=re.escape(f"{argument}={value!r} is not one a request can take")) as raised:
        aws.tuner.resume(_EARLIER_JOB, **{argument: value})

    assert type(raised.value) is ValueError
    assert "create_training_job" not in aws.sm.operations()
    # Raised before the checkpoint folder is read.
    assert not aws.s3.calls


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


def test_wait_logs_each_status_change_once_with_sagemakers_message_then_the_reports_summary(aws, caplog):
    starting = [{"Status": "Starting", "StartTime": NOW, "StatusMessage": "Preparing the instances for training"}]
    training = starting + [{"Status": "Training", "StartTime": NOW, "StatusMessage": "Training in progress."}]
    aws.sm.training_jobs = [
        build_description("InProgress", SecondaryStatus="Starting", SecondaryStatusTransitions=starting),
        build_description("InProgress", SecondaryStatus="Starting", SecondaryStatusTransitions=starting),
        build_description("InProgress", SecondaryStatus="Training", SecondaryStatusTransitions=training),
        build_completed(SecondaryStatusTransitions=training),
    ]
    store_artifact(aws.s3, build_artifact(build_report()))
    with caplog.at_level("INFO", logger="synthefy.fine_tune"):
        aws.tuner.wait(JOB, poll_s=10)
    messages = [record.getMessage() for record in caplog.records if record.name == "synthefy.fine_tune"]
    assert messages == [
        f"training job {JOB}: InProgress / Starting: Preparing the instances for training",
        f"training job {JOB}: InProgress / Training: Training in progress.",
        f"training job {JOB}: Completed / Completed: Training in progress.",
        f"training job {JOB}:\n{TrainingReport.model_validate(build_report()).get_summary()}",
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
