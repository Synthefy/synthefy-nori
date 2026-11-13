"""The client's data-model mirrors, tested on their own.

`src/synthefy/nori_data_models.py` is a deliberate copy of
`synthefy_nori.inference.memory_policy.MemoryPolicy`, so the client can offer a typed policy
object without depending on the model package (which would pull torch into a thin API client).

The comparison against the authoritative model does not live here: it runs in CI, where both
packages are installed. What remains here is the behavior the mirror must have on its own:
frozen, forbid-extra on the policy, allow-extra on the report, bounds enforced, documented
defaults.

The fine-tuning models describe the Nori fine-tuning Algorithm, and CI checks them against it.
Here they are held to what the client relies on: hyperparameters survive the all-strings form
SageMaker delivers them in, a fine-tuned model never takes a released name, and a report parses
whatever validation mode a newer Algorithm version names, with or without the counts of the rows
training visited and of the workers and GPUs that trained the children, and the children that failed.

The fine-tuning request and result schemas at the end of the module mirror nothing; their tests
check what a request refuses, or derives, before a job is paid for.
"""

from __future__ import annotations

import json
import re
import warnings

import pytest
from pydantic import BaseModel, ValidationError

from synthefy._training_jobs import refuse_task_settings_the_version_lacks
from synthefy.errors import FineTuningError
from synthefy.nori_data_models import (
    FINE_TUNED_MODEL_NAME_PATTERN,
    MEMORY_PRESETS,
    MEMORY_RUNGS,
    RELEASED_MODEL_NAMES,
    FineTuneHyperparameters,
    FineTuneJob,
    FineTuneJobRequest,
    get_task_tables,
    serialize_hyperparameters,
    FineTunedEndpoint,
    TaskTable,
    MemoryAttempt,
    MemoryPolicy,
    MemoryReport,
    TrainingReport,
    serialize_hyperparameters,
)

from ._fine_tune_fakes import MODEL, build_hyperparameters, build_request


# ------------------------------------------------- the mirror stands on its own
def test_the_policy_rejects_unknown_fields_like_the_server_does():
    with pytest.raises(ValueError):
        MemoryPolicy(int8=True)


def test_the_policy_is_frozen_so_a_shared_instance_cannot_be_mutated():
    policy = MemoryPolicy(cache_dtype="int8")
    with pytest.raises(ValueError):
        policy.cache_dtype = "bf16"


def test_bounds_are_enforced_locally():
    with pytest.raises(ValueError):
        MemoryPolicy(gpu_budget_frac=1.5)  # a fraction of VRAM cannot exceed 1
    with pytest.raises(ValueError):
        MemoryPolicy(gpu_budget_frac=0)  # exclusive minimum: 0 would mean "never cache"
    assert MemoryPolicy(gpu_budget_absolute_gb=0).gpu_budget_absolute_gb == 0


def test_the_report_keeps_fields_this_version_does_not_know():
    """extra="allow" on the RESPONSE model: a newer server may report more than we know.

    Dropping an unrecognised field would be worse than carrying it — the report is diagnostic,
    and silently shortening it hides exactly what someone is debugging.
    """
    report = MemoryReport(rung="resident_int8", something_new=17)
    assert report.rung == "resident_int8"
    assert report.model_dump()["something_new"] == 17


def test_streaming_report_keeps_the_resolved_policy_details():
    report = MemoryReport(
        rung="stream_bf16",
        stream_context=True,
        context_row_chunk=2048,
        reuse_context_cache=False,
    )
    dumped = report.model_dump()
    assert dumped["rung"] == "stream_bf16"
    assert dumped["stream_context"] is True
    assert dumped["context_row_chunk"] == 2048
    assert dumped["reuse_context_cache"] is False


def test_attempt_history_keeps_newer_server_fields():
    report = MemoryReport(
        attempt_history=[
            {
                "pipeline_ids": [0],
                "path": "cached",
                "rung": "resident_bf16",
                "cache_dtype": "bf16",
                "offload_to_host": False,
                "context_row_chunk": None,
                "query_chunk": 128,
                "outcome": "success",
                "reason": "resolved",
                "dropped_context_rows": 0,
                "new_attempt_detail": "kept",
            }
        ]
    )
    assert report.model_dump()["attempt_history"][0]["new_attempt_detail"] == "kept"
    assert report.model_dump()["attempt_history"][0]["query_chunk"] == 128


def test_defaults_match_the_documented_behaviour():
    policy = MemoryPolicy()
    assert policy.cache is True and policy.reuse_context_cache is True
    assert policy.cache_dtype == "bf16"
    assert policy.allow_quantization is True and policy.offload_to_host is True
    assert policy.gpu_budget_frac == 0.4 and policy.host_budget_frac == 0.25
    assert policy.stream_context is False
    assert policy.allow_subsample is True


def test_stream_context_remains_a_partial_wire_policy():
    policy = MemoryPolicy(stream_context=True)
    assert policy.model_dump(exclude_unset=True) == {"stream_context": True}
    assert {"stream_bf16", "stream_int8"} <= set(MEMORY_RUNGS)


# ------------------------------------------------- the fine-tuning job's models
def test_hyperparameters_survive_the_all_strings_form_sagemaker_delivers():
    full = FineTuneHyperparameters(
        target_column="price",
        model_name="ft-acme-churn-v1",
        base_model="nori-100m",
        feature_columns=["sqft", "rooms", "age"],
        bag=3,
        row_partition_strategy="folds",
        n_steps_per_model=40,
        lr=3e-6,
        patience=7,
        max_training_hours=1.5,
        validation_every_n_steps=4,
        seed=11,
    )
    # patience and max_training_hours default to a value, so None travels as "none" rather than being left out.
    uncapped = FineTuneHyperparameters(
        target_column="y", model_name="ft-m", n_passes_per_row=2.5, patience=None, max_training_hours=None
    )
    for hyperparameters in (FineTuneHyperparameters(target_column="y", model_name="ft-m"), full, uncapped):
        strings = serialize_hyperparameters(hyperparameters)
        assert all(isinstance(value, str) for value in strings.values())
        assert FineTuneHyperparameters.model_validate(strings) == hyperparameters


def test_feature_columns_with_a_comma_in_a_name_travel_as_a_json_list():
    columns = ["price, usd", "rooms"]
    parsed = FineTuneHyperparameters.model_validate(
        {"target_column": "y", "model_name": "ft-m", "feature_columns": json.dumps(columns)}
    )
    assert parsed.feature_columns == columns


def test_unknown_hyperparameters_are_refused_before_a_job_is_paid_for():
    with pytest.raises(ValidationError, match="epochs"):
        FineTuneHyperparameters(target_column="y", model_name="ft-m", epochs=3)


@pytest.mark.parametrize("name", RELEASED_MODEL_NAMES)
def test_a_fine_tuned_model_may_not_take_a_released_name(name):
    with pytest.raises(ValidationError, match="released model name"):
        FineTuneHyperparameters(target_column="y", model_name=name)


@pytest.mark.parametrize(
    "name, accepted",
    [
        ("ft-acme-churn-v1", True),
        ("ft-9", True),
        ("ft-" + "a" * 60, True),
        ("ft-" + "a" * 61, False),
        ("acme-churn-v1", False),
        ("ft-", False),
        ("ft--churn", False),
        ("ft-Acme", False),
        ("ft-acme_churn", False),
        ("synthefy/ft-custom", False),
        ("", False),
    ],
    ids=[
        "hyphenated",
        "one-digit",
        "63-chars",
        "64-chars",
        "no-prefix",
        "prefix-only",
        "leading-hyphen-after-prefix",
        "uppercase",
        "underscore",
        "slash",
        "empty",
    ],
)
def test_a_fine_tuned_name_starts_with_ft_and_fits_63_characters(name, accepted):
    assert bool(re.fullmatch(FINE_TUNED_MODEL_NAME_PATTERN, name)) is accepted
    if accepted:
        assert FineTuneHyperparameters(target_column="y", model_name=name).model_name == name
    else:
        with pytest.raises(ValidationError, match="lowercase letters"):
            FineTuneHyperparameters(target_column="y", model_name=name)


def _report_fields(**fields):
    """A ``training_report.json`` body with only the fields every Algorithm version writes, plus ``fields``."""
    curve = {
        "child": 0,
        "steps": [1, 2],
        "train_loss": [0.9, None],
        "val_mse": [0.5, 0.4],
        "best_step": 2,
        "val_mse_best": 0.4,
        "interrupted": False,
        "joined_average": True,
    }
    return {
        "model_name": "ft-acme-churn-v1",
        "model_version": "sha256:" + "a" * 64,
        "base_model": "nori-6m",
        "base_model_version": "sha256:" + "b" * 64,
        "n_rows": 5000,
        "n_rows_used": 4500,
        "n_features": 4,
        "children": 1,
        "steps_run": [2],
        "validation_source": "kfold",
        "n_val_rows": 500,
        "val_mse_before": 0.6,
        "val_mse_after": None,
        "child_curves": [curve],
        "cut_short": True,
        "seconds": 12.5,
        "instance_type": "ml.g5.xlarge",
        **fields,
    }


def _plan_fields(**fields):
    """A ``training_plan`` of a bag-2 shard job on 9,000 training rows."""
    return {
        "n_rows_in_table": 10_000,
        "n_feature_columns": 4,
        "n_validation_rows": 1_000,
        "n_training_rows": 9_000,
        "bag": 2,
        "row_partition_strategy": "shard",
        "n_rows_per_model": 4_500,
        "rows_per_step": 1_500.0,
        "targets_per_step": 375.0,
        "rows_per_step_is_estimate": False,
        "steps_per_model": 100,
        "n_passes_per_row": 33.33,
        "validation_every_n_steps": 1,
        "fraction_of_table_read": 1.0,
        "fraction_of_rows_as_targets": 1.0,
        "seconds_per_step": 2.1,
        "instance_type": "ml.g5.12xlarge",
        "gpus": 4,
        "training_hours": 0.06,
        "estimated_cost_usd": 0.41,
        "prices_as_of": "2026-10-27",
        "patience": 25,
        "patience_steps": 25,
        "max_training_hours": 24.0,
        "hits_max_training_hours": False,
        "warnings": [],
        **fields,
    }


def test_a_report_from_a_job_with_a_newer_validation_mode_still_parses():
    report = TrainingReport.model_validate(_report_fields(validation_source="a-mode-added-later"))
    assert report.validation_source == "a-mode-added-later"
    assert report.val_mse_after is None
    assert report.child_curves[0].train_loss == [0.9, None]


def test_a_report_parses_with_or_without_the_fields_that_count_rows_workers_and_failed_children():
    without = TrainingReport.model_validate(_report_fields())
    assert (
        without.validation_interval,
        without.read_seconds,
        without.rows_visited,
        without.fraction_visited,
        without.steps_per_pass,
        without.workers,
        without.gpus,
        without.failed_children,
        without.training_plan,
        without.stop_reason,
        without.unstarted_children,
        without.bag,
        without.n_passes_per_row_completed,
        without.fraction_of_rows_as_targets,
        without.n_rows_unread,
    ) == (1, 0.0, 0, 0.0, 0, 1, 1, [], None, None, [], None, None, None, None)
    assert "n/a passes per row; n/a of the training rows served as targets; n/a rows unread" in without.get_summary()
    curve = without.child_curves[0]
    assert (
        curve.rows_visited,
        curve.failed,
        curve.failure,
        curve.restored,
        curve.stop_reason,
        curve.n_skipped_state_saves,
        curve.last_saved_step,
    ) == (0, False, "", False, None, 0, None)

    # A bag of two on a four-GPU type: one worker per child on two of the GPUs, and only child 0 joined the average.
    written = _report_fields(
        children=1,
        steps_run=[2, 0],
        instance_type="ml.g5.12xlarge",
        validation_interval=2,
        read_seconds=1.25,
        rows_visited=9000,
        fraction_visited=2.5,
        steps_per_pass=4,
        workers=2,
        gpus=2,
        failed_children=[1],
        training_plan=_plan_fields(),
        stop_reason="max_training_hours",
        n_passes_per_row_completed=1.5,
        seconds_per_step=2.1,
        fraction_of_rows_as_targets=0.62,
        unstarted_children=[],
        n_rows_unread=0,
        bag=2,
        row_partition_strategy="shard",
        tasks=[],
    )
    written["child_curves"][0].update(
        rows_visited=9000,
        failed=False,
        failure="",
        restored=False,
        stop_reason="max_training_hours",
        n_skipped_state_saves=1,
        last_saved_step=2,
    )
    written["child_curves"].append(
        {
            "child": 1,
            "steps": [],
            "train_loss": [],
            "val_mse": [],
            "best_step": 0,
            "val_mse_best": None,
            "interrupted": False,
            "joined_average": False,
            "rows_visited": 0,
            "failed": True,
            "failure": "RuntimeError: CUDA error: an illegal memory access was encountered",
            "restored": False,
            "stop_reason": "plan_done",
            "n_skipped_state_saves": 0,
            "last_saved_step": None,
        }
    )
    assert TrainingReport.model_validate_json(json.dumps(written)).model_dump() == written


# ------------------------------------------------- the fine-tuning request and result schemas
def test_request_derives_max_runtime_from_max_training_hours_plus_margin_within_sagemakers_limit():
    assert build_request().max_runtime_s == 88_200
    assert build_request(hyperparameters=build_hyperparameters(max_training_hours=0.25)).max_runtime_s == 2_700
    assert build_request(hyperparameters=build_hyperparameters(max_training_hours=1.1)).max_runtime_s == 5_760
    assert build_request(hyperparameters=build_hyperparameters(max_training_hours=119.5)).max_runtime_s == 432_000
    assert build_request(hyperparameters=build_hyperparameters(max_training_hours=None)).max_runtime_s == 432_000
    assert build_request(max_runtime_s=600).max_runtime_s == 600
    with pytest.raises(ValidationError, match="max_training_hours"):
        build_hyperparameters(max_training_hours=float("inf"))
    with pytest.raises(ValidationError, match="less than or equal to 119.5"):
        build_hyperparameters(max_training_hours=119.6)


def test_hyperparameters_refuse_a_step_count_with_a_pass_count():
    with pytest.raises(ValidationError, match="n_steps_per_model and n_passes_per_row both set the number of steps"):
        build_hyperparameters(n_steps_per_model=30, n_passes_per_row=2.0)
    with pytest.raises(ValidationError, match="n_steps_per_model and n_passes_per_row both set the number of steps"):
        build_hyperparameters(n_steps_per_model=30, n_passes_per_row=1.0)
    assert build_hyperparameters(n_steps_per_model=30).n_passes_per_row is None


def test_request_refuses_an_unknown_keyword():
    with pytest.raises(ValidationError, match="instance_count"):
        build_request(instance_count=2)


@pytest.mark.parametrize(
    "base_model, instance_type",
    [("nori-6m", "ml.g5.xlarge"), ("nori-1-5", "ml.g5.xlarge"), ("nori-100m", "ml.g6e.xlarge")],
)
def test_request_defaults_the_instance_type_to_one_the_base_fits(base_model, instance_type):
    assert build_request(hyperparameters=build_hyperparameters(base_model=base_model)).instance_type == instance_type


@pytest.mark.parametrize("instance_type", ["ml.g6e.xlarge", "ml.g6e.2xlarge", "ml.g6e.12xlarge", "ml.g6e.48xlarge"])
def test_request_accepts_nori_100m_on_every_48_gb_type_with_one_gpu_or_several(instance_type):
    request = build_request(hyperparameters=build_hyperparameters(base_model="nori-100m"), instance_type=instance_type)
    assert request.instance_type == instance_type


@pytest.mark.parametrize("instance_type", ["ml.g5.2xlarge", "ml.g5.12xlarge", "ml.g5.48xlarge"])
def test_request_refuses_nori_100m_on_a_24_gb_type_with_one_gpu_or_several_naming_the_48_gb_types(instance_type):
    with pytest.raises(ValidationError) as raised:
        build_request(hyperparameters=build_hyperparameters(base_model="nori-100m"), instance_type=instance_type)
    assert (
        f"needs a 48 GB GPU, which {instance_type!r} does not have; use instance_type 'ml.g6e.xlarge' or "
        "'ml.g6e.2xlarge' or 'ml.g6e.12xlarge' or 'ml.g6e.48xlarge'"
    ) in str(raised.value)


def test_hyperparameters_refuse_a_released_model_the_algorithm_does_not_fine_tune():
    with pytest.raises(ValidationError, match=re.escape("Input should be 'nori-6m', 'nori-1-5' or 'nori-100m'")):
        build_hyperparameters(base_model="nori-30m")


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("training_data", "https://customer-bucket/train/", "training_data must be an s3://bucket/prefix URI"),
        ("training_data", "s3:///train/", "training_data must be an s3://bucket/prefix URI"),
        ("training_data", "s3://customer-bucket/train", "training_data must be an S3 folder ending in '/'"),
        ("training_data", "s3://customer-bucket/train/table.csv.gz", "the job reads only .parquet and .pq files"),
        ("validation_data", "s3://customer-bucket/TABLE.CSV", "validation_data names a CSV object"),
        ("training_data", "s3://customer-bucket/train/_table.parquet", "The job skips such files"),
        ("validation_data", "s3://customer-bucket/validation", "validation_data must be an S3 folder ending in '/'"),
        ("output_path", "customer-bucket/output/", "output_path must be an s3://bucket/prefix URI"),
    ],
)
def test_request_refuses_an_s3_location_the_job_cannot_read_naming_the_field(field, value, reason):
    with pytest.raises(ValidationError, match=re.escape(reason)):
        build_request(**{field: value})


@pytest.mark.parametrize(
    "uri",
    [
        "s3://customer-bucket/train/",
        "s3://customer-bucket/train/table.parquet",
        "s3://customer-bucket/TABLE.PQ",
    ],
)
def test_request_accepts_a_folder_or_one_parquet_object_for_either_channel(uri):
    request = build_request(training_data=uri, validation_data=uri)
    assert (request.training_data, request.validation_data) == (uri, uri)


def test_request_refuses_a_hyperparameter_value_longer_than_sagemaker_allows_naming_field_length_and_remedy():
    columns = [f"feature_{index:04d}" for index in range(250)]
    with pytest.raises(ValidationError) as raised:
        build_request(hyperparameters=build_hyperparameters(feature_columns=columns))
    message = str(raised.value)
    assert "feature_columns is 4,000 characters as sent to SageMaker, which allows at most 2,500" in message
    assert "name fewer columns, or leave feature_columns unset" in message


def test_request_refuses_feature_columns_that_include_the_target_before_the_job_runs():
    with pytest.raises(ValidationError, match="feature_columns includes the target column 'y'; drop it, or leave"):
        build_request(hyperparameters=build_hyperparameters(target_column="y", feature_columns=["a", "y"]))
    assert build_request(hyperparameters=build_hyperparameters(target_column="y", feature_columns=["a", "b"]))


def test_request_accepts_a_value_of_exactly_2500_characters_and_refuses_one_more():
    assert build_request(hyperparameters=build_hyperparameters(target_column="y" * 2500))
    with pytest.raises(ValidationError, match="target_column is 2,501 characters .* shorten it"):
        build_request(hyperparameters=build_hyperparameters(target_column="y" * 2501))


@pytest.mark.parametrize("job_name", ["-leading", "trailing-", "under_score", "a" * 64])
def test_request_refuses_a_job_name_sagemaker_would_refuse(job_name):
    with pytest.raises(ValidationError, match="job_name must be 1 to 63 letters, digits and '-'"):
        build_request(job_name=job_name)


def test_request_keeps_a_snapshot_of_the_hyperparameters_it_was_built_with():
    hyperparameters = build_hyperparameters()
    request = build_request(hyperparameters=hyperparameters)
    hyperparameters.base_model = "nori-100m"
    assert request.hyperparameters is not hyperparameters
    assert request.hyperparameters.base_model == "nori-1-5"


@pytest.mark.parametrize(
    "model", [FineTuneJobRequest, FineTuneJob, FineTunedEndpoint, FineTuneHyperparameters, TrainingReport]
)
def test_a_client_model_defines_without_a_protected_namespace_warning(model):
    """Pydantic before 2.10 warns about any ``model_*`` field unless the config clears protected_namespaces; the
    probe is built from the model's own fields and config under warnings-as-errors, in the model's own module so
    its string annotations resolve."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        type(
            f"{model.__name__}Probe",
            (BaseModel,),
            {
                "__annotations__": dict(model.__annotations__),
                "model_config": model.model_config,
                "__module__": model.__module__,
            },
        )


def test_serialized_hyperparameters_send_every_field_with_the_columns_as_a_json_list():
    assert serialize_hyperparameters(build_hyperparameters(feature_columns=["tenure", "plan"])) == {
        "target_column": "churn",
        "model_name": MODEL,
        "base_model": "nori-1-5",
        "task": "regression",
        "feature_columns": '["tenure", "plan"]',
        "bag": "8",
        "row_partition_strategy": "shard",
        "lr": "1e-05",
        "patience": "25",
        "max_training_hours": "24.0",
        "seed": "0",
    }


@pytest.mark.parametrize("columns", [["price, usd", "rooms"], [" padded", "rooms"], ["[odd]", "rooms"]])
def test_serialized_hyperparameters_keep_a_column_name_whatever_it_holds(columns):
    text = serialize_hyperparameters(build_hyperparameters(feature_columns=columns))["feature_columns"]
    assert json.loads(text) == columns
    assert build_hyperparameters(feature_columns=text).feature_columns == columns


def test_task_tables_and_row_counts_reach_the_job_as_the_client_set_them():
    hyperparameters = FineTuneHyperparameters(
        model_name="ft-acme-v1",
        query_rows=24,
        tasks=[
            TaskTable(path="sales.parquet", target_column="sales", context_rows={256: 1, 1024: 2}),
            TaskTable(path="stores/", target_column="units", task_column="store_id", time_column="day"),
        ],
    )

    # SageMaker carries strings, and JSON object keys are text: the job parses the rows back as integers.
    sent = serialize_hyperparameters(hyperparameters)
    received = FineTuneHyperparameters.model_validate(sent)

    assert json.loads(sent["tasks"])[0] == {
        "path": "sales.parquet",
        "target_column": "sales",
        "context_rows": {"256": 1.0, "1024": 2.0},
    }
    assert received == hyperparameters
    # Each table takes the job's query rows where it sets none.
    assert [table.query_rows for table in get_task_tables(received)] == [24, 24]
    assert get_task_tables(received)[0].context_rows == {256: 1.0, 1024: 2.0}


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({}, "set one of target_column"),
        (
            {
                "target_column": "y",
                "tasks": [{"path": "a/", "target_column": "y"}, {"path": "b/", "target_column": "y"}],
            },
            "set one of target_column",
        ),
        ({"tasks": [{"path": "a/", "target_column": "y"}]}, "is one task: set target_column"),
        (
            {"tasks": [{"path": "a/", "target_column": "y"}, {"path": "a", "target_column": "z"}]},
            "lists the path 'a' more than once",
        ),
        (
            {
                "task_column": "store",
                "tasks": [{"path": "a/", "target_column": "y"}, {"path": "b/", "target_column": "y"}],
            },
            "set task_column on each task table",
        ),
        ({"target_column": "y", "time_column": "day"}, "time_column needs query_rows"),
        ({"target_column": "y", "task_column": "y"}, "task_column must be neither target_column"),
        ({"target_column": "y", "context_rows": 8}, "context_rows must be at least 16 rows"),
        ({"target_column": "y", "query_rows": {"24": -1}}, "weights must be finite and not negative"),
    ],
    ids=[
        "neither",
        "both",
        "one-plain-table",
        "repeated-path",
        "top-level-task-column",
        "time-without-queries",
        "task-is-target",
        "short-context",
        "negative-weight",
    ],
)
def test_task_settings_that_cannot_describe_tasks_are_refused(settings, message):
    with pytest.raises(ValidationError, match=message):
        FineTuneHyperparameters(model_name="ft-acme-v1", **settings)


def test_a_job_of_tasks_is_refused_on_an_algorithm_version_that_predates_it():
    hyperparameters = FineTuneHyperparameters(model_name="ft-acme-v1", target_column="y", task_column="store")
    supported_before_tasks = frozenset({"target_column", "model_name", "n_passes_per_row"})

    with pytest.raises(FineTuningError, match="does not read task_column: its version predates jobs of tasks"):
        refuse_task_settings_the_version_lacks(hyperparameters, "nori-fine-tune", supported_before_tasks)
    refuse_task_settings_the_version_lacks(hyperparameters, "nori-fine-tune", supported_before_tasks | {"task_column"})
