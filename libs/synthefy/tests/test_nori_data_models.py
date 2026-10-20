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
whatever validation mode a newer Algorithm version names.

The fine-tuning request and result schemas at the end of the module mirror nothing; their tests
check what a request refuses, or derives, before a job is paid for.
"""

from __future__ import annotations

import json
import re
import warnings

import pytest
from pydantic import BaseModel, ValidationError

from synthefy.nori_data_models import (
    FINE_TUNED_MODEL_NAME_PATTERN,
    MEMORY_PRESETS,
    MEMORY_RUNGS,
    RELEASED_MODEL_NAMES,
    FineTuneHyperparameters,
    FineTuneJob,
    FineTuneJobRequest,
    FineTunedEndpoint,
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
        steps=40,
        lr=3e-6,
        patience=7,
        budget_s=120.5,
        seed=11,
    )
    for hyperparameters in (FineTuneHyperparameters(target_column="y", model_name="ft-m"), full):
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


def test_a_report_from_a_job_with_a_newer_validation_mode_still_parses():
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
    report = TrainingReport.model_validate(
        {
            "model_name": "ft-acme-churn-v1",
            "model_version": "sha256:" + "a" * 64,
            "base_model": "nori-6m",
            "base_model_version": "sha256:" + "b" * 64,
            "n_rows": 5000,
            "n_rows_used": 5000,
            "n_features": 4,
            "children": 1,
            "steps_run": [2],
            "validation_source": "a-mode-added-later",
            "n_val_rows": 500,
            "val_mse_before": 0.6,
            "val_mse_after": None,
            "child_curves": [curve],
            "cut_short": True,
            "seconds": 12.5,
            "instance_type": "ml.g5.xlarge",
        }
    )
    assert report.validation_source == "a-mode-added-later"
    assert report.val_mse_after is None
    assert report.child_curves[0].train_loss == [0.9, None]


# ------------------------------------------------- the fine-tuning request and result schemas
def test_request_derives_max_runtime_from_bag_and_budget_plus_margin():
    assert build_request().max_runtime_s == 12_600
    assert build_request(hyperparameters=build_hyperparameters(bag=2, budget_s=100.0)).max_runtime_s == 2_100


def test_request_refuses_max_runtime_one_second_below_bag_times_budget_and_accepts_it_at_the_floor():
    hyperparameters = build_hyperparameters(bag=2, budget_s=100.0)
    with pytest.raises(ValidationError) as raised:
        build_request(hyperparameters=hyperparameters, max_runtime_s=199)
    message = str(raised.value)
    assert "max_runtime_s=199 is below bag x budget_s=200, so SageMaker would stop the job" in message
    assert "Lower budget_s or bag, or raise max_runtime_s. The job also needs time besides training" in message
    assert build_request(hyperparameters=hyperparameters, max_runtime_s=200).max_runtime_s == 200


def test_request_refuses_an_unknown_keyword():
    with pytest.raises(ValidationError, match="instance_count"):
        build_request(instance_count=2)


@pytest.mark.parametrize("base_model, instance_type", [("nori-6m", "ml.g5.xlarge"), ("nori-100m", "ml.g6e.xlarge")])
def test_request_defaults_the_instance_type_to_one_the_base_fits(base_model, instance_type):
    assert build_request(hyperparameters=build_hyperparameters(base_model=base_model)).instance_type == instance_type


def test_request_refuses_nori_100m_on_a_24_gb_instance_naming_the_48_gb_types():
    with pytest.raises(ValidationError, match=re.escape("use instance_type 'ml.g6e.xlarge' or 'ml.g6e.2xlarge'")):
        build_request(hyperparameters=build_hyperparameters(base_model="nori-100m"), instance_type="ml.g5.2xlarge")


def test_hyperparameters_refuse_a_released_model_the_algorithm_does_not_fine_tune():
    with pytest.raises(ValidationError, match=re.escape("Input should be 'nori-6m' or 'nori-100m'")):
        build_hyperparameters(base_model="nori-30m")


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("training_data", "https://customer-bucket/train/", "training_data must be an s3://bucket/prefix URI"),
        ("training_data", "s3:///train/", "training_data must be an s3://bucket/prefix URI"),
        ("training_data", "s3://customer-bucket/train", "training_data must be an S3 folder ending in '/'"),
        ("training_data", "s3://customer-bucket/train/table.parquet", "the job reads only .csv files"),
        ("validation_data", "s3://customer-bucket/validation", "validation_data must be an S3 folder ending in '/'"),
        ("output_path", "customer-bucket/output/", "output_path must be an s3://bucket/prefix URI"),
    ],
)
def test_request_refuses_an_s3_location_the_job_cannot_read_naming_the_field(field, value, reason):
    with pytest.raises(ValidationError, match=re.escape(reason)):
        build_request(**{field: value})


@pytest.mark.parametrize(
    "uri", ["s3://customer-bucket/train/", "s3://customer-bucket/train/table.csv", "s3://customer-bucket/TABLE.CSV"]
)
def test_request_accepts_a_folder_or_one_csv_object_for_either_channel(uri):
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
    assert request.hyperparameters.base_model == "nori-6m"


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
        "base_model": "nori-6m",
        "task": "regression",
        "feature_columns": '["tenure", "plan"]',
        "bag": "8",
        "steps": "100",
        "lr": "1e-05",
        "patience": "25",
        "budget_s": "900.0",
        "seed": "0",
    }


@pytest.mark.parametrize("columns", [["price, usd", "rooms"], [" padded", "rooms"], ["[odd]", "rooms"]])
def test_serialized_hyperparameters_keep_a_column_name_whatever_it_holds(columns):
    text = serialize_hyperparameters(build_hyperparameters(feature_columns=columns))["feature_columns"]
    assert json.loads(text) == columns
    assert build_hyperparameters(feature_columns=text).feature_columns == columns
