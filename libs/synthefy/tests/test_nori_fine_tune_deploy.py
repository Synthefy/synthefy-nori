"""Deploying a fine-tuned model and removing it (``SynthefyNoriFineTuner.deploy``, ``undeploy`` and
``create_client``, with ``synthefy._sagemaker_deploy`` doing each step). No test reaches AWS; the fakes are in
``_fine_tune_fakes``."""

import re
from typing import Any, Dict, Sequence, Tuple

import pytest
from synthefy import nori_client as nori_client_module
from synthefy.errors import APITimeoutError, FineTuningError
from synthefy.nori_data_models import FineTunedEndpoint, serialize_hyperparameters

from ._fine_tune_fakes import (
    ACCOUNT,
    ARTIFACT,
    ARTIFACT_KEY,
    ASYNC_ROOT,
    BUCKET,
    CONFIG,
    JOB,
    MODEL,
    PACKAGE_ARN,
    REGION,
    ROLE,
    RESOURCE_NAME_TOKEN,
    AWSError,
    FakeSageMaker,
    FakeS3,
    build_artifact,
    build_completed,
    build_description,
    build_endpoint,
    build_hyperparameters,
    build_missing_package_error,
    build_missing_resource_error,
    build_package,
    build_report,
    expect_create_endpoint,
    expect_create_endpoint_config,
    expect_create_model,
    expect_create_model_package,
    expect_every_deploy_create,
    install_fake_aws,
    store_artifact,
)


@pytest.fixture
def aws(monkeypatch):
    with install_fake_aws(monkeypatch) as fakes:
        yield fakes


@pytest.fixture
def finished(aws):
    """A Completed job whose artifact holds a report."""
    aws.sm.training_jobs = [build_completed()]
    store_artifact(aws.s3, build_artifact(build_report()))
    return aws


def _build_missing_config_error() -> Dict[str, Any]:
    """CreateEndpoint's refusal while its config is not visible to it yet, as AWS documents it."""
    return {
        "service_error_code": "ValidationException",
        "service_message": f'Could not find endpoint configuration "arn:aws:sagemaker:{REGION}:{ACCOUNT}:{CONFIG}".',
        "expected_params": {"EndpointName": MODEL, "EndpointConfigName": CONFIG},
    }


# --------------------------------------------------------------------------- #
# deploy
# --------------------------------------------------------------------------- #


def test_deploy_makes_the_package_model_config_and_endpoint_in_order_and_waits_for_in_service(finished):
    aws = finished
    aws.sm.model_packages = [build_missing_package_error(), build_package("InProgress"), build_package("Completed")]
    aws.sm.endpoints = [
        build_missing_resource_error("endpoint"),
        build_endpoint("Creating"),
        build_endpoint("InService"),
    ]
    expect_every_deploy_create(aws.stubber)

    endpoint = aws.tuner.deploy(JOB)

    assert aws.sm.operations() == [
        "describe_training_job",
        "describe_endpoint",
        "describe_model_package",
        "create_model_package",
        "describe_model_package",
        "describe_model_package",
        "describe_model",
        "create_model",
        "create_endpoint_config",
        "describe_endpoint_config",
        "create_endpoint",
        "describe_endpoint",
        "describe_endpoint",
    ]
    assert endpoint == FineTunedEndpoint(
        endpoint_name=MODEL,
        model_name=MODEL,
        instance_type="ml.g5.xlarge",
        status="InService",
        model_package_arn=PACKAGE_ARN,
        sagemaker_model_name=MODEL,
        endpoint_config_name=CONFIG,
        async_output_uri=ASYNC_ROOT,
        cut_short=False,
        region_name=REGION,
    )
    assert aws.s3.calls == [("head_object", ARTIFACT_KEY)]
    assert aws.clock.sleeps == [5.0, 30.0]


def test_deploy_of_a_stopped_job_with_an_artifact_warns_that_it_was_cut_short(aws):
    aws.sm.training_jobs = [build_description("Stopped", ModelArtifacts={"S3ModelArtifacts": ARTIFACT})]
    store_artifact(aws.s3, build_artifact(build_report(cut_short=True)))
    aws.sm.model_packages = [build_missing_package_error(), build_package("Completed")]
    aws.sm.endpoints = [build_missing_resource_error("endpoint"), build_endpoint("InService")]
    expect_every_deploy_create(aws.stubber)

    with pytest.warns(UserWarning, match="stopped before its children finished"):
        endpoint = aws.tuner.deploy(JOB)

    assert endpoint.cut_short is True
    assert aws.s3.calls == [("head_object", ARTIFACT_KEY), ("get_object", ARTIFACT_KEY)]


def test_deploy_refuses_a_stopped_job_without_an_artifact_as_nothing_to_ship(aws):
    aws.sm.training_jobs = [build_description("Stopped")]
    with pytest.raises(FineTuningError, match="without a model artifact, so there is nothing to ship") as raised:
        aws.tuner.deploy(JOB)
    assert raised.value.status == "Stopped"
    assert aws.sm.operations() == ["describe_training_job"]


def test_deploy_refuses_a_job_that_failed_on_capacity_and_says_to_start_it_again(aws):
    reason = "CapacityError: Unable to provision requested ML compute capacity."
    aws.sm.training_jobs = [build_description("Failed", FailureReason=reason)]
    with pytest.raises(FineTuningError, match="start it again with fine_tune") as raised:
        aws.tuner.deploy(JOB)
    assert (raised.value.status, raised.value.failure_reason) == ("Failed", reason)
    assert aws.sm.operations() == ["describe_training_job"]


@pytest.mark.parametrize("status", ["InProgress", "Stopping"])
def test_deploy_refuses_a_job_still_running(aws, status):
    aws.sm.training_jobs = [build_description(status)]
    with pytest.raises(FineTuningError, match=re.escape(f"is still {status}; call wait('{JOB}') first")):
        aws.tuner.deploy(JOB)


def test_deploy_refuses_an_artifact_gone_from_s3_before_making_anything(aws):
    aws.sm.training_jobs = [build_completed()]
    with pytest.raises(FineTuningError, match="is gone from S3, so there is nothing to deploy"):
        aws.tuner.deploy(JOB)
    assert aws.sm.operations() == ["describe_training_job"]


def test_deploy_refuses_a_model_name_that_is_not_a_valid_endpoint_name_before_making_anything(finished):
    aws = finished
    trailing_hyphen = serialize_hyperparameters(build_hyperparameters(model_name="ft-acme-"))
    aws.sm.training_jobs = [build_completed(HyperParameters=trailing_hyphen)]
    with pytest.raises(ValueError, match="endpoint_name must be 1 to 63 letters"):
        aws.tuner.deploy(JOB)
    assert aws.sm.operations() == ["describe_training_job"]


def test_deploy_refuses_an_existing_endpoint_before_making_anything(finished):
    aws = finished
    aws.sm.endpoints = [build_endpoint("InService")]
    with pytest.raises(ValueError, match=re.escape(f"endpoint {MODEL} already exists (InService); call undeploy")):
        aws.tuner.deploy(JOB)
    assert aws.sm.operations() == ["describe_training_job", "describe_endpoint"]


def test_deploy_lets_a_validation_exception_that_is_not_about_a_missing_resource_through(finished):
    aws = finished
    refusal = AWSError("ValidationException", "Value 'x' at 'endpointName' failed to satisfy constraint")
    aws.sm.endpoints = [refusal]
    with pytest.raises(AWSError) as raised:
        aws.tuner.deploy(JOB)
    assert raised.value is refusal


def test_deploy_refuses_an_instance_type_the_package_does_not_serve_before_making_a_model(finished):
    aws = finished
    aws.sm.model_packages = [build_missing_package_error(), build_package("Completed")]
    expect_create_model_package(aws.stubber)
    with pytest.raises(ValueError, match="choose one of ml.g5.xlarge, ml.g5.2xlarge, ml.g6.xlarge"):
        aws.tuner.deploy(JOB, instance_type="ml.p4d.24xlarge")
    assert "describe_model" not in aws.sm.operations()


def test_deploy_again_after_a_failure_reuses_the_package_and_model_of_this_artifact_and_role(finished):
    aws = finished
    aws.sm.model_packages = [build_package("Completed")]
    aws.sm.models = [
        {"ModelName": MODEL, "PrimaryContainer": {"ModelPackageName": PACKAGE_ARN}, "ExecutionRoleArn": ROLE}
    ]
    aws.sm.endpoints = [build_missing_resource_error("endpoint"), build_endpoint("InService")]
    expect_create_endpoint_config(aws.stubber)
    expect_create_endpoint(aws.stubber)

    endpoint = aws.tuner.deploy(JOB)

    assert "create_model_package" not in aws.sm.operations() and "create_model" not in aws.sm.operations()
    assert (endpoint.model_package_arn, endpoint.status) == (PACKAGE_ARN, "InService")


def test_deploy_again_with_another_role_refuses_the_model_a_failed_deploy_left_under_the_old_role(finished):
    aws = finished
    new_role = f"arn:aws:iam::{ACCOUNT}:role/NoriWithOutputAccess"
    aws.sm.model_packages = [build_package("Completed")]
    aws.sm.models = [
        {"ModelName": MODEL, "PrimaryContainer": {"ModelPackageName": PACKAGE_ARN}, "ExecutionRoleArn": ROLE}
    ]

    with pytest.raises(ValueError) as raised:
        aws.tuner.deploy(JOB, role_arn=new_role)

    message = str(raised.value)
    assert f"with execution role {ROLE}, not {new_role}" in message
    assert f"sagemaker.delete_model(ModelName='{MODEL}')" in message
    assert "or pass another endpoint_name" in message
    assert "create_endpoint_config" not in aws.sm.operations()


def test_deploy_refuses_a_package_of_the_same_name_over_another_artifact(finished):
    aws = finished
    aws.sm.model_packages = [build_package("Completed", artifact=f"s3://{BUCKET}/elsewhere/model.tar.gz")]
    with pytest.raises(ValueError, match=re.escape(f"sagemaker.delete_model_package(ModelPackageName='{JOB}')")):
        aws.tuner.deploy(JOB)


def test_deploy_refuses_a_model_of_the_endpoints_name_on_another_package(finished):
    aws = finished
    aws.sm.model_packages = [build_package("Completed")]
    other = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package/another-job"
    aws.sm.models = [{"ModelName": MODEL, "PrimaryContainer": {"ModelPackageName": other}, "ExecutionRoleArn": ROLE}]
    with pytest.raises(ValueError, match=re.escape(f"sagemaker.delete_model(ModelName='{MODEL}')")):
        aws.tuner.deploy(JOB)
    assert "create_endpoint_config" not in aws.sm.operations()


def test_deploy_refuses_a_package_that_failed_validation_naming_the_reason(finished):
    aws = finished
    reason = "ModelDataUrl was not produced by a training job of this algorithm"
    aws.sm.model_packages = [build_missing_package_error(), build_package("Failed", reasons=[reason])]
    expect_create_model_package(aws.stubber)
    with pytest.raises(FineTuningError, match="Failed SageMaker's validation") as raised:
        aws.tuner.deploy(JOB)
    assert (raised.value.failure_reason, raised.value.created) == (reason, {"model_package": PACKAGE_ARN})


def test_deploy_times_out_on_a_package_still_validating_and_says_a_retry_reuses_it(finished):
    aws = finished
    aws.sm.model_packages = [build_missing_package_error(), build_package("InProgress")]
    expect_create_model_package(aws.stubber)
    with pytest.raises(APITimeoutError, match="still InProgress after 900 s .* call deploy\\(\\) again to reuse it"):
        aws.tuner.deploy(JOB)
    assert "describe_model" not in aws.sm.operations()


def test_deploy_waits_for_the_new_config_and_retries_create_endpoint_while_sagemaker_cannot_see_it(finished):
    aws = finished
    aws.sm.model_packages = [build_missing_package_error(), build_package("Completed")]
    aws.sm.endpoints = [build_missing_resource_error("endpoint"), build_endpoint("InService")]
    aws.sm.hidden_endpoint_config_reads = 1
    expect_create_model_package(aws.stubber)
    expect_create_model(aws.stubber)
    expect_create_endpoint_config(aws.stubber)
    aws.stubber.add_client_error("create_endpoint", **_build_missing_config_error())
    expect_create_endpoint(aws.stubber)

    endpoint = aws.tuner.deploy(JOB)

    assert endpoint.status == "InService"
    operations = aws.sm.operations()
    start = operations.index("create_endpoint_config")
    assert operations[start : start + 5] == [
        "create_endpoint_config",
        "describe_endpoint_config",
        "describe_endpoint_config",
        "create_endpoint",
        "create_endpoint",
    ]
    assert aws.clock.sleeps == [5.0, 5.0]


def test_deploy_lets_a_create_endpoint_validation_error_about_anything_else_through_without_a_retry(finished):
    aws = finished
    aws.sm.model_packages = [build_missing_package_error(), build_package("Completed")]
    expect_create_model_package(aws.stubber)
    expect_create_model(aws.stubber)
    expect_create_endpoint_config(aws.stubber)
    aws.stubber.add_client_error(
        "create_endpoint",
        service_error_code="ValidationException",
        service_message="Value 'ml.g5.xlarge' is not supported for asynchronous inference in this account",
        expected_params={"EndpointName": MODEL, "EndpointConfigName": CONFIG},
    )

    with pytest.raises(FineTuningError) as raised:
        aws.tuner.deploy(JOB)

    assert "is not supported for asynchronous inference" in str(raised.value.__cause__)
    assert raised.value.created == {"model_package": PACKAGE_ARN, "model": MODEL, "endpoint_config": CONFIG}
    assert aws.sm.operations().count("create_endpoint") == 1


def test_deploy_names_other_instance_types_and_undeploy_when_the_endpoint_has_no_capacity(finished):
    aws = finished
    aws.sm.model_packages = [build_missing_package_error(), build_package("Completed")]
    reason = "InsufficientInstanceCapacity: We currently do not have sufficient capacity for ml.g5.xlarge."
    aws.sm.endpoints = [
        build_missing_resource_error("endpoint"),
        build_endpoint("Creating"),
        build_endpoint("Failed", reason),
    ]
    expect_every_deploy_create(aws.stubber)

    with pytest.raises(FineTuningError) as raised:
        aws.tuner.deploy(JOB)

    message = str(raised.value)
    assert f"call undeploy('{MODEL}'), then deploy() again later, or with instance_type one of" in message
    assert "ml.g5.2xlarge, ml.g6.xlarge" in message
    assert (raised.value.status, raised.value.failure_reason) == ("Failed", reason)
    assert raised.value.created == {
        "model_package": PACKAGE_ARN,
        "model": MODEL,
        "endpoint_config": CONFIG,
        "endpoint": MODEL,
    }


def test_deploy_endpoint_poll_timeout_names_undeploy(finished):
    aws = finished
    aws.sm.model_packages = [build_missing_package_error(), build_package("Completed")]
    aws.sm.endpoints = [build_missing_resource_error("endpoint"), build_endpoint("Creating")]
    expect_every_deploy_create(aws.stubber)
    with pytest.raises(APITimeoutError, match="still Creating after 60 s.*undeploy"):
        aws.tuner.deploy(JOB, timeout_s=60)
    assert aws.clock.sleeps == [30.0, 30.0]


def test_deploy_without_wait_returns_the_endpoint_creating(finished):
    aws = finished
    aws.sm.model_packages = [build_missing_package_error(), build_package("Completed")]
    expect_every_deploy_create(aws.stubber)
    endpoint = aws.tuner.deploy(JOB, wait=False)
    assert endpoint.status == "Creating"
    assert aws.sm.operations()[-1] == "create_endpoint"


def test_a_failed_deploy_names_what_it_made_and_the_calls_that_delete_it(finished):
    aws = finished
    aws.sm.model_packages = [build_missing_package_error(), build_package("Completed")]
    expect_create_model_package(aws.stubber)
    expect_create_model(aws.stubber)
    expect_create_endpoint_config(aws.stubber)
    aws.stubber.add_client_error(
        "create_endpoint",
        service_error_code="AccessDeniedException",
        service_message="not authorized to perform: sagemaker:CreateEndpoint",
        expected_params={"EndpointName": MODEL, "EndpointConfigName": CONFIG},
    )
    with pytest.raises(FineTuningError) as raised:
        aws.tuner.deploy(JOB)
    assert raised.value.created == {"model_package": PACKAGE_ARN, "model": MODEL, "endpoint_config": CONFIG}
    message = str(raised.value)
    assert f"sagemaker.delete_endpoint_config(EndpointConfigName='{CONFIG}')" in message
    assert f"sagemaker.delete_model(ModelName='{MODEL}')" in message
    assert "AccessDeniedException" in message


# --------------------------------------------------------------------------- #
# undeploy
# --------------------------------------------------------------------------- #


def test_undeploy_deletes_the_endpoint_waits_for_it_then_its_configs_then_the_model_and_keeps_the_package(aws):
    aws.sm.endpoints = [build_endpoint("Deleting"), build_missing_resource_error("endpoint")]
    aws.sm.endpoint_config_names = [
        CONFIG,
        f"{MODEL}-20260930-080000",  # a config synthefy 7.2.0 made, without the token
        f"{MODEL}-2-20261001-120000-{RESOURCE_NAME_TOKEN}",
        f"{MODEL}-manual",
    ]

    aws.tuner.undeploy(MODEL)

    assert aws.sm.calls == [
        ("delete_endpoint", {"EndpointName": MODEL}),
        ("describe_endpoint", {"EndpointName": MODEL}),
        ("describe_endpoint", {"EndpointName": MODEL}),
        ("list_endpoint_configs", {"NameContains": f"{MODEL}-"}),
        ("delete_endpoint_config", {"EndpointConfigName": CONFIG}),
        ("delete_endpoint_config", {"EndpointConfigName": f"{MODEL}-20260930-080000"}),
        ("delete_model", {"ModelName": MODEL}),
    ]
    assert aws.clock.sleeps == [10.0]


def test_undeploy_of_a_handle_deletes_in_the_handles_region_not_the_fine_tuners(aws):
    regional = aws.regional.setdefault("eu-west-1", FakeSageMaker(aws.sm._stubbed))
    regional.failures["delete_endpoint"] = build_missing_resource_error("endpoint")
    endpoint = FineTunedEndpoint(
        endpoint_name="churn-endpoint",
        model_name=MODEL,
        instance_type="ml.g5.xlarge",
        status="InService",
        model_package_arn=PACKAGE_ARN,
        sagemaker_model_name="churn-endpoint",
        endpoint_config_name=CONFIG,
        async_output_uri=ASYNC_ROOT,
        cut_short=False,
        region_name="eu-west-1",
    )

    aws.tuner.undeploy(endpoint)

    assert aws.sm.calls == []
    assert regional.operations() == ["delete_endpoint", "list_endpoint_configs", "delete_model"]
    assert regional.closed
    assert aws.client_calls[-1] == {
        "region_name": "eu-west-1",
        "timeout": aws.client_calls[0]["timeout"],
        "max_retries": aws.client_calls[0]["max_retries"],
        "user_agent_extra": aws.client_calls[0]["user_agent_extra"],
    }


def test_undeploy_skips_what_is_already_gone(aws):
    aws.sm.failures["delete_endpoint"] = build_missing_resource_error("endpoint")
    aws.sm.failures["delete_model"] = build_missing_resource_error("model")
    aws.tuner.undeploy(MODEL)
    assert aws.sm.operations() == ["delete_endpoint", "list_endpoint_configs", "delete_model"]


@pytest.mark.parametrize("length", [40, 52], ids=["exactly-40", "longer"])
def test_undeploy_of_a_name_cut_to_40_characters_deletes_only_the_configs_serving_its_own_model(aws, length):
    shared = "a" * 40
    mine, theirs = (shared + "-mine")[:length], shared + "-theirs"
    my_config, their_config = (
        f"{shared}-20261001-120000-{RESOURCE_NAME_TOKEN}",
        f"{shared}-20261001-130000-{RESOURCE_NAME_TOKEN}",
    )
    my_legacy_config = f"{mine[:47]}-20261001-110000"  # synthefy 7.2.0 cut the name at 47 and added no token
    aws.sm.endpoint_config_names = [my_config, their_config, my_legacy_config]
    aws.sm.endpoint_config_models = {my_config: mine, their_config: theirs, my_legacy_config: mine}
    aws.sm.failures["delete_endpoint"] = build_missing_resource_error("endpoint", mine)

    aws.tuner.undeploy(mine)

    listed = [params for operation, params in aws.sm.calls if operation == "list_endpoint_configs"]
    assert listed == [{"NameContains": shared}]
    deleted = [params for operation, params in aws.sm.calls if operation == "delete_endpoint_config"]
    assert deleted == [{"EndpointConfigName": my_config}, {"EndpointConfigName": my_legacy_config}]


# --------------------------------------------------------------------------- #
# create_client
# --------------------------------------------------------------------------- #


def test_create_client_predicts_through_the_endpoint_with_the_fine_tuned_name_in_its_region(aws, monkeypatch):
    captured: Dict[str, Any] = {}

    def create_clients(_service_names: Sequence[str], **kwargs: Any) -> Tuple[Any, Any]:
        captured.update(kwargs)
        return FakeS3(), FakeS3()

    monkeypatch.setattr(nori_client_module, "create_aws_clients", create_clients)
    endpoint = FineTunedEndpoint(
        endpoint_name="churn-endpoint",
        model_name=MODEL,
        instance_type="ml.g5.xlarge",
        status="InService",
        model_package_arn=PACKAGE_ARN,
        sagemaker_model_name="churn-endpoint",
        endpoint_config_name=CONFIG,
        async_output_uri=ASYNC_ROOT,
        cut_short=False,
        region_name="eu-west-1",
    )
    with aws.tuner.create_client(endpoint, s3_staging_uri=f"s3://{BUCKET}/nori/staging/", timeout=42.0) as client:
        assert (client.mode, client.endpoint_name, client.region_name) == ("sagemaker", "churn-endpoint", "eu-west-1")
        assert client._sagemaker_model == MODEL
    assert (captured["region_name"], captured["timeout"]) == ("eu-west-1", 42.0)
