"""The AWS helpers the inference client and the fine-tuner share (``synthefy._aws``); no test reaches AWS."""

import pytest
from synthefy._aws import build_aws_client_config, is_missing_resource, list_s3_keys, parse_s3_uri

from ._fine_tune_fakes import AWSError, FakeS3, build_missing_package_error, build_missing_resource_error


def test_s3_uri_refusals_name_the_argument_the_uri_came_from():
    with pytest.raises(ValueError, match="^training_data must be an s3://bucket/prefix URI"):
        parse_s3_uri("https://customer-bucket/train.csv", name="training_data")
    with pytest.raises(ValueError, match="^output_path must include a customer-owned prefix"):
        parse_s3_uri("s3://customer-bucket", name="output_path", require_prefix=True)
    with pytest.raises(ValueError, match="^s3_staging_uri must be an s3://bucket/prefix URI"):
        parse_s3_uri("customer-bucket/nori", name="s3_staging_uri")
    assert parse_s3_uri("s3://customer-bucket/data/train.csv", name="training_data") == (
        "customer-bucket",
        "data/train.csv",
    )


def test_build_aws_client_config_sets_timeouts_standard_retries_and_the_user_agent():
    config = build_aws_client_config(12.5, 4, "synthefy-python")

    assert (config.connect_timeout, config.read_timeout) == (12.5, 12.5)
    assert config.retries == {"mode": "standard", "total_max_attempts": 5}
    assert config.user_agent_extra == "synthefy-python"


@pytest.mark.parametrize(
    "error, is_missing",
    [
        (build_missing_resource_error("endpoint"), True),
        (build_missing_resource_error("model"), True),
        (build_missing_package_error(), True),
        (AWSError("ValidationException", "Endpoint config acme could not be found."), True),
        (AWSError("ResourceNotFound", "Requested resource not found."), True),
        (AWSError("ValidationException", "Value 'x' at 'endpointName' failed to satisfy constraint"), False),
        (AWSError("AccessDeniedException", "Could not find a policy that allows this call"), False),
        (RuntimeError("could not find endpoint"), False),
    ],
    ids=[
        "endpoint",
        "model",
        "model-package",
        "endpoint-config",
        "training-job",
        "other-validation",
        "access-denied",
        "not-an-aws-error",
    ],
)
def test_only_an_error_saying_the_resource_is_missing_counts_as_missing(error, is_missing):
    assert is_missing_resource(error) is is_missing


def test_list_s3_keys_returns_the_first_keys_under_the_prefix_and_whether_more_exist():
    s3 = FakeS3()
    for key in ("data/train.csv", "data/train.csv.backup.csv", "data/train.csv.old", "data/validation.csv"):
        s3.objects[("customer-bucket", key)] = b""

    assert list_s3_keys(s3, bucket="customer-bucket", prefix="data/train.csv", max_keys=2) == (
        ["data/train.csv", "data/train.csv.backup.csv"],
        True,
    )
    assert list_s3_keys(s3, bucket="customer-bucket", prefix="data/train.csv", max_keys=10) == (
        ["data/train.csv", "data/train.csv.backup.csv", "data/train.csv.old"],
        False,
    )
    assert list_s3_keys(s3, bucket="customer-bucket", prefix="data/none", max_keys=10) == ([], False)
