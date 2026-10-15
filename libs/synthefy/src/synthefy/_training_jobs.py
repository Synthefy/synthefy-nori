"""The SageMaker training job API in the fine-tuner's terms, and the checks a job needs before it is created.

A :class:`~synthefy.FineTuneJobRequest` becomes CreateTrainingJob's parameters here, and a DescribeTrainingJob
response becomes a :class:`~synthefy.FineTuneJob`. The checks that run just before CreateTrainingJob live here too:
the request's own checks again, the Algorithm's region, and a one-object ``.csv`` channel that would read more than
one object. The one check that calls AWS takes the S3 client it is given, so this module imports no AWS SDK.
"""

from datetime import datetime
from typing import Any, Dict

from pydantic import ValidationError
from synthefy._aws import (
    MAX_RESOURCE_NAME_CHARS,
    RESOURCE_NAME_TIMESTAMP_CHARS,
    RESOURCE_NAME_TIMESTAMP_FORMAT,
    list_s3_keys,
    parse_s3_uri,
)
from synthefy.nori_data_models import (
    TRAIN_CHANNEL,
    VALIDATION_CHANNEL,
    FineTuneJob,
    FineTuneJobRequest,
    serialize_hyperparameters,
)

#: A default job name is nori-<model_name>-<UTC timestamp>, the model name cut so the whole fits SageMaker's
#: 63 characters. The model name brings its ft- prefix, so a job reads nori-ft-<name>-<timestamp>.
_JOB_NAME_PREFIX = "nori-"
_JOB_NAME_MODEL_CHARS = MAX_RESOURCE_NAME_CHARS - len(_JOB_NAME_PREFIX) - 1 - RESOURCE_NAME_TIMESTAMP_CHARS
#: How many look-alike keys a refused one-object channel names.
_MAX_LISTED_SIBLINGS = 10


def build_job_name(model_name: str, now: datetime) -> str:
    """nori-<model_name>-<UTC timestamp>, cut to SageMaker's 63 characters: unique per second."""
    return f"{_JOB_NAME_PREFIX}{model_name[:_JOB_NAME_MODEL_CHARS]}-{now:{RESOURCE_NAME_TIMESTAMP_FORMAT}}"


def build_training_job_request(request: FineTuneJobRequest, job_name: str) -> Dict[str, Any]:
    """CreateTrainingJob's parameters for one request; network isolation always, as the Algorithm requires."""
    channels = [(TRAIN_CHANNEL, request.training_data)]
    if request.validation_data is not None:
        channels.append((VALIDATION_CHANNEL, request.validation_data))
    output: Dict[str, str] = {"S3OutputPath": request.output_path}
    if request.kms_key_id is not None:
        output["KmsKeyId"] = request.kms_key_id
    return {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {"AlgorithmName": request.algorithm, "TrainingInputMode": "File"},
        "RoleArn": request.role_arn,
        "HyperParameters": serialize_hyperparameters(request.hyperparameters),
        "InputDataConfig": [
            {
                "ChannelName": name,
                "ContentType": "text/csv",
                "InputMode": "File",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
            for name, uri in channels
        ],
        "OutputDataConfig": output,
        "ResourceConfig": {
            "InstanceType": request.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": request.volume_size_gb,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": request.max_runtime_s},
        "EnableNetworkIsolation": True,
    }


def parse_job_description(description: Dict[str, Any]) -> FineTuneJob:
    """A FineTuneJob from one DescribeTrainingJob response."""
    transitions = description.get("SecondaryStatusTransitions") or []
    specification = description.get("AlgorithmSpecification") or {}
    return FineTuneJob(
        job_name=description["TrainingJobName"],
        job_arn=description["TrainingJobArn"],
        status=description["TrainingJobStatus"],
        secondary_status=description.get("SecondaryStatus"),
        status_message=transitions[-1].get("StatusMessage") if transitions else None,
        created_at=description["CreationTime"],
        ended_at=description.get("TrainingEndTime"),
        artifact_uri=(description.get("ModelArtifacts") or {}).get("S3ModelArtifacts"),
        failure_reason=description.get("FailureReason"),
        latest_metrics={
            metric["MetricName"]: metric["Value"] for metric in description.get("FinalMetricDataList") or []
        },
        algorithm=specification.get("AlgorithmName") or specification.get("TrainingImage") or "",
        model_name=(description.get("HyperParameters") or {}).get("model_name"),
        billable_seconds=description.get("BillableTimeInSeconds"),
    )


def revalidate_request(request: FineTuneJobRequest) -> FineTuneJobRequest:
    """The request rebuilt from its own fields, so a value changed inside it after it was built is checked again.

    The request is frozen, but its ``hyperparameters`` is a plain model; a base or budget changed there would
    otherwise reach SageMaker with the instance type and runtime derived for the old one.
    """
    try:
        return FineTuneJobRequest.model_validate(request.model_dump())
    except ValidationError as exc:
        raise ValueError(
            "the request no longer passes its own checks, so its hyperparameters were changed after it was built; "
            f"build a new FineTuneJobRequest. {exc}"
        ) from exc


def refuse_foreign_region(algorithm: str, region_name: str) -> None:
    """Refuse an Algorithm ARN from another region before SageMaker does, naming both regions."""
    if not algorithm.startswith("arn:"):
        return
    parts = algorithm.split(":")
    region = parts[3] if len(parts) > 3 else ""
    if region and region != region_name:
        raise ValueError(
            f"the Algorithm {algorithm} is in {region}, but this fine-tuner calls SageMaker in {region_name}; "
            f"SageMaker runs an Algorithm only in its own region, so build SynthefyNoriFineTuner("
            f"region_name={region!r}) and keep the job's buckets in that region too"
        )


def refuse_csv_siblings(s3: Any, uri: str, argument: str) -> None:
    """Refuse a one-object ``.csv`` channel whose key other keys start with.

    SageMaker's S3Prefix channel reads every key that starts with the given text, so ``data/train.csv`` also
    reads ``data/train.csv.backup.csv`` and the job would train on both tables concatenated.
    """
    bucket, key = parse_s3_uri(uri, name=argument)
    keys, truncated = list_s3_keys(s3, bucket=bucket, prefix=key, max_keys=_MAX_LISTED_SIBLINGS + 1)
    siblings = [name for name in keys if name != key]
    if not siblings:
        return
    shown = ", ".join(f"s3://{bucket}/{name}" for name in siblings[:_MAX_LISTED_SIBLINGS])
    more = " and more" if truncated or len(siblings) > _MAX_LISTED_SIBLINGS else ""
    raise ValueError(
        f"{argument}={uri!r} names one .csv object, but SageMaker reads every key that starts with it, so the job "
        f"would also read {shown}{more}. Rename or move those objects, or put the table in a folder of its own "
        "and pass the folder, ending in '/'"
    )
