"""The SageMaker resources that serve a fine-tuned model: made in order, reused on a retry, deleted on undeploy.

A deployment is a model package over the training job's artifact, a SageMaker model on that package, an endpoint
config and an asynchronous endpoint. Each step is a function taking the boto3 SageMaker (or S3) client;
create_deployment() and delete_deployment() run them in order, so :class:`~synthefy.SynthefyNoriFineTuner` only
checks the job and names the endpoint. ``created`` maps what a deployment made or reused so far to its name, so a
failure can say what exists and how to delete it.
"""

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from synthefy._aws import (
    FINE_TUNING_LOGGER_NAME,
    MAX_RESOURCE_NAME_CHARS,
    RESOURCE_NAME_TIMESTAMP_CHARS,
    RESOURCE_NAME_TIMESTAMP_FORMAT,
    format_seconds,
    is_missing_resource,
    is_missing_s3_object,
    parse_s3_uri,
    read_aws_error,
)
from synthefy.errors import APITimeoutError, FineTuningError
from synthefy.nori_data_models import FineTuneJob

_LOG = logging.getLogger(FINE_TUNING_LOGGER_NAME)

#: The endpoint instance when deploy() names none: one small fine-tuned model fits the smallest supported GPU.
DEFAULT_ENDPOINT_INSTANCE_TYPE = "ml.g5.xlarge"
#: deploy()'s default wait for InService: when GPU capacity is scarce an endpoint can take over half an hour to
#: reach InService or to fail on capacity.
ENDPOINT_TIMEOUT_S = 3600.0
#: The timestamp a generated endpoint config name ends with, as listing matches it.
_TIMESTAMP_PATTERN = re.compile(r"[0-9]{8}-[0-9]{6}")
#: A job in one of these still holds its instance and has not written its artifact, so deploy() refuses it.
_RUNNING_JOB_STATUSES = frozenset({"InProgress", "Stopping"})
#: The statuses deploy() ships from: a Completed job, or a Stopped one that wrote an artifact.
_DEPLOYABLE_JOB_STATUSES = frozenset({"Completed", "Stopped"})
#: An endpoint config name is <endpoint_name>-<UTC timestamp>, the endpoint name cut so the whole fits SageMaker's
#: 63 characters; each deploy() gets its own.
_ENDPOINT_CONFIG_NAME_CHARS = MAX_RESOURCE_NAME_CHARS - 1 - RESOURCE_NAME_TIMESTAMP_CHARS
#: The model package poll: SageMaker validates a package from a training job in minutes.
_PACKAGE_POLL_S = 5.0
_PACKAGE_TIMEOUT_S = 900.0
#: CreateEndpoint looks its config up through an eventually consistent read, so a config created a moment ago can
#: read as missing; AWS documents this and recommends retrying. Six tries five seconds apart outlast that delay.
_CONFIG_VISIBILITY_ATTEMPTS = 6
_CONFIG_VISIBILITY_POLL_S = 5.0
#: The endpoint poll while it pulls the image and loads the model.
_ENDPOINT_POLL_S = 30.0
#: undeploy()'s wait for a deleted endpoint to disappear, after which its config and model can be deleted.
_ENDPOINT_DELETE_POLL_S = 10.0
_ENDPOINT_DELETE_TIMEOUT_S = 600.0
_VARIANT_NAME = "AllTraffic"
#: One queued request at a time per instance: the container runs one prediction at a time, so more would only
#: raise peak memory on the same GPU. Scale with instance count instead.
_MAX_CONCURRENT_INVOCATIONS_PER_INSTANCE = 1
#: What a failed deploy() made, and the boto3 call that deletes each, newest first.
_CLEANUP_CALLS = (
    ("endpoint", "delete_endpoint", "EndpointName"),
    ("endpoint_config", "delete_endpoint_config", "EndpointConfigName"),
    ("model", "delete_model", "ModelName"),
    ("model_package", "delete_model_package", "ModelPackageName"),
)


def build_endpoint_config_name(endpoint_name: str, now: datetime) -> str:
    """<endpoint_name[:47]>-<UTC timestamp>: within SageMaker's 63 characters, and one per deploy()."""
    return f"{endpoint_name[:_ENDPOINT_CONFIG_NAME_CHARS]}-{now:{RESOURCE_NAME_TIMESTAMP_FORMAT}}"


def format_cleanup_calls(created: Dict[str, str]) -> str:
    """The boto3 delete calls for what a failed deploy() made, newest first, ready to paste."""
    return "; ".join(
        f"sagemaker.{call}({parameter}={created[kind]!r})"
        for kind, call, parameter in _CLEANUP_CALLS
        if kind in created
    )


def format_failed_deploy_message(problem: str, created: Dict[str, str], endpoint_name: str) -> str:
    """A failed deploy()'s message: the problem, then what exists now and how to remove it."""
    if not created:
        return problem
    made = ", ".join(f"{kind} {name}" for kind, name in created.items())
    return (
        f"{problem} deploy() had made or reused: {made}. undeploy({endpoint_name!r}) deletes the endpoint, its "
        "endpoint configs and the SageMaker model and keeps the model package, which the next deploy() of this job "
        f"reuses; to delete everything: {format_cleanup_calls(created)}"
    )


def find_deployable_artifact(s3: Any, job: FineTuneJob) -> str:
    """The artifact of a Completed or Stopped job, confirmed in S3; every other state is refused with why."""
    if job.status in _RUNNING_JOB_STATUSES:
        raise FineTuningError(
            f"training job {job.job_name} is still {job.status}; call wait({job.job_name!r}) first, then deploy "
            "the finished job",
            job_name=job.job_name,
            status=job.status,
        )
    if job.status == "Failed":
        reason = job.failure_reason or "no reason given"
        hint = ""
        if reason.startswith("CapacityError"):
            hint = (
                ". SageMaker could not get an instance for the job: start it again with fine_tune() later, or on "
                "another instance_type the Algorithm supports"
            )
        raise FineTuningError(
            f"training job {job.job_name} Failed, so there is nothing to deploy: {reason}{hint}",
            job_name=job.job_name,
            status=job.status,
            failure_reason=job.failure_reason,
        )
    if job.status not in _DEPLOYABLE_JOB_STATUSES:
        raise FineTuningError(
            f"training job {job.job_name} is {job.status}; deploy() ships a Completed or Stopped job",
            job_name=job.job_name,
            status=job.status,
            failure_reason=job.failure_reason,
        )
    if job.artifact_uri is None:
        why = ": it was stopped before it had produced a model" if job.status == "Stopped" else ""
        raise FineTuningError(
            f"training job {job.job_name} ended {job.status} without a model artifact, so there is nothing to "
            f"ship{why}",
            job_name=job.job_name,
            status=job.status,
            failure_reason=job.failure_reason,
        )
    bucket, key = parse_s3_uri(job.artifact_uri, name="ModelArtifacts.S3ModelArtifacts")
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if not is_missing_s3_object(exc):
            raise
        raise FineTuningError(
            f"the model artifact {job.artifact_uri} of training job {job.job_name} is gone from S3, so there is "
            "nothing to deploy",
            job_name=job.job_name,
            status=job.status,
        ) from exc
    return job.artifact_uri


def refuse_existing_endpoint(sagemaker: Any, endpoint_name: str) -> None:
    """Refuse an endpoint name already in use, before anything is made."""
    try:
        described = sagemaker.describe_endpoint(EndpointName=endpoint_name)
    except Exception as exc:
        if is_missing_resource(exc):
            return
        raise
    raise ValueError(
        f"endpoint {endpoint_name} already exists ({described.get('EndpointStatus')}); call "
        f"undeploy({endpoint_name!r}) first, or pass another endpoint_name"
    )


def create_or_reuse_model_package(
    sagemaker: Any, job: FineTuneJob, artifact: str, created: Dict[str, str]
) -> Dict[str, Any]:
    """The job's model package, Completed: a package of the job's name over this artifact is reused."""
    name = job.job_name
    try:
        existing: Optional[Dict[str, Any]] = sagemaker.describe_model_package(ModelPackageName=name)
    except Exception as exc:
        if not is_missing_resource(exc):
            raise
        existing = None
    if existing is not None:
        sources = (existing.get("SourceAlgorithmSpecification") or {}).get("SourceAlgorithms") or []
        if not sources or sources[0].get("ModelDataUrl") != artifact:
            found = sources[0].get("ModelDataUrl") if sources else None
            raise ValueError(
                f"a model package named {name} already exists over {found or 'another artifact'}, not {artifact}; "
                f"delete it with sagemaker.delete_model_package(ModelPackageName={name!r}) and deploy again"
            )
        arn = existing["ModelPackageArn"]
        _LOG.info("reusing model package %s", arn)
    else:
        arn = sagemaker.create_model_package(
            ModelPackageName=name,
            ModelPackageDescription=f"Fine-tuned Nori model {job.model_name} from training job {name}",
            SourceAlgorithmSpecification={
                "SourceAlgorithms": [{"AlgorithmName": job.algorithm, "ModelDataUrl": artifact}]
            },
        )["ModelPackageArn"]
        _LOG.info("created model package %s", arn)
    created["model_package"] = arn
    return wait_for_model_package(sagemaker, arn, job=job, created=created)


def wait_for_model_package(sagemaker: Any, arn: str, *, job: FineTuneJob, created: Dict[str, str]) -> Dict[str, Any]:
    """Poll the package until SageMaker's validation Completed it; Failed and a timeout are raised."""
    deadline = time.monotonic() + _PACKAGE_TIMEOUT_S
    while True:
        described = sagemaker.describe_model_package(ModelPackageName=arn)
        status = described["ModelPackageStatus"]
        if status == "Completed":
            return described
        if status == "Failed":
            details = (described.get("ModelPackageStatusDetails") or {}).get("ValidationStatuses") or []
            reasons = "; ".join(item["FailureReason"] for item in details if item.get("FailureReason"))
            raise FineTuningError(
                f"model package {arn} Failed SageMaker's validation: {reasons or 'no reason given'}. Delete it "
                f"with sagemaker.delete_model_package(ModelPackageName={arn!r}) before deploying this job again",
                job_name=job.job_name,
                status=status,
                failure_reason=reasons or None,
                created=created,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APITimeoutError(
                f"model package {arn} is still {status} after {format_seconds(_PACKAGE_TIMEOUT_S)} s of "
                "SageMaker's validation; call deploy() again to reuse it once it is Completed, or delete it with "
                f"sagemaker.delete_model_package(ModelPackageName={arn!r})"
            )
        time.sleep(min(_PACKAGE_POLL_S, remaining))


def check_instance_type_served(package: Dict[str, Any], instance_type: str) -> List[str]:
    """The instance types the package serves on; an instance type outside them is refused."""
    supported = list((package.get("InferenceSpecification") or {}).get("SupportedRealtimeInferenceInstanceTypes") or [])
    if supported and instance_type not in supported:
        raise ValueError(
            f"instance_type {instance_type!r} is not one model package {package['ModelPackageArn']} serves on; "
            f"choose one of {', '.join(supported)}. The package is kept, and the next deploy() of this job reuses it."
        )
    return supported


def create_or_reuse_model(
    sagemaker: Any,
    sagemaker_model_name: str,
    package_arn: str,
    package_name: str,
    role_arn: str,
    created: Dict[str, str],
) -> None:
    """The SageMaker model on the package, run as ``role_arn``.

    A model of that sagemaker_model_name on this package and with this execution role is reused. One on another package, or one a
    failed deploy() left with another role, is refused: SageMaker cannot change a model's role, so reusing it would
    silently serve under the old permissions.
    """
    try:
        existing = sagemaker.describe_model(ModelName=sagemaker_model_name)
    except Exception as exc:
        if not is_missing_resource(exc):
            raise
        sagemaker.create_model(
            ModelName=sagemaker_model_name,
            PrimaryContainer={"ModelPackageName": package_arn},
            ExecutionRoleArn=role_arn,
            EnableNetworkIsolation=True,
        )
        _LOG.info("created SageMaker model %s", sagemaker_model_name)
        created["model"] = sagemaker_model_name
        return
    serves = (existing.get("PrimaryContainer") or {}).get("ModelPackageName")
    if serves not in (package_arn, package_name):
        raise ValueError(
            f"a SageMaker model named {sagemaker_model_name} already exists on {serves or 'something else'}, not on {package_arn}; "
            f"delete it with sagemaker.delete_model(ModelName={sagemaker_model_name!r}), or pass another endpoint_name"
        )
    existing_role = existing.get("ExecutionRoleArn")
    if existing_role != role_arn:
        raise ValueError(
            f"a SageMaker model named {sagemaker_model_name} already exists on {package_arn} with execution role "
            f"{existing_role or 'none'}, not {role_arn}, and SageMaker cannot change a model's role; delete it with "
            f"sagemaker.delete_model(ModelName={sagemaker_model_name!r}) and deploy again, or pass another endpoint_name"
        )
    _LOG.info("reusing SageMaker model %s", sagemaker_model_name)
    created["model"] = sagemaker_model_name


def create_endpoint_config(
    sagemaker: Any, *, config_name: str, model_name: str, instance_type: str, output_root: str
) -> None:
    """One asynchronous endpoint config: one instance of the model, outputs and failures under ``output_root``."""
    sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": _VARIANT_NAME,
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": f"{output_root}/outputs/",
                "S3FailurePath": f"{output_root}/failures/",
            },
            "ClientConfig": {"MaxConcurrentInvocationsPerInstance": _MAX_CONCURRENT_INVOCATIONS_PER_INSTANCE},
        },
    )


def _is_missing_endpoint_config(exc: BaseException) -> bool:
    """Whether CreateEndpoint refused only because its endpoint config is not visible to it yet."""
    code, message = read_aws_error(exc)
    return code == "ValidationException" and is_missing_resource(exc) and "endpoint config" in message.lower()


def wait_for_endpoint_config(sagemaker: Any, config_name: str) -> None:
    """Poll DescribeEndpointConfig until it sees a new config; a config still missing after the last poll raises."""
    for attempt in range(1, _CONFIG_VISIBILITY_ATTEMPTS + 1):
        try:
            sagemaker.describe_endpoint_config(EndpointConfigName=config_name)
            return
        except Exception as exc:
            if not is_missing_resource(exc) or attempt == _CONFIG_VISIBILITY_ATTEMPTS:
                raise
        time.sleep(_CONFIG_VISIBILITY_POLL_S)


def create_endpoint(sagemaker: Any, endpoint_name: str, config_name: str) -> None:
    """Create the endpoint once its config is visible, retrying only CreateEndpoint's "config not found" refusal.

    That refusal is the eventually consistent lookup AWS documents; every other error, any other
    ``ValidationException`` included, reaches the caller at once.
    """
    wait_for_endpoint_config(sagemaker, config_name)
    for attempt in range(1, _CONFIG_VISIBILITY_ATTEMPTS + 1):
        try:
            sagemaker.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
            return
        except Exception as exc:
            if not _is_missing_endpoint_config(exc) or attempt == _CONFIG_VISIBILITY_ATTEMPTS:
                raise
            _LOG.info("CreateEndpoint does not see endpoint config %s yet; trying again", config_name)
        time.sleep(_CONFIG_VISIBILITY_POLL_S)


def wait_for_endpoint_in_service(
    sagemaker: Any,
    endpoint_name: str,
    *,
    job: FineTuneJob,
    instance_type: str,
    supported: List[str],
    created: Dict[str, str],
    timeout_s: float,
) -> None:
    """Poll the endpoint until InService; Failed raises with SageMaker's reason, a timeout names undeploy()."""
    deadline = time.monotonic() + timeout_s
    last: Optional[str] = None
    while True:
        described = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = described["EndpointStatus"]
        if status != last:
            _LOG.info("endpoint %s: %s", endpoint_name, status)
            last = status
        if status == "InService":
            return
        if status == "Failed":
            reason = described.get("FailureReason") or "no reason given"
            problem = f"endpoint {endpoint_name} Failed: {reason}."
            if "InsufficientInstanceCapacity" in reason:
                others = [name for name in supported if name != instance_type]
                elsewhere = f", or with instance_type one of {', '.join(others)}" if others else ""
                problem += (
                    f" SageMaker has no spare {instance_type} capacity now: call undeploy({endpoint_name!r}), "
                    f"then deploy() again later{elsewhere}; the model package is reused."
                )
            raise FineTuningError(
                format_failed_deploy_message(problem, created, endpoint_name),
                job_name=job.job_name,
                status=status,
                failure_reason=reason,
                created=created,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APITimeoutError(
                f"endpoint {endpoint_name} is still {status} after {format_seconds(timeout_s)} s; it keeps "
                f"starting and billing on SageMaker. Check it later with describe_endpoint, or call "
                f"undeploy({endpoint_name!r}) to delete it."
            )
        time.sleep(min(_ENDPOINT_POLL_S, remaining))


def create_deployment(
    sagemaker: Any,
    *,
    job: FineTuneJob,
    artifact: str,
    endpoint_name: str,
    config_name: str,
    instance_type: str,
    output_root: str,
    role_arn: str,
    wait: bool,
    timeout_s: float,
) -> Tuple[str, str]:
    """Make or reuse the package, the model, the config and the endpoint, in order; ``(package ARN, status)``.

    A refusal (``ValueError``), a timeout or a :class:`FineTuningError` reaches the caller as raised; any other
    failure after something was made becomes a :class:`FineTuningError` whose ``created`` names it.
    """
    created: Dict[str, str] = {}
    try:
        package = create_or_reuse_model_package(sagemaker, job, artifact, created)
        package_arn = package["ModelPackageArn"]
        supported = check_instance_type_served(package, instance_type)
        create_or_reuse_model(sagemaker, endpoint_name, package_arn, job.job_name, role_arn, created)
        create_endpoint_config(
            sagemaker,
            config_name=config_name,
            model_name=endpoint_name,
            instance_type=instance_type,
            output_root=output_root,
        )
        created["endpoint_config"] = config_name
        create_endpoint(sagemaker, endpoint_name, config_name)
        created["endpoint"] = endpoint_name
        _LOG.info("creating endpoint %s on %s for model %s", endpoint_name, instance_type, job.model_name)
        if not wait:
            return package_arn, "Creating"
        wait_for_endpoint_in_service(
            sagemaker,
            endpoint_name,
            job=job,
            instance_type=instance_type,
            supported=supported,
            created=created,
            timeout_s=timeout_s,
        )
        return package_arn, "InService"
    except (FineTuningError, APITimeoutError, ValueError):
        raise
    except Exception as exc:
        if not created:
            raise
        raise FineTuningError(
            format_failed_deploy_message(
                f"deploy() of training job {job.job_name} failed: {exc}.", created, endpoint_name
            ),
            job_name=job.job_name,
            status=job.status,
            failure_reason=str(exc),
            created=created,
        ) from exc


def delete_endpoint_and_wait(sagemaker: Any, endpoint_name: str) -> None:
    """Delete the endpoint and wait until SageMaker no longer describes it; an endpoint already gone is fine."""
    try:
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
    except Exception as exc:
        if is_missing_resource(exc):
            return
        raise
    _LOG.info("deleting endpoint %s", endpoint_name)
    deadline = time.monotonic() + _ENDPOINT_DELETE_TIMEOUT_S
    while True:
        try:
            sagemaker.describe_endpoint(EndpointName=endpoint_name)
        except Exception as exc:
            if is_missing_resource(exc):
                return
            raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APITimeoutError(
                f"endpoint {endpoint_name} still exists {format_seconds(_ENDPOINT_DELETE_TIMEOUT_S)} s after "
                f"DeleteEndpoint; call undeploy({endpoint_name!r}) again once it is gone"
            )
        time.sleep(min(_ENDPOINT_DELETE_POLL_S, remaining))


def read_endpoint_config_model(sagemaker: Any, config_name: str) -> Optional[str]:
    """The SageMaker model an endpoint config serves; None when the config is already gone."""
    try:
        described = sagemaker.describe_endpoint_config(EndpointConfigName=config_name)
    except Exception as exc:
        if is_missing_resource(exc):
            return None
        raise
    variants = described.get("ProductionVariants") or []
    return variants[0].get("ModelName") if variants else None


def list_deployed_endpoint_configs(sagemaker: Any, endpoint_name: str, model_name: str) -> List[str]:
    """The endpoint configs deploy() made for this endpoint: ``<endpoint_name[:47]>-<UTC timestamp>``.

    The timestamp must match exactly, so the configs of an endpoint whose name merely starts with this one
    (``acme`` and ``acme-2``) are never listed. A name of 47 characters or more is cut to the same 47 as any
    longer name sharing them, so its candidates are described and only those serving ``model_name`` are kept;
    another running endpoint's configs are never listed.
    """
    prefix = f"{endpoint_name[:_ENDPOINT_CONFIG_NAME_CHARS]}-"
    request: Dict[str, Any] = {"NameContains": prefix}
    names: List[str] = []
    while True:
        page = sagemaker.list_endpoint_configs(**request)
        for config in page.get("EndpointConfigs") or []:
            name = config["EndpointConfigName"]
            if name.startswith(prefix) and _TIMESTAMP_PATTERN.fullmatch(name[len(prefix) :]):
                names.append(name)
        token = page.get("NextToken")
        if not token:
            break
        request["NextToken"] = token
    if len(endpoint_name) < _ENDPOINT_CONFIG_NAME_CHARS:
        return names
    return [name for name in names if read_endpoint_config_model(sagemaker, name) == model_name]


def delete_deployment(sagemaker: Any, *, endpoint_name: str, model_name: str) -> None:
    """Delete the endpoint and wait for it, then its endpoint configs, then the SageMaker model; keep the package.

    A resource already gone is skipped, so this can run again after a partial failure.
    """
    delete_endpoint_and_wait(sagemaker, endpoint_name)
    for config_name in list_deployed_endpoint_configs(sagemaker, endpoint_name, model_name):
        try:
            sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
            _LOG.info("deleted endpoint config %s", config_name)
        except Exception as exc:
            if not is_missing_resource(exc):
                raise
    try:
        sagemaker.delete_model(ModelName=model_name)
        _LOG.info("deleted SageMaker model %s", model_name)
    except Exception as exc:
        if not is_missing_resource(exc):
            raise
