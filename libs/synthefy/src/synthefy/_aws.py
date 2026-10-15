"""The AWS plumbing the SageMaker inference client and the fine-tuner share.

One place builds boto3 clients through the standard credential chain with one botocore config, refuses a bad timeout
or retry count, splits ``s3://`` URIs, lists the S3 keys under a prefix, applies SageMaker's resource naming rule and
reads what an AWS error says. boto3 is the optional ``aws`` extra, so it is imported only inside
:func:`load_aws_sdk`, when a client is built: importing this module loads neither boto3 nor botocore, and an AWS
error is recognized by its ``response`` mapping rather than by its class.
"""

import math
import re
from datetime import datetime
from numbers import Integral, Real
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

#: Appended to botocore's user agent on every AWS client this package builds, unless the caller passes its own, so
#: requests from this package can be told apart in CloudTrail.
AWS_USER_AGENT_EXTRA = "synthefy-python"

#: SageMaker's rule for training job, endpoint and endpoint config names; the length limit is separate.
_RESOURCE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")
MAX_RESOURCE_NAME_CHARS = 63
#: The UTC timestamp a generated SageMaker resource name ends with: unique per second.
RESOURCE_NAME_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
RESOURCE_NAME_TIMESTAMP_CHARS = len(datetime(2000, 1, 1).strftime(RESOURCE_NAME_TIMESTAMP_FORMAT))
#: The logger the fine-tuner and its deployment steps report progress on.
FINE_TUNING_LOGGER_NAME = "synthefy.fine_tune"


def parse_s3_uri(uri: str, *, name: str, require_prefix: bool = False) -> Tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into its bucket and key.

    ``name`` is the argument the refusal names, so every S3 location the package takes
    (the staging prefix, a fine-tuning job's data and output paths) shares one rule while
    the message still points at the argument the caller passed.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"{name} must be an s3://bucket/prefix URI")
    key = parsed.path.lstrip("/").rstrip("/")
    if require_prefix and not key:
        raise ValueError(f"{name} must include a customer-owned prefix, not only a bucket")
    return parsed.netloc, key


def list_s3_keys(s3: Any, *, bucket: str, prefix: str, max_keys: int) -> Tuple[List[str], bool]:
    """The first ``max_keys`` keys that start with ``prefix``, in S3's order, and whether more exist.

    One ListObjectsV2 call: S3 lists keys in UTF-8 binary order, so a key equal to the prefix comes first.
    """
    page = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
    keys = [item["Key"] for item in page.get("Contents") or []]
    return keys, bool(page.get("IsTruncated"))


def read_aws_error(exc: BaseException) -> Tuple[str, str]:
    """Read ``(code, message)`` from a botocore ``ClientError`` by duck typing.

    The package never imports botocore at module scope (it is the optional ``aws`` extra),
    so an AWS error is recognized by its ``response`` mapping rather than by its class. Any
    other exception reads as an empty code and its own text.
    """
    response = getattr(exc, "response", None)
    error = response.get("Error", {}) if isinstance(response, dict) else {}
    return str(error.get("Code", "")), str(error.get("Message") or exc)


def is_missing_s3_object(exc: BaseException) -> bool:
    """Whether an S3 read failed only because the object does not exist."""
    return read_aws_error(exc)[0] in {"404", "NoSuchKey", "NotFound"}


def is_missing_resource(exc: BaseException) -> bool:
    """Whether a SageMaker describe or delete failed only because the resource does not exist.

    SageMaker answers a missing endpoint, endpoint config, model or model package with a ``ValidationException``
    whose message says so ("Could not find endpoint", "ModelPackage ... does not exist"), and a missing training job
    with ``ResourceNotFound``. Any other ``ValidationException`` is a real refusal and must reach the caller.
    """
    code, message = read_aws_error(exc)
    if code in ("ResourceNotFound", "ResourceNotFoundException"):
        return True
    if code != "ValidationException":
        return False
    text = message.lower()
    return any(phrase in text for phrase in ("does not exist", "could not be found", "could not find"))


def check_sagemaker_resource_name(value: str, argument: str) -> None:
    """Refuse a name SageMaker would refuse, naming the argument, before any resource is created."""
    if len(value) > MAX_RESOURCE_NAME_CHARS or not _RESOURCE_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"{argument} must be 1 to {MAX_RESOURCE_NAME_CHARS} letters, digits and '-', starting and ending with "
            f"a letter or digit (SageMaker's naming rule); got {value!r}"
        )


def format_seconds(value: float) -> str:
    """A number of seconds as a message about an AWS wait or limit states it: ``7200``, not ``7200.0``."""
    return str(int(value)) if float(value).is_integer() else f"{value:.10g}"


def load_aws_sdk() -> Tuple[Any, Any]:
    """Load the optional AWS SDK only when a SageMaker deployment is selected."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError('A SageMaker deployment needs the AWS extra: install `pip install "synthefy[aws]"`.') from exc
    return boto3, Config


def build_aws_client_config(timeout: float, max_retries: int, user_agent_extra: str) -> Any:
    """Build the botocore ``Config`` shared by every AWS client this package creates.

    One place sets the connect and read timeouts, botocore's standard retry mode with
    ``max_retries`` retries after the first attempt, and the user-agent suffix that
    attributes requests, so the inference and fine-tuning clients cannot drift apart.
    """
    _, Config = load_aws_sdk()
    return Config(
        connect_timeout=timeout,
        read_timeout=timeout,
        retries={"mode": "standard", "total_max_attempts": max_retries + 1},
        user_agent_extra=user_agent_extra,
    )


def check_timeout_and_retries(timeout: Any, max_retries: Any) -> None:
    """Refuse a timeout or retry count before any client is built.

    Shared by :class:`~synthefy.SynthefyNoriClient` and :class:`~synthefy.SynthefyNoriFineTuner`, so
    both accept exactly the same values. ``bool`` is refused although it is an ``int``,
    because ``timeout=True`` is a mistake, not one second.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, Real) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite number greater than zero")
    if isinstance(max_retries, bool) or not isinstance(max_retries, Integral) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")


def create_aws_clients(
    service_names: Sequence[str],
    *,
    region_name: Optional[str],
    timeout: float,
    max_retries: int,
    user_agent_extra: str,
) -> Tuple[Any, ...]:
    """Create one boto3 client per service name through boto3's credential chain.

    The inference client takes ``("sagemaker-runtime", "s3")``, the fine-tuner
    ``("sagemaker", "logs", "s3")``; both get the same session, config and user agent.

    Deliberately constructs an argument-free ``Session``: environment variables,
    shared config/credentials, web identity (including GitHub OIDC), ECS/EC2 role
    credentials, and SSO profiles retain boto3's normal precedence. The public
    client never accepts raw AWS access keys.
    """
    boto3, _ = load_aws_sdk()
    config = build_aws_client_config(timeout, max_retries, user_agent_extra)
    session = boto3.Session()
    return tuple(session.client(name, region_name=region_name, config=config) for name in service_names)
