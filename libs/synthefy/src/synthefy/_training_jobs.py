"""The SageMaker training job API in the fine-tuner's terms, and the checks a job needs before it is created.

A :class:`~synthefy.FineTuneJobRequest` becomes CreateTrainingJob's parameters here (with a ``CheckpointConfig`` when
it names a checkpoint folder), and a DescribeTrainingJob response becomes a :class:`~synthefy.FineTuneJob`. The checks
that run just before CreateTrainingJob live here too: the request's own checks again, a ``max_runtime_s`` that ends the
job before ``max_training_hours`` stops its models, the Algorithm's region, and a one-object channel that would read
more than one object. The one check that calls AWS takes the S3 client it is given, so this module imports no AWS SDK.
"""

import functools
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Collection, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union, get_args

import numpy as np

from pydantic import ValidationError
from synthefy._aws import (
    MAX_RESOURCE_NAME_CHARS,
    RESOURCE_NAME_TIMESTAMP_CHARS,
    RESOURCE_NAME_TIMESTAMP_FORMAT,
    RESOURCE_NAME_TOKEN_CHARS,
    list_s3_keys,
    load_parquet_reader,
    parse_s3_uri,
)
from synthefy.errors import FineTuningError
from synthefy.nori_data_models import (
    SAGEMAKER_MAX_RUNTIME_S,
    _NONE_HYPERPARAMETER,
    _RUNTIME_MARGIN_S,
    TRAIN_CHANNEL,
    VALIDATION_CHANNEL,
    _CHANNEL_FILE_SUFFIXES,
    _CHANNEL_SKIPPED_NAME_PREFIXES,
    MULTI_GPU_INSTANCE_TYPES,
    BaseModelName,
    FineTuneHyperparameters,
    FineTuneJob,
    FineTuneJobRequest,
    TaskTable,
    TrainingPlan,
    get_runtime_for_training_cap,
    serialize_hyperparameters,
    get_task_tables,
    is_job_of_tasks,
)

#: A default job name is nori-<model_name>-<UTC timestamp>-<token>, the model name cut so the whole fits SageMaker's
#: 63 characters. The model name brings its ft- prefix, so a job reads nori-ft-<name>-<timestamp>-<token>.
_JOB_NAME_PREFIX = "nori-"
_JOB_NAME_MODEL_CHARS = (
    MAX_RESOURCE_NAME_CHARS - len(_JOB_NAME_PREFIX) - 1 - RESOURCE_NAME_TIMESTAMP_CHARS - 1 - RESOURCE_NAME_TOKEN_CHARS
)
#: How many columns a refusal names.
_MAX_NAMED_COLUMNS = 10
#: How many look-alike keys a refused one-object channel names.
_MAX_LISTED_SIBLINGS = 10


class _SameAsJob:
    """The type of ``SAME_AS_JOB``."""

    def __repr__(self) -> str:
        return "<same as the job>"


#: A setting that keeps what the job had, where None is a value of its own.
SAME_AS_JOB: Any = _SameAsJob()


def build_job_name(model_name: str, now: datetime, token: str) -> str:
    """nori-<model_name>-<UTC timestamp>-<token>, cut to SageMaker's 63 characters. The timestamp is unique per
    second; ``token`` (random, ``RESOURCE_NAME_TOKEN_CHARS`` long) keeps two jobs started in the same second, or two
    long names cut to the same characters, apart."""
    return f"{_JOB_NAME_PREFIX}{model_name[:_JOB_NAME_MODEL_CHARS]}-{now:{RESOURCE_NAME_TIMESTAMP_FORMAT}}-{token}"


def build_training_job_request(
    request: FineTuneJobRequest,
    job_name: str,
    *,
    trains_on_whole_table: Optional[bool] = True,
    resume: bool = False,
) -> Dict[str, Any]:
    """CreateTrainingJob's parameters for one request; network isolation always, as the Algorithm requires.

    ``trains_on_whole_table=False`` gives the hyperparameters and MaxRuntimeInSeconds of an Algorithm version that
    trains each model for a fixed number of steps (:func:`get_serialized_legacy_hyperparameters`); ``None``, for an
    Algorithm that could not be described, gives only the hyperparameters every version reads
    (:func:`get_hyperparameters_every_version_reads`). ``resume`` adds the ``RESUME_HYPERPARAMETER`` as ``"true"``,
    which only an Algorithm version that declares it accepts (:func:`has_resume_hyperparameter`).

    File input mode, the one a SageMaker Algorithm's channels can serve this job in: SageMaker copies each channel to
    the instance's local disk before the job starts. A ``checkpoint_uri`` becomes ``CheckpointConfig`` with only its
    ``S3Uri``: the job finds the folder at the default ``LocalPath``.
    """
    channels = [(TRAIN_CHANNEL, request.training_data)]
    if request.validation_data is not None:
        channels.append((VALIDATION_CHANNEL, request.validation_data))
    output: Dict[str, str] = {"S3OutputPath": request.output_path}
    if request.kms_key_id is not None:
        output["KmsKeyId"] = request.kms_key_id
    if trains_on_whole_table is False:
        hyperparameters = get_serialized_legacy_hyperparameters(request.hyperparameters)
    elif trains_on_whole_table is None:
        hyperparameters = get_hyperparameters_every_version_reads(request.hyperparameters)
    else:
        hyperparameters = serialize_hyperparameters(request.hyperparameters)
    if resume:
        hyperparameters[RESUME_HYPERPARAMETER] = "true"
    parameters: Dict[str, Any] = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {"AlgorithmName": request.algorithm, "TrainingInputMode": "File"},
        "RoleArn": request.role_arn,
        "HyperParameters": hyperparameters,
        "InputDataConfig": [
            {
                "ChannelName": name,
                "ContentType": "application/x-parquet",
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
        "StoppingCondition": {
            "MaxRuntimeInSeconds": (
                get_legacy_max_runtime_s(request) if trains_on_whole_table is False else request.max_runtime_s
            )
        },
        "EnableNetworkIsolation": True,
    }
    if request.checkpoint_uri is not None:
        parameters["CheckpointConfig"] = {"S3Uri": request.checkpoint_uri}
    return parameters


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


def build_request_from_job_description(
    description: Dict[str, Any],
    *,
    instance_type: Optional[str] = None,
    max_training_hours: Any = SAME_AS_JOB,
) -> FineTuneJobRequest:
    """The request that rebuilds one DescribeTrainingJob response, the inverse of :func:`build_training_job_request`:
    its Algorithm, channels, output path, hyperparameters, instance type, role, volume, KMS key, checkpoint folder and
    MaxRuntimeInSeconds, with no job name. Hyperparameters the request does not take (``RESUME_HYPERPARAMETER``) are
    dropped, and those of an Algorithm version that trains each model for a fixed number of steps are read back as
    the request's. ``instance_type`` replaces the job's when given; ``max_training_hours`` replaces the job's unless it
    is ``SAME_AS_JOB``, and the runtime is then derived from it as a new request derives it. Raises ``ValueError``
    naming the argument when ``instance_type`` or ``max_training_hours`` is one a request cannot take, and
    :class:`FineTuningError` naming the job when the description does not make a valid request, or when the job has
    managed spot training, a VPC or a volume KMS key, none of which a request carries. A job's tags are not in its
    description, so the request has none."""
    job_name = description.get("TrainingJobName", "")
    specification = description.get("AlgorithmSpecification") or {}
    channels = {
        channel.get("ChannelName"): ((channel.get("DataSource") or {}).get("S3DataSource") or {}).get("S3Uri")
        for channel in description.get("InputDataConfig") or []
    }
    output = description.get("OutputDataConfig") or {}
    resources = description.get("ResourceConfig") or {}
    settings_a_request_lacks = [
        name
        for name, is_set in (
            ("managed spot training (EnableManagedSpotTraining)", description.get("EnableManagedSpotTraining")),
            ("a VPC (VpcConfig)", description.get("VpcConfig")),
            ("a volume KMS key (ResourceConfig.VolumeKmsKeyId)", resources.get("VolumeKmsKeyId")),
        )
        if is_set
    ]
    if settings_a_request_lacks:
        raise FineTuningError(
            f"cannot resume training job {job_name}: it was created with {', '.join(settings_a_request_lacks)}, "
            "which a job this client creates does not have. To continue it with the same settings, call "
            "CreateTrainingJob with the job's own parameters, a new TrainingJobName and the hyperparameter "
            f'{RESUME_HYPERPARAMETER} set to "true".',
            job_name=job_name,
            status=description.get("TrainingJobStatus"),
        )
    hyperparameters = get_hyperparameters_from_strings(description.get("HyperParameters") or {})
    if max_training_hours is not SAME_AS_JOB:
        hyperparameters["max_training_hours"] = (
            _NONE_HYPERPARAMETER if max_training_hours is None else str(max_training_hours)
        )
    data: Dict[str, Any] = {
        "algorithm": specification.get("AlgorithmName"),
        "training_data": channels.get(TRAIN_CHANNEL),
        "validation_data": channels.get(VALIDATION_CHANNEL),
        "output_path": output.get("S3OutputPath"),
        "hyperparameters": hyperparameters,
        "instance_type": instance_type if instance_type is not None else resources.get("InstanceType"),
        "role_arn": description.get("RoleArn"),
        "volume_size_gb": resources.get("VolumeSizeInGB"),
        "kms_key_id": output.get("KmsKeyId"),
        "checkpoint_uri": (description.get("CheckpointConfig") or {}).get("S3Uri"),
        "max_runtime_s": (
            None
            if max_training_hours is not SAME_AS_JOB
            else (description.get("StoppingCondition") or {}).get("MaxRuntimeInSeconds")
        ),
    }
    # The arguments the caller gave, by where a request's validation names them.
    argument_by_error_location: Dict[Tuple[Any, ...], Tuple[str, Any]] = {}
    if instance_type is not None:
        argument_by_error_location[("instance_type",)] = ("instance_type", instance_type)
    if max_training_hours is not SAME_AS_JOB:
        argument_by_error_location[("hyperparameters", "max_training_hours")] = (
            "max_training_hours",
            max_training_hours,
        )
    try:
        return FineTuneJobRequest.model_validate({name: value for name, value in data.items() if value is not None})
    except ValidationError as exc:
        for error in exc.errors():
            argument = argument_by_error_location.get(tuple(error["loc"]))
            if argument is not None:
                raise ValueError(
                    f"{argument[0]}={argument[1]!r} is not one a request can take: {error['msg']}"
                ) from exc
        raise FineTuningError(
            f"training job {job_name} does not describe a request this client can send again: {exc}",
            job_name=job_name,
            status=description.get("TrainingJobStatus"),
        ) from exc


def revalidate_request(request: FineTuneJobRequest) -> FineTuneJobRequest:
    """The request rebuilt from its own fields, so a value changed inside it after it was built is checked again.

    The request is frozen, but its ``hyperparameters`` is a plain model; a base changed there would otherwise
    reach SageMaker with the instance type derived for the old one.
    """
    data = request.model_dump()
    if request._max_runtime_derived:
        data["max_runtime_s"] = None  # derived again from the hyperparameters as they stand now
    try:
        return FineTuneJobRequest.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            "the request no longer passes its own checks, so its hyperparameters were changed after it was built; "
            f"build a new FineTuneJobRequest. {exc}"
        ) from exc


def refuse_max_runtime_below_training_cap(request: FineTuneJobRequest, *, version_known: bool) -> None:
    """Refuse a ``max_runtime_s`` that ends the job before ``max_training_hours`` stops its models. Lowering
    ``max_training_hours`` is offered only when ``version_known``: a request to an Algorithm that could not be
    described cannot send it."""
    hours = request.hyperparameters.max_training_hours
    if hours is None:
        return
    needed = get_runtime_for_training_cap(hours)
    if request.max_runtime_s < needed:
        floor = f"set it to at least {needed:,} (max_training_hours x 3,600 + 1,800)"
        remedies = f"unset, {floor}, or lower max_training_hours" if version_known else f"unset or {floor}"
        raise ValueError(
            f"max_runtime_s {request.max_runtime_s:,} ends the job before max_training_hours {hours:g} stops its "
            f"models; leave max_runtime_s {remedies}"
        )


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


def refuse_lookalike_keys(s3: Any, uri: str, argument: str) -> None:
    """Refuse a one-object channel whose key other keys start with.

    SageMaker's S3Prefix channel reads every key that starts with the given text, so ``data/train.parquet`` also
    reads ``data/train.parquet.backup.parquet`` and the job would train on both tables concatenated.
    """
    bucket, key = parse_s3_uri(uri, name=argument)
    keys, truncated = list_s3_keys(s3, bucket=bucket, prefix=key, max_keys=_MAX_LISTED_SIBLINGS + 1)
    siblings = [name for name in keys if name != key]
    if not siblings:
        return
    shown = ", ".join(f"s3://{bucket}/{name}" for name in siblings[:_MAX_LISTED_SIBLINGS])
    more = " and more" if truncated or len(siblings) > _MAX_LISTED_SIBLINGS else ""
    raise ValueError(
        f"{argument}={uri!r} names one object, but SageMaker reads every key that starts with it, so the job "
        f"would also read {shown}{more}. Rename or move those objects, or put the table in a folder of its own "
        "and pass the folder, ending in '/'"
    )


# ------------------------------------------------------------------ Parquet footers
#: The four bytes a Parquet file starts and ends with.
_PARQUET_MAGIC = b"PAR1"
#: The longest Parquet footer the plan reads: a table of 10,000 columns in 200 row groups has a footer of about 230 MB.
_MAX_FOOTER_BYTES = 256 * 1024 * 1024
#: The bytes the first read takes from the end of each file; a longer footer takes a second read.
_FOOTER_TAIL_BYTES = 256 * 1024
#: The footers read at once, botocore's default connection pool.
_FOOTER_READ_THREADS = 10
#: Keys one ListObjectsV2 page asks for (S3's own maximum).
_LIST_PAGE_KEYS = 1000


class TableRefused(ValueError):
    """The job would refuse the tables the request names: a channel holds no Parquet file or a file of another
    extension, a file cannot be read as Parquet or its pandas metadata cannot be applied, a column it needs is
    missing, a file names a column twice or has other columns than the first, or the rows are too few."""


class FooterTooLong(ValueError):
    """A file's Parquet footer is longer than the plan reads. The job reads the file whole, so it has no such limit."""


class ParquetFooter(NamedTuple):
    """What the plan reads of one Parquet file's footer: its rows, the rows of each row group and its column labels
    as pandas reads them, less the index columns."""

    num_rows: int
    row_group_rows: List[int]
    columns: List[Any]


# Copies of the job's column rules; a contract test holds them equal.
def _table_to_frame(table: Any, pa: Any) -> Any:
    """``table`` as a pandas frame: decimal columns, and the null type pandas writes for a column of only missing
    values, become float64, and the pandas metadata restores the dtype of every other column."""
    converted = set()
    for index, arrow_field in enumerate(table.schema):
        if pa.types.is_decimal(arrow_field.type) or pa.types.is_null(arrow_field.type):
            table = table.set_column(index, arrow_field.name, table.column(index).cast(pa.float64()))
            converted.add(arrow_field.name)
    metadata = dict(table.schema.metadata or {})
    if converted and b"pandas" in metadata:
        pandas_metadata = json.loads(metadata[b"pandas"])
        for column in pandas_metadata.get("columns", []):
            if column.get("field_name", column.get("name")) in converted:
                column["pandas_type"] = "float64"
                column["numpy_type"] = "float64"
        table = table.replace_schema_metadata({**metadata, b"pandas": json.dumps(pandas_metadata).encode()})
    return table.to_pandas()


def _frame_columns(schema: Any, pa: Any) -> List[Any]:
    """The column labels of the frame a table of ``schema`` reads as, in order: the labels its pandas metadata gives
    (an int label stays an int), less the index columns it names. Raises ValueError when the metadata gives the
    columns too few, too many or repeated labels."""
    labels = list(_table_to_frame(schema.empty_table(), pa).columns)
    pandas_metadata = schema.pandas_metadata or {}
    index_columns = {entry for entry in pandas_metadata.get("index_columns", []) if isinstance(entry, str)}
    names = [column for column in schema.names if column not in index_columns]
    if len(labels) != len(names) or len(set(labels)) != len(labels):
        raise ValueError(f"its pandas metadata gives the columns {names} the labels {labels}")
    return labels


def _get_object_tail(s3: Any, bucket: str, key: str, size: int) -> bytes:
    """The last ``size`` bytes of the S3 object ``bucket``/``key``, or the whole object when it is shorter."""
    return s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=-{size}")["Body"].read()


def get_parquet_footer(s3: Any, bucket: str, key: str) -> ParquetFooter:
    """One object's footer, from one ranged GetObject of the object's last 256 KiB, or two when the footer is longer."""
    uri = f"s3://{bucket}/{key}"
    pa, pq = load_parquet_reader()
    tail = _get_object_tail(s3, bucket, key, _FOOTER_TAIL_BYTES)
    if len(tail) < 12 or tail[-4:] != _PARQUET_MAGIC:
        raise TableRefused(f"could not read {uri} as Parquet: it does not end in {_PARQUET_MAGIC!r}")
    length = int.from_bytes(tail[-8:-4], "little")
    if length + 8 > len(tail):
        if len(tail) < _FOOTER_TAIL_BYTES:
            raise TableRefused(f"could not read {uri} as Parquet: it is shorter than its footer")
        if length > _MAX_FOOTER_BYTES:
            raise FooterTooLong(f"{uri} gives a Parquet footer of {length:,} bytes, which this reader does not read")
        tail = _get_object_tail(s3, bucket, key, length + 8)
        if len(tail) != length + 8:
            raise TableRefused(f"could not read {uri} as Parquet: it is shorter than its footer")
    try:
        parquet = pq.ParquetFile(io.BytesIO(tail))
        metadata = parquet.metadata
        row_group_rows = [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]
        schema = parquet.schema_arrow
    except Exception as exc:  # pyarrow raises several error types for a malformed footer
        raise TableRefused(f"could not read {uri} as Parquet: {exc}") from exc
    seen = set()
    repeated = [name for name in schema.names if name in seen or seen.add(name)]
    if repeated:
        raise TableRefused(f"{uri} names the column {repeated[0]!r} more than once; every column needs its own name")
    try:
        columns = _frame_columns(schema, pa)
    except Exception as exc:  # pyarrow raises several error types for pandas metadata it cannot apply
        raise TableRefused(f"could not read {uri} as Parquet: {exc}") from exc
    return ParquetFooter(sum(row_group_rows), row_group_rows, columns)


def get_channel_table_keys(s3: Any, uri: str, argument: str) -> List[str]:
    """The keys of a channel's table files in key order, as the job reads them: every key under a folder (every
    page) less the empty ones and those with a part starting with ``_`` or ``.``, or the one object a file channel
    names. A key that is not Parquet is refused, as the job refuses it."""
    bucket, key = parse_s3_uri(uri, name=argument)
    if not uri.endswith("/"):
        return [key]
    prefix = f"{key}/" if key else ""
    keys: List[str] = []
    params: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": _LIST_PAGE_KEYS}
    while True:
        page = s3.list_objects_v2(**params)
        for item in page.get("Contents") or []:
            relative = item["Key"][len(prefix) :]
            if not relative or item.get("Size", 1) == 0:
                continue
            if any(part.startswith(_CHANNEL_SKIPPED_NAME_PREFIXES) for part in relative.split("/")):
                continue
            if not relative.lower().endswith(_CHANNEL_FILE_SUFFIXES):
                raise TableRefused(
                    f"{argument}={uri!r} holds {relative}, which is neither .parquet nor .pq; the job refuses it"
                )
            keys.append(item["Key"])
        token = page.get("NextContinuationToken")
        if not page.get("IsTruncated") or not token:
            break
        params["ContinuationToken"] = token
    if not keys:
        raise TableRefused(f"{argument}={uri!r} holds no .parquet or .pq file")
    return sorted(keys)


# ------------------------------------------------------------------ the training plan
# The job's figures, copied because the client cannot import the job; a contract test holds each copy equal to the
# job's.
#: The day USD_PER_HOUR was read from the AWS Price List API's AmazonSageMaker offer for us-east-1.
PRICES_AS_OF = "2026-10-27"
#: On-demand SageMaker training in us-east-1, USD per instance-hour as of PRICES_AS_OF; an estimate, not a bill.
USD_PER_HOUR: Dict[str, float] = {
    "ml.g5.xlarge": 1.408,
    "ml.g5.2xlarge": 1.515,
    "ml.g5.12xlarge": 7.09,
    "ml.g5.48xlarge": 20.36,
    "ml.g6.xlarge": 1.127,
    "ml.g6e.xlarge": 2.61,
    "ml.g6e.2xlarge": 2.80,
    "ml.g6e.12xlarge": 13.12,
    "ml.g6e.48xlarge": 37.66,
}
#: Seconds of one training step by instance type and base model, measured on a 50,000,000-row table of 20 numeric
#: feature columns (MEASURED_STEP_CELLS cells a step, MEASURED_SCORE_CELLS cells a score); a pair not listed has no
#: estimate. nori-1-5's ml.g5 figure is one measurement, on ml.g5.12xlarge, taken by every ml.g5 type.
SECONDS_PER_STEP_BY_INSTANCE_AND_BASE_MODEL: Dict[Tuple[str, str], float] = {
    ("ml.g5.xlarge", "nori-6m"): 2.085,
    ("ml.g5.2xlarge", "nori-6m"): 2.085,
    ("ml.g6.xlarge", "nori-6m"): 2.897,
    ("ml.g5.12xlarge", "nori-6m"): 2.085,
    ("ml.g5.48xlarge", "nori-6m"): 2.085,
    ("ml.g5.xlarge", "nori-1-5"): 6.0,
    ("ml.g5.2xlarge", "nori-1-5"): 6.0,
    ("ml.g5.12xlarge", "nori-1-5"): 6.0,
    ("ml.g5.48xlarge", "nori-1-5"): 6.0,
}
#: The cost of one validation score of MEASURED_SCORE_CELLS cells, in training steps of MEASURED_STEP_CELLS cells.
VALIDATION_COST_IN_STEPS = 1.79
#: One training step of the measured table.
MEASURED_STEP_CELLS = 349_985
#: One validation score of the measured table.
MEASURED_SCORE_CELLS = 1_867_776
#: The most cells one validation score reads on a GPU of 24 GB.
_MAX_SCORE_CELLS = 2_000_000
MIN_TRAIN_ROWS = 1000
MIN_STEPS_PER_MODEL = 100
VALIDATIONS_PER_MODEL = 100
#: The longest wait for a better validation score, in training steps, that the default validation interval gives (the
#: job's ``MAX_DEFAULT_PATIENCE_STEPS``); an interval the buyer sets is kept.
MAX_DEFAULT_PATIENCE_STEPS = 100
MAX_VAL_ROWS = 4096
_REPORT_HOLDOUT_FRACTION = 0.10
_SINGLE_CHILD_FOLDS = 5
_FIT_SAMPLE_ROWS = 1_000_000
_FIT_SAMPLE_CELLS = 100_000_000
_BUFFER_MIN_ROWS = 1_000_000
_BUFFER_CHUNKS = 8
_BUFFER_CELLS = 160_000_000
_BUFFER_FLOOR_ROWS = 100_000
_PARTIAL_READ_WARNING_FRACTION = 0.99
_FOLD_COUNT_BLOCK = 1 << 22
_STREAM_TRAIN_ROWS = 1
_STREAM_REPORT_HOLDOUT = 2
#: A training step's caps, the recipe's (``FineTuneRecipe``): cells, query rows, context rows, the context share of
#: a small buffer and the fewest context rows.
_MAX_STEP_CELLS = 350_000
_MAX_QUERY_ROWS_PER_STEP = 1024
_MAX_CONTEXT_ROWS_PER_STEP = 20480
_CONTEXT_RATIO = (0.6, 0.9)
_MIN_EPISODE_CONTEXT = 16
#: Above this many rows the plan takes each model's share as the table's even split rather than hashing every row
#: position; the fold hash's error there is a fraction of a percent.
_EXACT_PARTITION_ROWS = 1 << 24
#: The feature columns a member featurizes at most, and the derived columns it adds at most.
_MEMBER_MAX_FEATURES = 500
_MEMBER_DERIVED_COLUMNS = 10
#: The kind of a member, which sets its featurized width for a column count (see get_estimated_member_widths).
MemberKind = Literal["wide", "narrow", "narrow_every_column"]
#: The kind of each member a base fits, in the order it fits them.
MEMBER_KINDS_BY_BASE_MODEL: Dict[BaseModelName, Tuple[MemberKind, ...]] = {
    "nori-6m": ("wide",) * 4 + ("narrow",) * 4,
    "nori-1-5": ("wide", "narrow_every_column", "narrow", "narrow_every_column"),
    "nori-100m": ("wide",) * 4 + ("narrow",) * 4,
}


class RowPartition(NamedTuple):
    """How the job divides the training rows among its models: the bag and strategy it uses, the folds of its split,
    whether model ``k`` trains on fold ``k`` (a shard), each model's training rows and the plan's warnings."""

    bag: int
    strategy: str
    n_folds: int
    train_inside_fold: bool
    rows_per_model: List[int]
    warnings: List[str]


def count_report_holdout_rows(n_rows: int) -> int:
    """Rows the report hold-out takes from ``n_rows`` when no validation channel is given."""
    return min(MAX_VAL_ROWS, max(1, int(round(_REPORT_HOLDOUT_FRACTION * n_rows))))


def _min_table_rows(has_validation: bool) -> int:
    """Fewest rows a train channel may have: with a validation channel ``MIN_TRAIN_ROWS``; without one, a single
    model must keep ``MIN_TRAIN_ROWS`` after the report hold-out and the fifth it is scored on."""
    if has_validation:
        return MIN_TRAIN_ROWS
    rows = MIN_TRAIN_ROWS
    while (rows - count_report_holdout_rows(rows)) * (_SINGLE_CHILD_FOLDS - 1) // _SINGLE_CHILD_FOLDS < MIN_TRAIN_ROWS:
        rows += 1
    return rows


def _splitmix64(value: int) -> int:
    """The splitmix64 output for ``value``: the 64-bit constant :func:`_fold_of` derives from a seed."""
    mask = (1 << 64) - 1
    mixed = (value + 0x9E3779B97F4A7C15) & mask
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
    return mixed ^ (mixed >> 31)


def _fold_of(positions: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """The job's fold of each table row position: a splitmix64 finalizer over the position xor the seed's constant."""
    with np.errstate(over="ignore"):
        hashed = np.asarray(positions, dtype=np.int64).astype(np.uint64) ^ np.uint64(_splitmix64(seed))
        hashed ^= hashed >> np.uint64(30)
        hashed *= np.uint64(0xBF58476D1CE4E5B9)
        hashed ^= hashed >> np.uint64(27)
        hashed *= np.uint64(0x94D049BB133111EB)
        hashed ^= hashed >> np.uint64(31)
    return (hashed % np.uint64(n_folds)).astype(np.int64)


def _get_report_holdout_positions(n_rows: int, n_features: int, seed: int, has_validation: bool) -> np.ndarray:
    """The table positions of the job's report hold-out: the head of a seeded order of pass 0's seeded sample."""
    if has_validation:
        return np.empty(0, dtype=np.int64)
    fit_rows = max(_min_table_rows(False), min(_FIT_SAMPLE_ROWS, _FIT_SAMPLE_CELLS // max(1, n_features)))
    if fit_rows >= n_rows:
        keep = np.arange(n_rows, dtype=np.int64)
    else:
        keep = np.sort(np.random.default_rng([seed, _STREAM_TRAIN_ROWS]).choice(n_rows, fit_rows, replace=False))
    holdout = count_report_holdout_rows(len(keep))
    order = np.random.default_rng([seed, _STREAM_REPORT_HOLDOUT]).permutation(len(keep))
    return np.asarray(keep, dtype=np.int64)[np.sort(order[:holdout])]


def _count_eligible_rows(n_rows: int, held: np.ndarray, n_folds: int, seed: int, train_inside_fold: bool) -> List[int]:
    """The rows of each model's share of the table, one entry per fold: the table less the report hold-out ``held``
    and, with more than one fold, only fold ``k`` for model ``k`` when ``train_inside_fold`` (a shard), else all but
    fold ``k``; exact up to ``_EXACT_PARTITION_ROWS`` table rows, an even split among the folds above."""
    n_used = n_rows - len(held)
    if n_folds <= 1:
        return [n_used]
    if n_rows > _EXACT_PARTITION_ROWS:
        in_fold = np.full(n_folds, n_used // n_folds, dtype=np.int64)
        in_fold[: n_used % n_folds] += 1
    else:
        in_fold = np.zeros(n_folds, dtype=np.int64)
        for start in range(0, n_rows, _FOLD_COUNT_BLOCK):
            block = np.arange(start, min(start + _FOLD_COUNT_BLOCK, n_rows), dtype=np.int64)
            in_fold += np.bincount(_fold_of(block, n_folds, seed), minlength=n_folds)
        if len(held):
            in_fold -= np.bincount(_fold_of(held, n_folds, seed), minlength=n_folds)
    if train_inside_fold:
        return [int(count) for count in in_fold]
    return [int(n_used - count) for count in in_fold]


def get_table_partition(
    n_rows: int, n_features: int, hyperparameters: FineTuneHyperparameters, has_validation: bool
) -> RowPartition:
    """The job's division of a table of ``n_rows`` rows among its models, before any row is read: a shard below
    ``MIN_TRAIN_ROWS`` rows lowers the bag to the training rows over ``MIN_TRAIN_ROWS`` (at least 1), and then one at
    a time while the fold hash leaves a shard short."""
    held = _get_report_holdout_positions(n_rows, n_features, hyperparameters.seed, has_validation)
    n_training = n_rows - len(held)
    bag, strategy = hyperparameters.bag, hyperparameters.row_partition_strategy
    warnings: List[str] = []
    if strategy == "shard" and bag >= 2 and n_training // bag < MIN_TRAIN_ROWS:
        bag = max(1, n_training // MIN_TRAIN_ROWS)

    def get_folds_and_rows_per_model(bag: int, strategy: str) -> Tuple[int, bool, List[int]]:
        """``(n_folds, train_inside_fold, rows_per_model)`` for ``bag`` models under ``strategy``: the folds of the
        split, whether model k trains on fold k, and each model's training rows."""
        train_inside_fold = strategy == "shard" and bag >= 2
        if train_inside_fold or bag > 1:
            n_folds = bag
        else:
            n_folds = 1 if has_validation else _SINGLE_CHILD_FOLDS
        return (
            n_folds,
            train_inside_fold,
            _count_eligible_rows(n_rows, held, n_folds, hyperparameters.seed, train_inside_fold)[:bag],
        )

    n_folds, train_inside_fold, rows = get_folds_and_rows_per_model(bag, strategy)
    while train_inside_fold and min(rows) < MIN_TRAIN_ROWS and bag > 2:
        bag -= 1
        n_folds, train_inside_fold, rows = get_folds_and_rows_per_model(bag, strategy)
    if train_inside_fold and min(rows) < MIN_TRAIN_ROWS:
        bag = 1
        n_folds, train_inside_fold, rows = get_folds_and_rows_per_model(bag, strategy)
    if bag != hyperparameters.bag:
        warnings.append(
            f"bag is {bag}, not {hyperparameters.bag}: each model needs a shard of at least {MIN_TRAIN_ROWS:,} of the "
            f"{n_training:,} training rows"
        )
    if bag == 1 and not has_validation:
        warnings.append(
            "with bag 1 and no validation table the model trains on four fifths of the training rows and is scored "
            "on the other fifth; a validation table lets it train on every row"
        )
    if strategy == "folds" and bag > 1 and has_validation:
        warnings.append(
            f"folds: each model leaves out its validation fold, 1/{bag} of the training rows, although the validation "
            "table scores it"
        )
    return RowPartition(bag, strategy, n_folds, train_inside_fold, rows, warnings)


def _count_columns_with_derived(n_read: int) -> int:
    """The columns a member holds after adding the derived ones to ``n_read`` read columns."""
    return n_read + min(_MEMBER_DERIVED_COLUMNS, n_read * (n_read + 1) // 2)


def get_estimated_member_widths(n_features: int, base_model: BaseModelName) -> List[int]:
    """The featurized width of each member ``base_model`` fits, for a table of ``n_features`` feature columns of
    continuous values.

    For ``nori-6m`` and ``nori-100m``, measured on tables of 1 to 700 normal or integer columns, the widths are exact
    up to 256 columns and within 0.4% above. A column of few distinct values makes a member narrower."""
    n_capped_columns_with_derived = _count_columns_with_derived(min(n_features, _MEMBER_MAX_FEATURES))
    width_by_kind: Dict[MemberKind, int] = {
        "wide": 2 * n_capped_columns_with_derived + max(1, n_capped_columns_with_derived // 2) + 1,
        "narrow": n_capped_columns_with_derived + 1,
        "narrow_every_column": _count_columns_with_derived(n_features) + 1,
    }
    return [width_by_kind[kind] for kind in MEMBER_KINDS_BY_BASE_MODEL[base_model]]


def get_rows_and_targets_per_step(width: int, buffer_rows: Optional[int]) -> Tuple[int, int]:
    """``(rows, targets)`` of a training step of a member of featurized ``width``, from a full shuffle buffer of
    ``buffer_rows`` rows, as the job draws them."""
    row_budget = max(32, _MAX_STEP_CELLS // max(1, width))
    queries = min(_MAX_QUERY_ROWS_PER_STEP, max(8, row_budget // 4))
    context = max(_MIN_EPISODE_CONTEXT, min(_MAX_CONTEXT_ROWS_PER_STEP, row_budget - queries))
    if buffer_rows is not None and buffer_rows >= 2:
        ratio = sum(_CONTEXT_RATIO) / 2
        pool = int(np.clip(int(buffer_rows * ratio), 1, buffer_rows - 1))
        queries = min(queries, buffer_rows - pool)
        context = min(pool, max(_MIN_EPISODE_CONTEXT, min(_MAX_CONTEXT_ROWS_PER_STEP, row_budget - queries)))
    return context + queries, queries


def get_steps_to_read(rows_by_member: Sequence[int], n_rows: float) -> int:
    """The fewest training steps whose draws reach ``n_rows`` rows; step ``s`` (from 1) draws
    ``rows_by_member[s % len(rows_by_member)]`` rows, as the steps cycle the members."""
    n_members, cycle = len(rows_by_member), sum(rows_by_member)
    full = max(0, math.ceil(n_rows / cycle) - 1)
    steps, drawn = full * n_members, full * cycle
    while drawn < n_rows:
        steps += 1
        drawn += rows_by_member[steps % n_members]
    return steps


def get_rows_drawn(rows_by_member: Sequence[int], steps: int) -> int:
    """The rows ``steps`` training steps draw, step ``s`` (from 1) drawing
    ``rows_by_member[s % len(rows_by_member)]``."""
    n_members = len(rows_by_member)
    full = steps // n_members
    rest = range(full * n_members + 1, steps + 1)
    return full * sum(rows_by_member) + sum(rows_by_member[step % n_members] for step in rest)


def compute_shuffle_buffer_rows(largest_row_group: int, n_features: int, rows_eligible: int) -> int:
    """Rows of a model's shuffle buffer, at least 2 and at most the model's share of the table."""
    rows = min(
        max(_BUFFER_MIN_ROWS, _BUFFER_CHUNKS * largest_row_group),
        max(_BUFFER_CELLS // max(1, n_features), _BUFFER_FLOOR_ROWS),
    )
    return max(2, min(rows, rows_eligible))


def get_seconds_per_step(
    instance_type: str, base_model: str, validation_interval: int, step_cells: float, score_cells: float
) -> Optional[float]:
    """Seconds of one training step of ``step_cells`` cells with a validation score of ``score_cells`` cells every
    ``validation_interval`` steps; None for an instance type and base model not measured."""
    step = SECONDS_PER_STEP_BY_INSTANCE_AND_BASE_MODEL.get((instance_type, base_model))
    if step is None:
        return None
    score_share = VALIDATION_COST_IN_STEPS * score_cells / MEASURED_SCORE_CELLS / validation_interval
    return step * (step_cells / MEASURED_STEP_CELLS + score_share)


def get_validation_interval(hyperparameters: FineTuneHyperparameters, steps_per_model: int) -> int:
    """Steps between two validation scores, as the job sets them: ``validation_every_n_steps`` when set, else
    ``steps_per_model // VALIDATIONS_PER_MODEL``, at most ``MAX_DEFAULT_PATIENCE_STEPS // patience`` while patience is
    on, at least 1."""
    if hyperparameters.validation_every_n_steps is not None:
        return hyperparameters.validation_every_n_steps
    interval = steps_per_model // VALIDATIONS_PER_MODEL
    if hyperparameters.patience is not None:
        interval = min(interval, MAX_DEFAULT_PATIENCE_STEPS // hyperparameters.patience)
    return max(1, interval)


def get_patience_steps(patience: Optional[int], validation_interval: int) -> Optional[int]:
    """Training steps a model trains without a better validation score before ``patience`` scores, one every
    ``validation_interval`` steps, stop it: ``patience`` x ``validation_interval``; None when patience is off."""
    if patience is None:
        return None
    return patience * max(1, validation_interval)


def get_training_plan_from_counts(
    hyperparameters: FineTuneHyperparameters,
    *,
    n_rows: int,
    n_validation_rows: Optional[int],
    n_features: int,
    largest_row_group: int,
    instance_type: str,
    widths: Sequence[int],
    rows_per_step_is_estimate: bool = True,
    seconds_per_model: Optional[float] = None,
) -> TrainingPlan:
    """The job's :class:`~synthefy.TrainingPlan` for a train table of ``n_rows`` rows (row groups of at most
    ``largest_row_group``) and ``n_features`` feature columns, with a validation table of ``n_validation_rows`` rows
    or None, on ``instance_type``, for members of featurized ``widths``. With ``seconds_per_model``, each model stops
    after that many seconds, and the plan counts the steps that fit at its step time."""
    has_validation = n_validation_rows is not None
    partition = get_table_partition(n_rows, n_features, hyperparameters, has_validation)
    n_training = (
        n_rows
        if has_validation
        else n_rows - len(_get_report_holdout_positions(n_rows, n_features, hyperparameters.seed, False))
    )
    rows_per_model = max(partition.rows_per_model)
    buffer_rows = compute_shuffle_buffer_rows(largest_row_group, n_features, rows_per_model)
    step_rows = [get_rows_and_targets_per_step(width, buffer_rows) for width in widths]
    member_rows = [rows for rows, _ in step_rows]
    rows_per_step = sum(member_rows) / len(step_rows)
    targets_per_step = sum(targets for _, targets in step_rows) / len(step_rows)
    if hyperparameters.n_steps_per_model is not None:
        steps = hyperparameters.n_steps_per_model
    else:
        n_passes = 1.0 if hyperparameters.n_passes_per_row is None else hyperparameters.n_passes_per_row
        steps = max(MIN_STEPS_PER_MODEL, get_steps_to_read(member_rows, n_passes * rows_per_model))
    validation_interval = get_validation_interval(hyperparameters, steps)
    patience_steps = get_patience_steps(hyperparameters.patience, validation_interval)
    step_cells = sum(rows * width for rows, width in zip(member_rows, widths)) / len(widths)
    n_context = min(_MAX_CONTEXT_ROWS_PER_STEP, rows_per_model)
    query_pool = n_validation_rows if has_validation else n_training - rows_per_model
    n_queries = max(0, min(MAX_VAL_ROWS, query_pool))
    score_cells = min(_MAX_SCORE_CELLS, (n_context + n_queries) * widths[0])
    seconds_per_step = get_seconds_per_step(
        instance_type, hyperparameters.base_model, validation_interval, step_cells, score_cells
    )
    requested = steps
    if seconds_per_model is not None and seconds_per_step is not None:
        steps = min(steps, max(1, math.floor(seconds_per_model / seconds_per_step)))
    rows_read = get_rows_drawn(member_rows, steps)
    passes = rows_read / rows_per_model
    share_read = min(1.0, passes)
    target_share = targets_per_step / rows_per_step
    if partition.train_inside_fold:
        fraction_read = sum(min(rows, rows_read) for rows in partition.rows_per_model) / n_training
        fraction_targets = 1 - (1 - target_share) ** passes
    elif partition.n_folds == 1:
        fraction_read = share_read
        fraction_targets = 1 - (1 - target_share) ** passes
    elif partition.bag == 1:  # one model outside fold 0 of five
        fraction_read = partition.rows_per_model[0] / n_training * share_read
        fraction_targets = partition.rows_per_model[0] / n_training * (1 - (1 - target_share) ** passes)
    else:  # folds: each row is in the share of every model but the one whose fold holds it
        fraction_read = 1 - (1 - share_read) ** (partition.bag - 1)
        fraction_targets = 1 - (1 - target_share) ** (passes * (partition.bag - 1))
    fraction_read = min(1.0, fraction_read)
    gpus = min(MULTI_GPU_INSTANCE_TYPES.get(instance_type, 1), partition.bag)
    rounds = math.ceil(partition.bag / max(1, gpus))
    training_hours = None if seconds_per_step is None else rounds * steps * seconds_per_step / 3600
    price = USD_PER_HOUR.get(instance_type)
    cost = None if training_hours is None or price is None else training_hours * price
    hits_cap = (
        hyperparameters.max_training_hours is not None
        and training_hours is not None
        and training_hours > hyperparameters.max_training_hours
    )
    warnings = list(partition.warnings)
    if fraction_read < _PARTIAL_READ_WARNING_FRACTION:
        if seconds_per_model is None:
            remedy = "raise n_passes_per_row or n_steps_per_model to read more"
        elif steps < requested:
            remedy = "a newer version of the Algorithm reads more"
        else:
            remedy = "raise n_steps_per_model to read more"
        warnings.append(f"reads {100 * fraction_read:.1f}% of the training rows; {remedy}")
    if seconds_per_step is None:
        warnings.append(
            f"{hyperparameters.base_model} on {instance_type} has no measured step time, so the plan gives no hours or "
            "cost and cannot tell whether max_training_hours or SageMaker's 120-hour limit ends the training first"
        )
    if hits_cap:
        text = (
            f"max_training_hours of {hyperparameters.max_training_hours:g} ends the training before the plan's "
            f"{training_hours:,.2f} hours"
        )
        if rounds > 1:
            text += f"; the models train {max(1, gpus)} at a time, so the last ones may not start"
        warnings.append(text)
    if steps < requested:
        warnings.append(
            f"this Algorithm version stops each model after {seconds_per_model:,.0f} seconds, about {steps:,} of its "
            f"{requested:,} steps at the plan's step time; a newer version of the Algorithm trains each model without "
            "that limit"
        )
    return TrainingPlan(
        n_rows_in_table=n_rows,
        n_feature_columns=n_features,
        n_validation_rows=n_validation_rows if has_validation else n_rows - n_training,
        n_training_rows=n_training,
        bag=partition.bag,
        row_partition_strategy=partition.strategy,
        n_rows_per_model=rows_per_model,
        rows_per_step=rows_per_step,
        targets_per_step=targets_per_step,
        rows_per_step_is_estimate=rows_per_step_is_estimate,
        steps_per_model=steps,
        n_passes_per_row=passes,
        validation_every_n_steps=validation_interval,
        fraction_of_table_read=fraction_read,
        fraction_of_rows_as_targets=min(1.0, fraction_targets),
        seconds_per_step=seconds_per_step,
        instance_type=instance_type,
        gpus=max(1, gpus),
        training_hours=training_hours,
        estimated_cost_usd=cost,
        prices_as_of=PRICES_AS_OF,
        patience=hyperparameters.patience,
        patience_steps=patience_steps,
        max_training_hours=hyperparameters.max_training_hours,
        hits_max_training_hours=hits_cap,
        warnings=warnings,
    )


#: The hyperparameters an Algorithm version that trains on the whole table accepts and an earlier one does not.
SETTINGS_ONLY_WHOLE_TABLE_ALGORITHM_READS = (
    "row_partition_strategy",
    "n_passes_per_row",
    "max_training_hours",
    "validation_every_n_steps",
    "task_column",
    "time_column",
    "context_rows",
    "query_rows",
    "tasks",
)
#: The hyperparameters every Algorithm version reads.
HYPERPARAMETERS_EVERY_VERSION_READS = frozenset(
    {"target_column", "model_name", "base_model", "task", "feature_columns", "bag", "lr", "patience", "seed"}
)
#: The hyperparameter whose presence in an Algorithm's SupportedHyperParameters marks a version that trains on the
#: whole table.
_WHOLE_TABLE_ALGORITHM_MARKER_HYPERPARAMETER = "n_passes_per_row"
#: The hyperparameter whose Categorical values in an Algorithm's SupportedHyperParameters are the bases it offers.
_BASE_MODEL_HYPERPARAMETER = "base_model"
#: The hyperparameter that lets a job continue another job's checkpoint folder; only resume() sends it, and only to
#: an Algorithm version that declares it.
RESUME_HYPERPARAMETER = "resume"
#: What an earlier Algorithm version takes: steps per model when n_steps_per_model is unset, and the seconds each
#: model may train.
LEGACY_DEFAULT_STEPS_PER_MODEL = 100
LEGACY_TRAINING_SECONDS_PER_MODEL = 900.0
#: The names an earlier Algorithm version takes the steps per model and the seconds each model may train under.
_LEGACY_STEPS_PER_MODEL_HYPERPARAMETER = "steps"
_LEGACY_TRAINING_SECONDS_PER_MODEL_HYPERPARAMETER = "budget_s"
#: The share of the seconds each model may train that an earlier version's MaxRuntimeInSeconds allows.
_LEGACY_RUNTIME_FACTOR = 1.5
#: Steps between two validation scores on an earlier version: the steps over this, at least 1.
_LEGACY_VALIDATIONS_PER_MODEL = 500


def has_whole_table_hyperparameter(description: Dict[str, Any]) -> bool:
    """Whether a DescribeAlgorithm response describes a version that trains each model on the whole table."""
    supported = (description.get("TrainingSpecification") or {}).get("SupportedHyperParameters") or []
    return any(item.get("Name") == _WHOLE_TABLE_ALGORITHM_MARKER_HYPERPARAMETER for item in supported)


def get_offered_base_models(description: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
    """The ``base_model`` values a DescribeAlgorithm response says its version accepts; None when it lists none."""
    supported = (description.get("TrainingSpecification") or {}).get("SupportedHyperParameters") or []
    for item in supported:
        if item.get("Name") == _BASE_MODEL_HYPERPARAMETER:
            categorical = (item.get("Range") or {}).get("CategoricalParameterRangeSpecification") or {}
            values = categorical.get("Values") or []
            return tuple(values) if values else None
    return None


def get_declared_default_base_model(description: Dict[str, Any]) -> Optional[str]:
    """The ``base_model`` default a DescribeAlgorithm response declares for its version; None when it declares
    none."""
    supported = (description.get("TrainingSpecification") or {}).get("SupportedHyperParameters") or []
    for item in supported:
        if item.get("Name") == _BASE_MODEL_HYPERPARAMETER:
            return item.get("DefaultValue") or None
    return None


def get_request_with_version_default_base_model(
    request: FineTuneJobRequest, set_by_caller: Collection[str], version_default_base_model: Optional[str]
) -> FineTuneJobRequest:
    """``request`` with ``base_model`` set to its Algorithm version's declared default, when the caller left
    ``base_model`` unset and that default is another base this client knows; otherwise ``request`` itself.

    The client's default is nori-1-5, but a version published before nori-1-5 offers only nori-6m and nori-100m and
    declares nori-6m as its default. A caller who never chose a base trains what their version trains by default
    instead of being refused. The instance type and runtime derived from the old base are derived again."""
    if (
        "base_model" in set_by_caller
        or version_default_base_model is None
        or version_default_base_model == request.hyperparameters.base_model
        or version_default_base_model not in get_args(BaseModelName)
    ):
        return request
    data = request.model_dump()
    data["hyperparameters"]["base_model"] = version_default_base_model
    if "instance_type" not in request.model_fields_set:
        data["instance_type"] = None
    if request._max_runtime_derived:
        data["max_runtime_s"] = None
    return FineTuneJobRequest.model_validate(data)


def refuse_base_model_not_offered(
    hyperparameters: FineTuneHyperparameters, algorithm: str, offered_base_models: Optional[Sequence[str]]
) -> None:
    """Refuse a ``base_model`` that the Algorithm's version does not offer; ``offered_base_models`` None (a version
    that lists no values, or one that could not be described) refuses nothing."""
    if offered_base_models is None or hyperparameters.base_model in offered_base_models:
        return
    raise FineTuningError(
        f"Algorithm {algorithm} does not offer base_model={hyperparameters.base_model!r}; it offers "
        f"{', '.join(repr(name) for name in offered_base_models)}. Choose one of them, or subscribe to a version of "
        "the Algorithm that offers this base."
    )


def has_resume_hyperparameter(description: Dict[str, Any]) -> bool:
    """Whether a DescribeAlgorithm response describes a version that accepts ``RESUME_HYPERPARAMETER``."""
    supported = (description.get("TrainingSpecification") or {}).get("SupportedHyperParameters") or []
    return any(item.get("Name") == RESUME_HYPERPARAMETER for item in supported)


def refuse_table_pass_settings(
    hyperparameters: FineTuneHyperparameters, algorithm: str, *, set_by_caller: Collection[str]
) -> None:
    """Refuse, on an Algorithm version that trains each model for a fixed number of steps, every setting in
    ``set_by_caller`` (the names the caller set) that only a newer version reads, and a ``patience=None`` the caller
    set."""
    named = [name for name in SETTINGS_ONLY_WHOLE_TABLE_ALGORITHM_READS if name in set_by_caller]
    if named:
        raise FineTuningError(
            f"Algorithm {algorithm} trains each model for a fixed number of steps and does not read "
            f"{', '.join(named)}; a newer version of the Algorithm supports "
            f"{'it' if len(named) == 1 else 'them'}. Subscribe to the newer version, or leave "
            f"{'it' if len(named) == 1 else 'them'} unset."
        )
    if "patience" in set_by_caller and hyperparameters.patience is None:
        raise FineTuningError(
            f"Algorithm {algorithm} trains each model for a fixed number of steps and needs a patience; "
            "patience=None turns it off only on a newer version of the Algorithm. Set patience to a number of validation scores."
        )


def refuse_version_specific_settings(
    hyperparameters: FineTuneHyperparameters, algorithm: str, *, set_by_caller: Collection[str], reason: str
) -> None:
    """Refuse, on an Algorithm that could not be described (``reason``: the error DescribeAlgorithm gave), every
    setting its versions read differently, in field order: one in ``set_by_caller`` (the names the caller set) that
    only a newer version reads, ``n_steps_per_model`` when set and a ``patience=None`` the caller set."""
    named: List[str] = []
    for name in FineTuneHyperparameters.model_fields:
        if name in SETTINGS_ONLY_WHOLE_TABLE_ALGORITHM_READS and name in set_by_caller:
            named.append(name)
        elif name == "n_steps_per_model" and hyperparameters.n_steps_per_model is not None:
            named.append(name)
        elif name == "patience" and name in set_by_caller and hyperparameters.patience is None:
            named.append("patience=None")
    if named:
        pronoun = "it" if len(named) == 1 else "them"
        raise FineTuningError(
            f"Algorithm {algorithm} could not be described ({reason}), so its version is unknown, and "
            f"{', '.join(named)} cannot be sent: the Algorithm's versions read {pronoun} differently. Let the identity "
            "that runs this code call sagemaker:DescribeAlgorithm on the Algorithm, then call again, or leave "
            f"{pronoun} unset."
        )


def get_serialized_legacy_hyperparameters(hyperparameters: FineTuneHyperparameters) -> Dict[str, str]:
    """The HyperParameters of an Algorithm version that trains each model for a fixed number of steps."""
    job_hyperparameters = serialize_hyperparameters(hyperparameters)
    for name in (*SETTINGS_ONLY_WHOLE_TABLE_ALGORITHM_READS, "n_steps_per_model"):
        job_hyperparameters.pop(name, None)
    steps = hyperparameters.n_steps_per_model or LEGACY_DEFAULT_STEPS_PER_MODEL
    job_hyperparameters[_LEGACY_STEPS_PER_MODEL_HYPERPARAMETER] = str(steps)
    job_hyperparameters[_LEGACY_TRAINING_SECONDS_PER_MODEL_HYPERPARAMETER] = str(LEGACY_TRAINING_SECONDS_PER_MODEL)
    if hyperparameters.patience is not None:
        # That version counts patience in steps, so the scores go as the steps between that many of its scores.
        job_hyperparameters["patience"] = str(hyperparameters.patience * max(1, steps // _LEGACY_VALIDATIONS_PER_MODEL))
    return job_hyperparameters


def get_hyperparameters_from_strings(job_hyperparameters: Dict[str, str]) -> Dict[str, str]:
    """The request's hyperparameters, as strings, from a job's HyperParameters: those of an Algorithm version that
    trains each model for a fixed number of steps read back (``steps`` as ``n_steps_per_model``, its patience in
    validation scores), and every name :class:`FineTuneHyperparameters` does not have dropped. A value that is not an
    integer is kept as it is, for the request's validation to refuse."""
    hyperparameters = dict(job_hyperparameters)
    if (
        _LEGACY_STEPS_PER_MODEL_HYPERPARAMETER in hyperparameters
        and _LEGACY_TRAINING_SECONDS_PER_MODEL_HYPERPARAMETER in hyperparameters
    ):
        hyperparameters["n_steps_per_model"] = hyperparameters[_LEGACY_STEPS_PER_MODEL_HYPERPARAMETER]
        patience = hyperparameters.get("patience")
        try:
            steps = int(hyperparameters["n_steps_per_model"])
            if patience is not None and patience.strip().lower() != _NONE_HYPERPARAMETER:
                hyperparameters["patience"] = str(int(patience) // max(1, steps // _LEGACY_VALIDATIONS_PER_MODEL))
        except ValueError:
            pass
    return {name: value for name, value in hyperparameters.items() if name in FineTuneHyperparameters.model_fields}


def get_hyperparameters_every_version_reads(hyperparameters: FineTuneHyperparameters) -> Dict[str, str]:
    """The HyperParameters of an Algorithm that could not be described: the serialized hyperparameters every version
    reads (``HYPERPARAMETERS_EVERY_VERSION_READS``)."""
    return {
        name: value
        for name, value in serialize_hyperparameters(hyperparameters).items()
        if name in HYPERPARAMETERS_EVERY_VERSION_READS
    }


def get_legacy_max_runtime_s(request: FineTuneJobRequest) -> int:
    """The MaxRuntimeInSeconds of a job on an Algorithm version that trains each model for at most
    LEGACY_TRAINING_SECONDS_PER_MODEL: every round of models (one per GPU) at that budget with a margin, plus the
    time to read the table and package the model; the caller's own ``max_runtime_s`` when it gave one."""
    if not request._max_runtime_derived and request.max_runtime_s is not None:
        return request.max_runtime_s
    gpus = MULTI_GPU_INSTANCE_TYPES.get(request.instance_type or "", 1)
    rounds = math.ceil(request.hyperparameters.bag / gpus)
    return min(
        SAGEMAKER_MAX_RUNTIME_S,
        math.ceil(_LEGACY_RUNTIME_FACTOR * rounds * LEGACY_TRAINING_SECONDS_PER_MODEL) + _RUNTIME_MARGIN_S,
    )


def get_legacy_hyperparameters(hyperparameters: FineTuneHyperparameters) -> FineTuneHyperparameters:
    """The hyperparameters an earlier Algorithm version trains with, in the plan's terms: a fixed number of steps
    on every row outside each model's fold."""
    steps = hyperparameters.n_steps_per_model or LEGACY_DEFAULT_STEPS_PER_MODEL
    return hyperparameters.model_copy(
        update={
            "n_steps_per_model": steps,
            "row_partition_strategy": "folds",
            "max_training_hours": None,
            "validation_every_n_steps": max(1, steps // _LEGACY_VALIDATIONS_PER_MODEL),
        }
    )


def _get_channel_footers(s3: Any, uri: str, argument: str) -> List[Tuple[str, ParquetFooter]]:
    """Each table file of a channel with its footer, in key order, ten files at a time."""
    bucket, _ = parse_s3_uri(uri, name=argument)
    keys = get_channel_table_keys(s3, uri, argument)
    pool = ThreadPoolExecutor(max_workers=max(1, min(_FOOTER_READ_THREADS, len(keys))))
    try:
        footers = list(pool.map(functools.partial(get_parquet_footer, s3, bucket), keys))
    finally:
        pool.shutdown(cancel_futures=True)
    return [(f"s3://{bucket}/{key}", footer) for key, footer in zip(keys, footers)]


def _check_same_columns(
    files: List[Tuple[str, ParquetFooter]], columns: List[Any], reference: str, argument: str, rule: str
) -> None:
    """Refuse a file whose columns are not ``columns`` (those of the file ``reference``), in any order, as the job
    does: it selects columns by name."""
    for uri, footer in files:
        missing = [column for column in columns if column not in footer.columns]
        extra = [column for column in footer.columns if column not in columns]
        if missing or extra:
            raise TableRefused(
                f"{argument}: {uri} has other columns than {reference}: missing {missing[:_MAX_NAMED_COLUMNS]}, "
                f"extra {extra[:_MAX_NAMED_COLUMNS]}; {rule}"
            )


def get_feature_columns(
    columns: List[str], table: Union[TaskTable, FineTuneHyperparameters], argument: str
) -> List[str]:
    """The feature columns the job reads from a table of ``columns`` (by default every column but the target, task and
    time columns), refusing a target or feature column it does not have. ``table`` is a task table, or a one-table
    request's hyperparameters, which hold the same settings."""
    if table.target_column not in columns:
        raise TableRefused(
            f"target_column {table.target_column!r} is not a column of {argument}; its columns are "
            f"{columns[:_MAX_NAMED_COLUMNS]}{' ...' if len(columns) > _MAX_NAMED_COLUMNS else ''}"
        )
    if table.feature_columns is None:
        roles = (table.target_column, table.task_column, table.time_column)
        features = [column for column in columns if column not in roles]
    else:
        missing = [column for column in table.feature_columns if column not in columns]
        if missing:
            raise TableRefused(f"feature_columns not in {argument}: {missing}")
        features = list(table.feature_columns)
    if not features:
        raise TableRefused(f"{argument} has no feature columns besides target_column")
    return features


def is_file_of_table(relative: str, table_path: str) -> bool:
    """Whether the channel file at ``relative`` (its path inside the channel) belongs to the task table at
    ``table_path``, as the job decides it: every file for an empty path, else the file itself or any file under the
    folder."""
    folder = table_path.strip("/")
    return not folder or relative == folder or relative.startswith(folder + "/")


def check_task_table_footers(s3: Any, request: FineTuneJobRequest) -> None:
    """Check a job of tasks' channels from their footers as the job will: every file belongs to exactly one task
    table, a table's files have the same columns, each table has its target, feature and time columns and its task
    column (in the files, or as ``<task_column>=<value>`` folders above every file inside the channel), and a table's
    validation files have its training files' columns. Raises :class:`TableRefused`, naming the table."""
    tables = get_task_tables(request.hyperparameters)
    columns_by_table: List[List[Any]] = []
    for argument, uri in (("training_data", request.training_data), ("validation_data", request.validation_data)):
        if uri is None:
            continue
        _, key = parse_s3_uri(uri, name=argument)
        # A folder's keys land under the channel as they sit under the folder; one object lands by its file name.
        folder = (f"{key}/" if key else "") if uri.endswith("/") else None
        files_by_table: List[List[Tuple[str, str, ParquetFooter]]] = [[] for _ in tables]
        for file_uri, footer in _get_channel_footers(s3, uri, argument):
            object_key = file_uri.split("/", 3)[3]
            relative = object_key[len(folder) :] if folder is not None else object_key.rsplit("/", 1)[-1]
            holders = [index for index, table in enumerate(tables) if is_file_of_table(relative, table.path)]
            if len(holders) != 1:
                held = "no task table" if not holders else f"the task tables {[tables[i].path for i in holders]}"
                raise TableRefused(f"{argument}: {file_uri} belongs to {held}; every file belongs to exactly one")
            files_by_table[holders[0]].append((file_uri, relative, footer))
        for index, (table, files) in enumerate(zip(tables, files_by_table)):
            where = f"{argument} table {table.path!r}" if table.path.strip("/") else argument
            if not files:
                raise TableRefused(f"{where} has no .parquet or .pq file")
            footers = [(file_uri, footer) for file_uri, _, footer in files]
            if argument == "training_data":
                columns = files[0][2].columns
                _check_same_columns(
                    footers, columns, files[0][0], where, "every file of a task table must have the same columns"
                )
                columns_by_table.append(columns)
            else:
                _check_same_columns(
                    footers,
                    columns_by_table[index],
                    f"its training table {table.path!r}",
                    where,
                    "a table's validation files must have its training files' columns",
                )
                continue
            get_feature_columns(columns, table, where)
            if table.time_column is not None and table.time_column not in columns:
                raise TableRefused(f"time_column {table.time_column!r} is not a column of {where}")
            if table.task_column is not None and table.task_column not in columns:
                folder_prefix = f"{table.task_column}="
                lacking = [
                    file_uri
                    for file_uri, relative, _ in files
                    if not any(part.startswith(folder_prefix) for part in relative.split("/")[:-1])
                ]
                if lacking:
                    raise TableRefused(
                        f"task_column {table.task_column!r} is neither a column of {where} nor a "
                        f"{folder_prefix}<value> folder above {lacking[0]} inside the channel"
                    )


#: The hyperparameters that make a job of tasks; an Algorithm version that does not list them predates it.
_JOB_OF_TASKS_HYPERPARAMETERS = ("tasks", "task_column", "time_column", "context_rows", "query_rows")


def get_supported_hyperparameter_names(description: Dict[str, Any]) -> Optional[frozenset]:
    """The hyperparameter names a DescribeAlgorithm response says its version reads; None when it lists none."""
    supported = (description.get("TrainingSpecification") or {}).get("SupportedHyperParameters") or []
    names = frozenset(item.get("Name") for item in supported if item.get("Name"))
    return names or None


def refuse_task_settings_the_version_lacks(
    hyperparameters: FineTuneHyperparameters, algorithm: str, supported_names: Optional[frozenset]
) -> None:
    """Refuse the job-of-tasks settings a described Algorithm version does not read (it predates them), naming each;
    ``supported_names`` None (a version that lists none, or one that could not be described) refuses nothing."""
    if supported_names is None:
        return
    lacking = [
        name
        for name in _JOB_OF_TASKS_HYPERPARAMETERS
        if getattr(hyperparameters, name) is not None and name not in supported_names
    ]
    if lacking:
        raise FineTuningError(
            f"Algorithm {algorithm} does not read {', '.join(lacking)}: its version predates jobs of tasks. Subscribe "
            "to a newer version of the Algorithm, or fine-tune one table with target_column."
        )


def get_training_plan_from_footers(
    s3: Any, request: FineTuneJobRequest, *, trains_on_whole_table: Optional[bool] = True
) -> Optional[TrainingPlan]:
    """The :class:`~synthefy.TrainingPlan` of a request, from the footers of its channels' Parquet files alone; None
    for a job of tasks (:func:`~synthefy.nori_data_models.is_job_of_tasks`), whose job works out its plan once it has
    read the tables, after :func:`check_task_table_footers`.

    Raises :class:`TableRefused` for a channel with no Parquet file or with a file of another extension, a file that
    cannot be read as Parquet or whose pandas metadata cannot be applied, a target or feature column the training
    table lacks, a file that names a column twice, a file whose columns differ from the first's, and a table too
    small for one model's ``MIN_TRAIN_ROWS`` rows; ValueError for a footer this reader does not read. On an
    Algorithm version that trains each model for a fixed number of steps (``trains_on_whole_table=False``), the plan
    and the minimum rows are that version's: it needs ``MIN_TRAIN_ROWS`` training rows in the table, not in each
    model's share. For an Algorithm that could not be described (``trains_on_whole_table=None``), the plan is the
    newest version's, with a warning that says so.
    """
    if is_job_of_tasks(request.hyperparameters):
        check_task_table_footers(s3, request)
        return None
    job_hyperparameters = (
        get_legacy_hyperparameters(request.hyperparameters)
        if trains_on_whole_table is False
        else request.hyperparameters
    )
    train = _get_channel_footers(s3, request.training_data, "training_data")
    columns = train[0][1].columns
    _check_same_columns(train, columns, train[0][0], "training_data", "every table file must have the same columns")
    features = get_feature_columns(columns, job_hyperparameters, "training_data")
    n_validation_rows: Optional[int] = None
    if request.validation_data is not None:
        validation = _get_channel_footers(s3, request.validation_data, "validation_data")
        _check_same_columns(
            validation,
            columns,
            train[0][0],
            "validation_data",
            "the validation table must have the training table's columns",
        )
        n_validation_rows = sum(footer.num_rows for _, footer in validation)
    n_rows = sum(footer.num_rows for _, footer in train)
    n_features = len(features)
    remedy = "" if n_validation_rows is not None else ", or give validation_data so no rows are held out"
    if trains_on_whole_table is not False:
        partition = get_table_partition(n_rows, n_features, job_hyperparameters, n_validation_rows is not None)
        if min(partition.rows_per_model) < MIN_TRAIN_ROWS:
            raise TableRefused(
                f"training_data has {n_rows:,} rows; the smallest model's share is {min(partition.rows_per_model):,} "
                f"rows, fewer than the {MIN_TRAIN_ROWS:,} training rows each model needs, so the job would refuse "
                f"the table; add rows{remedy}"
            )
    else:
        n_training = n_rows if n_validation_rows is not None else n_rows - count_report_holdout_rows(n_rows)
        if n_training < MIN_TRAIN_ROWS:
            raise TableRefused(
                f"training_data has {n_rows:,} rows, {n_training:,} after the report hold-out, fewer than the "
                f"{MIN_TRAIN_ROWS:,} training rows the job needs, so it would refuse the table; add rows{remedy}"
            )
    largest_row_group = max((rows for _, footer in train for rows in footer.row_group_rows), default=0)
    plan = get_training_plan_from_counts(
        job_hyperparameters,
        n_rows=n_rows,
        n_validation_rows=n_validation_rows,
        n_features=n_features,
        largest_row_group=largest_row_group,
        instance_type=request.instance_type or "",
        widths=get_estimated_member_widths(n_features, job_hyperparameters.base_model),
        seconds_per_model=LEGACY_TRAINING_SECONDS_PER_MODEL if trains_on_whole_table is False else None,
    )
    if trains_on_whole_table is None:
        unknown = (
            f"Algorithm {request.algorithm} could not be described, so this plan is the newest version's; a version "
            f"that trains each model for a fixed number of steps trains each for {LEGACY_DEFAULT_STEPS_PER_MODEL} "
            f"steps or {LEGACY_TRAINING_SECONDS_PER_MODEL:,.0f} seconds, whichever ends first"
        )
        plan = plan.model_copy(update={"warnings": [*plan.warnings, unknown]})
    return plan
