"""Read the ``training_report.json`` a fine-tuning job wrote into its ``model.tar.gz``, and nothing else from it.

The artifact is streamed from S3 once and its members walked in order: the report is read, and every other member
is decompressed past, never extracted, written or loaded, so reading a report runs nothing the artifact holds.
Every refusal is a :class:`~synthefy.errors.FineTuningError` naming the job.
"""

import tarfile
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Optional

from pydantic import ValidationError
from synthefy._aws import is_missing_s3_object, parse_s3_uri
from synthefy.errors import FineTuningError
from synthefy.nori_data_models import REPORT_FILENAME, FineTuneJob, TrainingReport

#: The largest training report read from an artifact; a report is a few KB, so more means a wrong artifact.
MAX_REPORT_BYTES = 64 << 20


def _refuse(job: FineTuneJob, message: str) -> FineTuningError:
    """The job's FineTuningError: every refusal here names the job and carries its status and failure reason."""
    return FineTuningError(message, job_name=job.job_name, status=job.status, failure_reason=job.failure_reason)


def normalize_member_name(name: str) -> str:
    """``./x`` and ``x`` are one member: SageMaker's tar writes the first."""
    return str(PurePosixPath(name))


def read_report_member(body: BinaryIO, *, uri: str, job: FineTuneJob) -> Optional[bytes]:
    """The report member's bytes, read from the artifact as a stream; None when the artifact holds no report."""
    limit = MAX_REPORT_BYTES
    with tarfile.open(fileobj=body, mode="r|gz") as archive:
        for member in archive:
            if normalize_member_name(member.name) != REPORT_FILENAME or not member.isfile():
                continue
            if member.size > limit:
                raise _refuse(
                    job,
                    f"{REPORT_FILENAME} in {uri} is {member.size:,} bytes, above the {limit:,} this client reads; "
                    "a Nori training report is a few KB, so this artifact did not come from the Nori fine-tuning "
                    "job",
                )
            handle = archive.extractfile(member)
            return handle.read() if handle is not None else None
    return None


def format_first_validation_error(exc: ValidationError) -> str:
    """The first problem pydantic found, as ``<field path>: <message>``, for a one-line refusal."""
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "the document"
    return f"{location}: {first.get('msg', 'invalid')}"


def read_training_report(s3: Any, *, uri: str, job: FineTuneJob) -> TrainingReport:
    """The report in the artifact at ``uri``, through the given S3 client; one GetObject, streamed."""
    bucket, key = parse_s3_uri(uri, name="ModelArtifacts.S3ModelArtifacts")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    except Exception as exc:
        if not is_missing_s3_object(exc):
            raise
        raise _refuse(job, f"the model artifact {uri} of training job {job.job_name} is gone from S3") from exc
    try:
        raw = read_report_member(body, uri=uri, job=job)
    finally:
        body.close()
    if raw is None:
        raise _refuse(
            job,
            f"the model artifact {uri} of training job {job.job_name} holds no {REPORT_FILENAME}, so the Nori "
            "fine-tuning job did not write it",
        )
    try:
        return TrainingReport.model_validate_json(raw)
    except ValidationError as exc:
        raise _refuse(
            job,
            f"{REPORT_FILENAME} in the model artifact {uri} of training job {job.job_name} is not a training "
            f"report this client reads ({format_first_validation_error(exc)}): the artifact did not come from "
            "the Nori fine-tuning job, or came from a newer one, which a newer synthefy reads",
        ) from exc
