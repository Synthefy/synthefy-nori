"""Read what a fine-tuning job wrote to S3: the ``training_report.json`` in its ``model.tar.gz``, and what its
checkpoint folder holds.

The artifact is streamed from S3 once and its members walked in order: the report is read, and every other member
is decompressed past, never extracted, written or loaded, so reading a report runs nothing the artifact holds.
Every refusal is a :class:`~synthefy.errors.FineTuningError` naming the job. A checkpoint folder is read from one
listing and the records it lists, each read up to a byte bound; no model file is read.
"""

import json
import math
import re
import tarfile
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Dict, List, NamedTuple, Optional, Tuple

from pydantic import ValidationError
from synthefy._aws import is_missing_s3_object, list_s3_keys, parse_s3_uri
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


# ------------------------------------------------------------------ checkpoint folders
#: The only names a checkpoint folder holds, the job's own pattern: model k's finished files (``child-<k>.pt`` and its
#: record ``child-<k>.json``), its state after step s (``child-<k>.step-<s>.pt``, ``.json``) and its best weights as of
#: that state (``child-<k>.step-<s>.best.pt``), k and s without leading zeros.
CHECKPOINT_FILE_NAME_PATTERN = re.compile(
    r"child-(0|[1-9][0-9]*)(?:\.step-([1-9][0-9]*)(?:\.best(?=\.pt))?)?\.(pt|json)"
)
#: The keys one ListObjectsV2 call returns at most: a folder with more is reported as cut off.
MAX_CHECKPOINT_FOLDER_KEYS = 1000
#: The bytes read of one record at most; a record the job writes is a few hundred KB, so a larger one is a file the
#: job did not write.
_MAX_CHECKPOINT_RECORD_BYTES = 16 << 20


class CheckpointFolderContents(NamedTuple):
    """What a checkpoint folder holds, from one listing and the records it lists."""

    #: The ``checkpoint_uri`` as the caller gave it.
    uri: str
    #: Every key listed under the folder, the folder's own key excluded.
    keys: Tuple[str, ...]
    #: The models whose finished files are both there, by index, in index order.
    finished_children: Tuple[int, ...]
    #: For each other model with a saved state, the highest step whose state and record are both there.
    highest_step_by_child: Dict[int, int]
    #: The key of the newest record that parses, as the job ranks records (:func:`_get_checkpoint_record_newness`, then
    #: the key); the highest key when no record parses; None when the folder holds no record. A record over
    #: ``_MAX_CHECKPOINT_RECORD_BYTES`` is not ranked: it counts as a file the job does not write.
    newest_record_key: Optional[str]
    #: The training job that record names; None when no record parses or the newest names none.
    newest_record_job_name: Optional[str]
    #: Keys whose name the job does not write, in listing order.
    foreign_keys: Tuple[str, ...]
    #: Whether the folder holds more keys than one listing returns.
    truncated: bool


def _read_checkpoint_record_bytes(s3: Any, bucket: str, key: str) -> Optional[bytes]:
    """The bytes of the record at ``key``; None when it holds more than ``_MAX_CHECKPOINT_RECORD_BYTES``. A GetObject
    that S3 refuses or that finds no object raises the AWS error."""
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        raw = body.read(_MAX_CHECKPOINT_RECORD_BYTES + 1)
    finally:
        body.close()
    return None if len(raw) > _MAX_CHECKPOINT_RECORD_BYTES else raw


def _parse_checkpoint_record(raw: bytes) -> Optional[Dict[str, Any]]:
    """The JSON object a record holds; None when it is not one whole JSON object, as a write cut short leaves it."""
    try:
        record = json.loads(raw)
    except ValueError:  # bytes that are not UTF-8, or not JSON
        return None
    return record if isinstance(record, dict) else None


def _get_checkpoint_record_newness(record: Dict[str, Any]) -> Tuple[bool, float]:
    """How recently a record was written, ranked as the job ranks a folder's records: ``(True, its saved_at)`` in
    epoch seconds, or ``(False, 0.0)`` for a record with no ``saved_at``, which ranks below every record with one. A
    time S3 sets is not used: it is when SageMaker uploaded the file."""
    saved_at = record.get("saved_at")
    if isinstance(saved_at, (int, float)) and not isinstance(saved_at, bool) and math.isfinite(saved_at):
        return True, float(saved_at)
    return False, 0.0


def read_checkpoint_folder(s3: Any, uri: str) -> CheckpointFolderContents:
    """What the checkpoint folder at ``uri`` holds: one ListObjectsV2 call of up to ``MAX_CHECKPOINT_FOLDER_KEYS``
    keys, then one GetObject of each record listed, of up to ``_MAX_CHECKPOINT_RECORD_BYTES`` bytes, to find the
    newest record that parses and the job it names. A record larger than that is a file the job does not write. A
    listing, or a GetObject that S3 refuses or that finds no object, raises the AWS error."""
    bucket, folder_key = parse_s3_uri(uri, name="checkpoint_uri")
    # The folder's keys, and only its keys: without the '/' the listing would also hold a sibling 'churn-old/'.
    prefix = f"{folder_key}/" if folder_key else ""
    listed, truncated = list_s3_keys(s3, bucket=bucket, prefix=prefix, max_keys=MAX_CHECKPOINT_FOLDER_KEYS)
    keys = [key for key in listed if key != prefix]
    names = {key[len(prefix) :] for key in keys}
    foreign: List[str] = []
    finished_children: List[int] = []
    highest_step_by_child: Dict[int, int] = {}
    records: List[Tuple[Tuple[bool, float], str, Dict[str, Any]]] = []
    unparsed_record_keys: List[str] = []
    for key in keys:
        name = key[len(prefix) :]
        match = CHECKPOINT_FILE_NAME_PATTERN.fullmatch(name)
        if match is None:
            foreign.append(key)
            continue
        if match[3] != "json":
            continue
        raw = _read_checkpoint_record_bytes(s3, bucket, key)
        if raw is None:
            foreign.append(key)
            continue
        record = _parse_checkpoint_record(raw)
        if record is None:
            unparsed_record_keys.append(key)
        else:
            records.append((_get_checkpoint_record_newness(record), key, record))
        child = int(match[1])
        if match[2] is None:
            if f"child-{child}.pt" in names:
                finished_children.append(child)
        elif f"child-{child}.step-{match[2]}.pt" in names:
            highest_step_by_child[child] = max(int(match[2]), highest_step_by_child.get(child, 0))
    for child in finished_children:
        highest_step_by_child.pop(child, None)
    newest_record_key: Optional[str] = None
    newest_record_job_name: Optional[str] = None
    if records:
        # Of the records that parse, the latest saved_at, then the highest name, as the job ranks its folder's records.
        _, newest_record_key, newest_record = max(records, key=lambda item: (item[0], item[1]))
        recorded_job_name = newest_record.get("training_job_name")
        newest_record_job_name = recorded_job_name if isinstance(recorded_job_name, str) and recorded_job_name else None
    elif unparsed_record_keys:
        newest_record_key = max(unparsed_record_keys)
    return CheckpointFolderContents(
        uri=uri,
        keys=tuple(keys),
        finished_children=tuple(sorted(finished_children)),
        highest_step_by_child=dict(sorted(highest_step_by_child.items())),
        newest_record_key=newest_record_key,
        newest_record_job_name=newest_record_job_name,
        foreign_keys=tuple(foreign),
        truncated=truncated,
    )


def describe_checkpoint_folder(contents: CheckpointFolderContents) -> str:
    """What a checkpoint folder holds, as one clause the refusals share: its finished models, each other model's
    highest saved step, the job that wrote its newest record, and the files the job does not write."""
    count = len(contents.finished_children)
    parts = [f"{count} finished model{'' if count == 1 else 's'}"]
    parts += [
        f"the saved state of model {child} after step {step}" for child, step in contents.highest_step_by_child.items()
    ]
    if contents.foreign_keys:
        count = len(contents.foreign_keys)
        parts.append(f"{count} file{'' if count == 1 else 's'} the job does not write")
    text = f"{contents.uri} holds {', '.join(parts)}"
    if contents.truncated:
        text += f" among its first {MAX_CHECKPOINT_FOLDER_KEYS:,} files"
    if contents.newest_record_key is None:
        return text
    if contents.newest_record_job_name is None:
        return (
            f"{text}; the training job that wrote its newest record is unknown: no record can be parsed, or the newest "
            "names none"
        )
    return f"{text}; its newest record was written by training job {contents.newest_record_job_name!r}"
