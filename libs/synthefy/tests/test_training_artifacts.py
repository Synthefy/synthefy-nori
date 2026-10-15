"""Reading ``training_report.json`` out of a fine-tuning job's artifact (``synthefy._training_artifacts``), against
the S3 fake in ``_fine_tune_fakes``."""

import re
import tarfile
from typing import Any, List

import pytest
from pydantic import ValidationError
from synthefy import _training_artifacts as artifacts_module
from synthefy._training_artifacts import read_training_report
from synthefy.errors import FineTuningError
from synthefy.nori_data_models import FineTuneJob, TrainingReport

from ._fine_tune_fakes import (
    ALGORITHM,
    ARTIFACT,
    ARTIFACT_KEY,
    JOB,
    JOB_ARN,
    MODEL,
    NOW,
    FakeS3,
    build_artifact,
    build_curve,
    build_report,
    store_artifact,
)

_JOB = FineTuneJob(
    job_name=JOB,
    job_arn=JOB_ARN,
    status="Completed",
    created_at=NOW,
    artifact_uri=ARTIFACT,
    algorithm=ALGORITHM,
    model_name=MODEL,
)


def _create_s3_with_artifact(data: bytes) -> FakeS3:
    s3 = FakeS3()
    store_artifact(s3, data)
    return s3


def test_report_reads_only_its_member_and_never_extracts_another(monkeypatch):
    s3 = _create_s3_with_artifact(build_artifact(build_report()))
    extracted: List[str] = []
    original = tarfile.TarFile.extractfile

    def spy(archive: tarfile.TarFile, member: Any) -> Any:
        extracted.append(member.name)
        return original(archive, member)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the artifact must never be extracted to disk")

    monkeypatch.setattr(tarfile.TarFile, "extractfile", spy)
    monkeypatch.setattr(tarfile.TarFile, "extract", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)

    report = read_training_report(s3, uri=ARTIFACT, job=_JOB)

    assert extracted == ["./training_report.json"]
    assert report == TrainingReport.model_validate(build_report())
    assert s3.calls == [("get_object", ARTIFACT_KEY)]


@pytest.mark.parametrize("member_prefix", ["./", ""], ids=["dot-slash-members", "bare-members"])
def test_report_parses_a_cut_short_report_whatever_the_member_prefix(member_prefix):
    cut_short_report = build_report(
        cut_short=True,
        val_mse_after=None,
        validation_source="channel",
        child_curves=[build_curve(0, interrupted=True)],
    )
    s3 = _create_s3_with_artifact(build_artifact(cut_short_report, member_prefix=member_prefix))
    report = read_training_report(s3, uri=ARTIFACT, job=_JOB)
    assert (report.cut_short, report.val_mse_after, report.child_curves[0].interrupted) == (True, None, True)


def test_report_refuses_a_report_above_the_size_limit(monkeypatch):
    monkeypatch.setattr(artifacts_module, "MAX_REPORT_BYTES", 100)
    s3 = _create_s3_with_artifact(build_artifact(build_report()))
    with pytest.raises(FineTuningError, match="above the 100 this client reads"):
        read_training_report(s3, uri=ARTIFACT, job=_JOB)


@pytest.mark.parametrize(
    "body, problem",
    [
        (
            {key: value for key, value in build_report().items() if key != "model_version"},
            "model_version: Field required",
        ),
        (b"not json", "the document: Invalid JSON"),
    ],
    ids=["missing-field", "not-json"],
)
def test_report_refuses_a_report_that_does_not_parse_naming_the_job_and_the_first_problem(body, problem):
    s3 = _create_s3_with_artifact(build_artifact(body))
    with pytest.raises(
        FineTuningError, match=re.escape(f"is not a training report this client reads ({problem}")
    ) as raised:
        read_training_report(s3, uri=ARTIFACT, job=_JOB)
    assert (raised.value.job_name, raised.value.status) == (JOB, "Completed")
    assert isinstance(raised.value.__cause__, ValidationError)


def test_report_refuses_an_artifact_without_a_report():
    s3 = _create_s3_with_artifact(build_artifact(None))
    with pytest.raises(FineTuningError, match="holds no training_report.json"):
        read_training_report(s3, uri=ARTIFACT, job=_JOB)


def test_report_refuses_an_artifact_gone_from_s3():
    with pytest.raises(FineTuningError, match="is gone from S3"):
        read_training_report(FakeS3(), uri=ARTIFACT, job=_JOB)
