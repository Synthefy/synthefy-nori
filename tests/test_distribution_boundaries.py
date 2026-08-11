"""Unit tests for the two-wheel artifact ownership gate."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "validate_distribution_boundaries.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("distribution_boundary_validator", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def _write_wheel(
    path: Path,
    *,
    distribution: str,
    version: str,
    namespace: str,
    requirements: tuple[str, ...] = (),
    extras: tuple[str, ...] = (),
    init_payload: bytes = b"__all__ = []\n",
    extra_runtime_files: tuple[str, ...] = (),
    tamper_after_record: str | None = None,
) -> Path:
    dist_info = f"{distribution.replace('-', '_')}-{version}.dist-info"
    metadata_lines = [
        "Metadata-Version: 2.4",
        f"Name: {distribution}",
        f"Version: {version}",
        *(f"Requires-Dist: {value}" for value in requirements),
        *(f"Provides-Extra: {value}" for value in extras),
        "",
        "",
    ]
    members = {
        f"{namespace}/__init__.py": init_payload,
        f"{dist_info}/METADATA": "\n".join(metadata_lines).encode(),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
    }
    for name in extra_runtime_files:
        members[name] = b"VALUE = 1\n"

    record_path = f"{dist_info}/RECORD"
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    for name, payload in sorted(members.items()):
        writer.writerow((name, _digest(payload), len(payload)))
    writer.writerow((record_path, "", ""))
    members[record_path] = stream.getvalue().encode()
    if tamper_after_record is not None:
        members[tamper_after_record] += b"# changed after RECORD was written\n"

    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)
    return path


def _valid_artifacts(tmp_path: Path):
    client_requirements = (
        "httpx<0.28.0,>=0.24.0",
        'boto3<2.0.0,>=1.34.0; extra == "aws"',
    )
    nori_requirements = (
        "numpy>=2.0",
        "synthefy<8,>=7",
        'synthefy[text]<8,>=7; extra == "text"',
    )
    values = {}
    for build in ("direct", "rebuilt"):
        values[f"client_{build}"] = _write_wheel(
            tmp_path / f"synthefy-{build}.whl",
            distribution="synthefy",
            version="7.0.0",
            namespace="synthefy",
            requirements=client_requirements,
            extras=("aws", "forecasting", "text"),
            extra_runtime_files=("synthefy/client.py",),
        )
        values[f"nori_{build}"] = _write_wheel(
            tmp_path / f"synthefy_nori-{build}.whl",
            distribution="synthefy-nori",
            version="0.16.0",
            namespace="synthefy_nori",
            requirements=nori_requirements,
            extras=("text",),
            extra_runtime_files=("synthefy_nori/api.py",),
        )
    return values


def test_valid_artifacts_have_disjoint_runtime_ownership(tmp_path):
    validator = _load_validator()
    artifacts = _valid_artifacts(tmp_path)

    client, nori = validator.validate_artifacts(**artifacts)

    assert client.runtime_files == {"synthefy/__init__.py", "synthefy/client.py"}
    assert nori.runtime_files == {"synthefy_nori/__init__.py", "synthefy_nori/api.py"}


def test_record_hash_mismatch_is_rejected(tmp_path):
    validator = _load_validator()
    wheel = _write_wheel(
        tmp_path / "synthefy-tampered.whl",
        distribution="synthefy",
        version="7.0.0",
        namespace="synthefy",
        extras=("aws", "forecasting", "text"),
        tamper_after_record="synthefy/__init__.py",
    )

    with pytest.raises(validator.BoundaryError, match="hash mismatch"):
        validator.inspect_wheel(wheel, distribution="synthefy", namespace="synthefy")


def test_client_reverse_dependency_is_rejected(tmp_path):
    validator = _load_validator()
    artifacts = _valid_artifacts(tmp_path)
    artifacts["client_direct"] = _write_wheel(
        tmp_path / "synthefy-bad-direct.whl",
        distribution="synthefy",
        version="7.0.0",
        namespace="synthefy",
        requirements=("synthefy-nori>=0.16",),
        extras=("aws", "forecasting", "text"),
    )
    artifacts["client_rebuilt"] = _write_wheel(
        tmp_path / "synthefy-bad-rebuilt.whl",
        distribution="synthefy",
        version="7.0.0",
        namespace="synthefy",
        requirements=("synthefy-nori>=0.16",),
        extras=("aws", "forecasting", "text"),
    )

    with pytest.raises(validator.BoundaryError, match="must not depend"):
        validator.validate_artifacts(**artifacts)


def test_sdist_rebuild_with_different_runtime_files_is_rejected(tmp_path):
    validator = _load_validator()
    artifacts = _valid_artifacts(tmp_path)
    artifacts["client_rebuilt"] = _write_wheel(
        tmp_path / "synthefy-drifted-rebuilt.whl",
        distribution="synthefy",
        version="7.0.0",
        namespace="synthefy",
        requirements=(
            "httpx<0.28.0,>=0.24.0",
            'boto3<2.0.0,>=1.34.0; extra == "aws"',
        ),
        extras=("aws", "forecasting", "text"),
        extra_runtime_files=("synthefy/client.py", "synthefy/unexpected.py"),
    )

    with pytest.raises(validator.BoundaryError, match="sdist-rebuilt wheel differ"):
        validator.validate_artifacts(**artifacts)


def test_sdist_rebuild_with_same_paths_but_different_bytes_is_rejected(tmp_path):
    validator = _load_validator()
    artifacts = _valid_artifacts(tmp_path)
    artifacts["client_rebuilt"] = _write_wheel(
        tmp_path / "synthefy-content-drifted-rebuilt.whl",
        distribution="synthefy",
        version="7.0.0",
        namespace="synthefy",
        requirements=(
            "httpx<0.28.0,>=0.24.0",
            'boto3<2.0.0,>=1.34.0; extra == "aws"',
        ),
        extras=("aws", "forecasting", "text"),
        init_payload=b"__all__ = ['changed']\n",
        extra_runtime_files=("synthefy/client.py",),
    )

    with pytest.raises(
        validator.BoundaryError, match="member_fingerprints"
    ):
        validator.validate_artifacts(**artifacts)
