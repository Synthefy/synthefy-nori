#!/usr/bin/env python3
"""Validate ownership and metadata boundaries between the two shipped wheels.

The consolidated repository builds two independent distributions.  ``synthefy``
owns the lightweight ``synthefy`` namespace; ``synthefy-nori`` owns the heavy
``synthefy_nori`` namespace and may depend on the lightweight distribution.  This
helper validates built artifacts rather than trusting the source-tree layout.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import NamedTuple
from zipfile import ZipFile


class BoundaryError(ValueError):
    """A built wheel violates the two-distribution ownership contract."""


class WheelInfo(NamedTuple):
    path: Path
    distribution: str
    version: str
    namespace: str
    runtime_files: frozenset[str]
    member_fingerprints: tuple[tuple[str, str, int], ...]
    requirements: tuple[str, ...]
    extras: frozenset[str]


def _canonicalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _requirement_name(value: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", value)
    if match is None:
        raise BoundaryError(f"cannot parse Requires-Dist value {value!r}")
    return _canonicalize_name(match.group(1))


def _canonical_requirement(value: str) -> str:
    requirement = value.partition(";")[0].strip()
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        raise BoundaryError(f"cannot parse Requires-Dist value {value!r}")
    name = _canonicalize_name(match.group(1))
    suffix = re.sub(r"\s+", "", requirement[match.end() :]).lower()
    return f"{name}{suffix}"


def _requirement_extra(value: str) -> str | None:
    marker = value.partition(";")[2]
    matches = re.findall(r"\bextra\s*==\s*(['\"])([A-Za-z0-9._-]+)\1", marker)
    extras = {_canonicalize_name(extra) for _, extra in matches}
    if len(extras) > 1:
        raise BoundaryError(f"Requires-Dist has ambiguous extra markers: {value!r}")
    return next(iter(extras), None)


def _record_digest(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _validate_record(archive: ZipFile, members: frozenset[str], record_path: str) -> None:
    rows = list(csv.reader(io.StringIO(archive.read(record_path).decode("utf-8"))))
    recorded: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise BoundaryError(f"{record_path}: RECORD row must have three columns: {row!r}")
        path, digest, size = row
        if path in recorded:
            raise BoundaryError(f"{record_path}: duplicate RECORD path {path!r}")
        if path not in members:
            raise BoundaryError(f"{record_path}: records missing archive member {path!r}")
        recorded[path] = (digest, size)

        if path == record_path:
            if digest or size:
                raise BoundaryError(f"{record_path}: its own hash and size must be empty")
            continue

        payload = archive.read(path)
        expected_digest = _record_digest(payload)
        if digest != expected_digest:
            raise BoundaryError(
                f"{record_path}: hash mismatch for {path!r}: {digest!r} != {expected_digest!r}"
            )
        if size != str(len(payload)):
            raise BoundaryError(
                f"{record_path}: size mismatch for {path!r}: {size!r} != {len(payload)!r}"
            )

    missing = members.difference(recorded)
    extra = set(recorded).difference(members)
    if missing or extra:
        raise BoundaryError(
            f"{record_path}: RECORD membership differs from the wheel; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def inspect_wheel(path: Path, *, distribution: str, namespace: str) -> WheelInfo:
    """Read and validate one wheel, returning its publishable ownership facts."""
    path = path.resolve()
    if not path.is_file() or path.suffix != ".whl":
        raise BoundaryError(f"wheel does not exist: {path}")

    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        members = frozenset(names)
        if len(names) != len(members):
            raise BoundaryError(f"{path.name}: duplicate archive paths are not allowed")

        dist_info_dirs = {
            name.split("/", 1)[0]
            for name in members
            if "/" in name and name.split("/", 1)[0].endswith(".dist-info")
        }
        if len(dist_info_dirs) != 1:
            raise BoundaryError(
                f"{path.name}: expected one .dist-info directory, found {sorted(dist_info_dirs)}"
            )
        dist_info = next(iter(dist_info_dirs))
        record_path = f"{dist_info}/RECORD"
        metadata_path = f"{dist_info}/METADATA"
        if record_path not in members or metadata_path not in members:
            raise BoundaryError(f"{path.name}: wheel must contain METADATA and RECORD")

        _validate_record(archive, members, record_path)
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_path))
        actual_distribution = _canonicalize_name(str(metadata["Name"] or ""))
        expected_distribution = _canonicalize_name(distribution)
        if actual_distribution != expected_distribution:
            raise BoundaryError(
                f"{path.name}: METADATA Name is {actual_distribution!r}, "
                f"expected {expected_distribution!r}"
            )
        version = str(metadata["Version"] or "")
        if not version:
            raise BoundaryError(f"{path.name}: METADATA Version is required")

        runtime_files = frozenset(
            name for name in members if not name.startswith(f"{dist_info}/")
        )
        member_fingerprints = tuple(
            (name, hashlib.sha256(payload).hexdigest(), len(payload))
            for name in sorted(members)
            if name != record_path and (payload := archive.read(name))
        )
        wrong_owner = sorted(
            name for name in runtime_files if not name.startswith(f"{namespace}/")
        )
        if wrong_owner:
            raise BoundaryError(
                f"{path.name}: {distribution} may own only {namespace}/ outside .dist-info; "
                f"found {wrong_owner}"
            )
        if f"{namespace}/__init__.py" not in runtime_files:
            raise BoundaryError(f"{path.name}: missing {namespace}/__init__.py")

        return WheelInfo(
            path=path,
            distribution=actual_distribution,
            version=version,
            namespace=namespace,
            runtime_files=runtime_files,
            member_fingerprints=member_fingerprints,
            requirements=tuple(metadata.get_all("Requires-Dist", [])),
            extras=frozenset(metadata.get_all("Provides-Extra", [])),
        )


def _validate_extra_requirements(
    wheel: WheelInfo,
    expected: dict[str, tuple[str, ...]],
) -> None:
    for extra, expected_requirements in expected.items():
        actual = tuple(
            sorted(
                _canonical_requirement(value)
                for value in wheel.requirements
                if _requirement_extra(value) == extra
            )
        )
        wanted = tuple(sorted(expected_requirements))
        if actual != wanted:
            raise BoundaryError(
                f"{wheel.distribution}[{extra}] requirements changed: "
                f"{list(actual)} != {list(wanted)}"
            )


def _validate_dependency_direction(client: WheelInfo, nori: WheelInfo) -> None:
    client_requirement_names = {_requirement_name(value) for value in client.requirements}
    if "synthefy-nori" in client_requirement_names:
        raise BoundaryError("synthefy must not depend on synthefy-nori in any dependency group")
    if client.extras != {"aws", "forecasting", "text"}:
        raise BoundaryError(
            f"synthefy extras changed: {sorted(client.extras)} != ['aws', 'forecasting', 'text']"
        )
    _validate_extra_requirements(
        client,
        {
            "aws": ("boto3<2.0.0,>=1.34.0",),
            "forecasting": ("datasets>=2.0", "gluonts>=0.16", "statsmodels>=0.14"),
            "text": ("sentence-transformers",),
        },
    )
    _validate_extra_requirements(
        nori,
        {
            "forecasting": ("synthefy[forecasting]<8,>=7",),
            "text": ("synthefy[text]<8,>=7",),
        },
    )

    nori_edges = [
        value for value in nori.requirements if _requirement_name(value) == "synthefy"
    ]
    base_edges = [value for value in nori_edges if ";" not in value]
    if len(base_edges) != 1:
        raise BoundaryError(
            f"synthefy-nori must have exactly one unconditional synthefy edge, found {base_edges}"
        )
    normalized_edge = re.sub(r"\s+", "", base_edges[0]).lower().replace("_", "-")
    if normalized_edge not in {"synthefy<8,>=7", "synthefy>=7,<8"}:
        raise BoundaryError(
            "synthefy-nori's base dependency must be synthefy>=7,<8; "
            f"found {base_edges[0]!r}"
        )


def _validate_rebuild(direct: WheelInfo, rebuilt: WheelInfo) -> None:
    comparable = (
        "distribution",
        "version",
        "namespace",
        "runtime_files",
        "member_fingerprints",
        "requirements",
        "extras",
    )
    changed = [name for name in comparable if getattr(direct, name) != getattr(rebuilt, name)]
    if changed:
        raise BoundaryError(
            f"{direct.distribution}: direct wheel and sdist-rebuilt wheel differ in {changed}"
        )


def validate_artifacts(
    *,
    client_direct: Path,
    client_rebuilt: Path,
    nori_direct: Path,
    nori_rebuilt: Path,
) -> tuple[WheelInfo, WheelInfo]:
    """Validate the direct and sdist-rebuilt artifact pair for both distributions."""
    client = inspect_wheel(client_direct, distribution="synthefy", namespace="synthefy")
    client_from_sdist = inspect_wheel(
        client_rebuilt, distribution="synthefy", namespace="synthefy"
    )
    nori = inspect_wheel(
        nori_direct, distribution="synthefy-nori", namespace="synthefy_nori"
    )
    nori_from_sdist = inspect_wheel(
        nori_rebuilt, distribution="synthefy-nori", namespace="synthefy_nori"
    )

    _validate_rebuild(client, client_from_sdist)
    _validate_rebuild(nori, nori_from_sdist)
    overlap = client.runtime_files.intersection(nori.runtime_files)
    if overlap:
        raise BoundaryError(f"wheel runtime ownership overlaps: {sorted(overlap)}")
    _validate_dependency_direction(client, nori)
    return client, nori


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-direct", type=Path, required=True)
    parser.add_argument("--client-rebuilt", type=Path, required=True)
    parser.add_argument("--nori-direct", type=Path, required=True)
    parser.add_argument("--nori-rebuilt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    client, nori = validate_artifacts(
        client_direct=args.client_direct,
        client_rebuilt=args.client_rebuilt,
        nori_direct=args.nori_direct,
        nori_rebuilt=args.nori_rebuilt,
    )
    print(
        "distribution boundaries valid: "
        f"synthefy {client.version} owns {len(client.runtime_files)} files; "
        f"synthefy-nori {nori.version} owns {len(nori.runtime_files)} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
