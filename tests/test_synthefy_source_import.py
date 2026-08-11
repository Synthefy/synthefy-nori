"""Guard the reviewed Synthefy SDK snapshot import for issue #216."""

import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _REPO_ROOT / "libs" / "synthefy"
_MANIFEST_PATH = _PROJECT_ROOT / "SOURCE_SNAPSHOT.json"

_IMPORTED_BLOBS = {
    "CHANGELOG.md": "7b69d85855e255bc5af57239fa4256862ff9994f",
    "README.md": "66e6d0f2cef027f4579084d775531374cd65a884",
    "pyproject.toml": "b5af6433f10ffe069a9796b4fa0be8f074b05c12",
    "pytest.ini": "46dcf9a9271b23f5faf9417a5ef5c48c041bd316",
    "src/synthefy/__init__.py": "056405957f4d1db3659ec1ff7154b274412f37f3",
    "src/synthefy/api_client.py": "93e3706c47020139b81b0850deee0ec4895f8907",
    "src/synthefy/data_models.py": "725e79d92c73313da6ab1c4154b73701e61458f6",
    "src/synthefy/nori_client.py": "a6fb1d1d91677ff9aa8c99186088bf63fd24eea7",
    "src/synthefy/nori_data_models.py": "d915657b53526f461246b74e5de354000b508b74",
    "tests/__init__.py": "df085fb789a46419a7a00f080c6e4f917ee156d3",
    "tests/conftest.py": "878c02f191956c44e9f8518cc43ab5a09e302e0a",
    "tests/test_data_models.py": "c15648965d81080db88e230c08cba86400829ca3",
    "tests/test_nori_client.py": "c46acbe2b083e88eb569fe00f83c8056b7d378bf",
    "tests/test_nori_data_models.py": "cafc0ab4d1dc55e2345ed65a3d2f6555dc18807d",
}

_EXCLUDED_BLOBS = {
    ".github/workflows/publish.yaml": "37ec020d5cf1d529988e9cbb8ce4ffe56f25e046",
    ".github/workflows/tests.yaml": "5e2b53558884cfc4db053adbf46003beabea9e5a",
    ".gitignore": "220a3ca1f278e9e308298e95f74b6df598d86af9",
    "INTERNAL_README.md": "bf551d97e2932eb789f22d28d8fc569c9c33a73d",
    "dogfood/DOGFOOD.md": "dec0706f4d6ef3964df8bcaa76a287f5ece900f4",
    "dogfood/dogfood_local.py": "a9f78cb9e9862b7b6d829f400bfcd13039208adc",
    "dogfood/dogfood_remote.py": "e49e372206ab4eb1a27e2a90f5d63eb79493e6c9",
    "dogfood/dogfood_remote_realdata.py": "af939f573bba594928c38aa7e515f11cdf8fafee",
    "dogfood/modal_t4.py": "2009921ec1e5d00d1c58ff8f2bc8bd27db7cb8e1",
    "tests/online_tests/__init__.py": "62dcbc5f4f18dfa9da32395806c89c4e15f617ca",
    "tests/online_tests/test_core_forecast_backtest_api.py": ("c3706d9b5cc51df03d0d09e23223165a11d35bd2"),
    "tests/online_tests/test_hotel_demand.py": ("c4502e24af24d1e42fc219c7177b3009aa8a503d"),
    "tests/online_tests/test_inventory_forecasting.py": ("59927286a5a9bf768e8a046d29683577bce70e74"),
    "tests/online_tests/test_nori_memory_rungs.py": ("6da3cb8dd1fcac9163ce18f19800316f124d62db"),
    "tests/online_tests/test_pricing_simulation.py": ("520ba09a0127585e6acfa979886ec17e4d3af78d"),
    "uv.lock": "8eb03be0f17a16a394b2d7226bb2ed5526b2d8d6",
}

_TARGET_FILES = set(_IMPORTED_BLOBS) | {
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "SOURCE_SNAPSHOT.json",
    "licenses/Apache-2.0.txt",
}


def _load_manifest():
    return json.loads(_MANIFEST_PATH.read_text())


def _git_blob(path):
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()  # noqa: S324


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_project_entries():
    project_prefix = f"{_PROJECT_ROOT.relative_to(_REPO_ROOT)}/"
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z", "--", project_prefix],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    entries = {}
    for record in result.stdout.rstrip("\0").split("\0"):
        metadata, path = record.split("\t", 1)
        entries[path.removeprefix(project_prefix)] = metadata.split(" ", 1)[0]
    return entries


def test_snapshot_manifest_pins_the_source_and_selected_boundary():
    manifest = _load_manifest()

    assert manifest["source"] == {
        "repository": "Synthefy/synthefy",
        "commit": "9ecc3d2fad8e37e95869379cc05f328597e258f9",
        "tree": "c99362d3c2449991fea1985ea1dc510e9af6d3cc",
        "subtrees": {
            "src/synthefy": "4caf181f092208d2fdb5d5a64997d6d1046e7944",
            "tests": "b5eb4713a8af8d0cc7031675da67c82d46cdd0f7",
            "tests/online_tests": "74cc3c0a994a113633392900a8b1fcb05c2c173f",
        },
        "version": "6.3.0",
    }
    assert manifest["import"]["method"] == "single_snapshot_commit"
    assert manifest["import"]["target_project_root"] == "libs/synthefy"
    assert manifest["import"]["included_path_count"] == len(_IMPORTED_BLOBS)
    assert manifest["import"]["included_blobs"] == _IMPORTED_BLOBS


def test_byte_identical_imports_match_the_pinned_git_blobs():
    imported = _load_manifest()["import"]

    for relative_path in imported["byte_identical_paths"]:
        assert _git_blob(_PROJECT_ROOT / relative_path) == _IMPORTED_BLOBS[relative_path]


def test_reviewed_transformations_and_license_files_are_pinned():
    manifest = _load_manifest()

    for relative_path, transformation in manifest["import"]["transformed_paths"].items():
        assert transformation["source_blob"] == _IMPORTED_BLOBS[relative_path]
        assert transformation["result_blob"] == _git_blob(_PROJECT_ROOT / relative_path)

    for relative_path, license_file in manifest["license_treatment"]["added_files"].items():
        target = _PROJECT_ROOT / relative_path
        assert license_file["git_blob"] == _git_blob(target)
        assert license_file["sha256"] == _sha256(target)

    for relative_path, build_file in manifest["build_treatment"]["added_files"].items():
        target = _PROJECT_ROOT / relative_path
        assert build_file["git_blob"] == _git_blob(target)
        assert build_file["sha256"] == _sha256(target)
        assert build_file["replaces_source_blob"] == _EXCLUDED_BLOBS[relative_path]


def test_imported_project_has_the_exact_reviewed_file_boundary():
    tracked = _tracked_project_entries()

    assert len(_TARGET_FILES) == 19
    assert set(tracked) == _TARGET_FILES
    assert set(tracked.values()) == {"100644"}


@pytest.mark.slow
def test_built_artifacts_match_reviewed_file_boundaries(tmp_path):
    manifest = _load_manifest()
    subprocess.run(
        ["uv", "build", "--project", str(_PROJECT_ROOT), "--out-dir", str(tmp_path)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    version = manifest["source"]["version"]
    sdist_prefix = f"synthefy-{version}/"
    with tarfile.open(tmp_path / f"synthefy-{version}.tar.gz") as archive:
        files = {member.name.removeprefix(sdist_prefix) for member in archive.getmembers() if member.isfile()}
        assert files == set(manifest["build_treatment"]["artifact_files"]["sdist"])
        gitignore = archive.extractfile(f"{sdist_prefix}.gitignore")
        assert gitignore is not None
        assert gitignore.read() == (_PROJECT_ROOT / ".gitignore").read_bytes()

    wheel = tmp_path / f"synthefy-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        assert set(archive.namelist()) == set(manifest["build_treatment"]["artifact_files"]["wheel"])


def test_deliberately_excluded_snapshot_paths_are_pinned_and_absent():
    manifest = _load_manifest()
    excluded = manifest["excluded"]
    replacement_paths = set(manifest["build_treatment"]["added_files"])
    recorded = {
        **excluded["repository_control_and_stale_guidance"]["blobs"],
        **excluded["dogfood_and_live_tests"]["blobs"],
    }

    assert excluded["path_count"] == len(_EXCLUDED_BLOBS)
    assert recorded == _EXCLUDED_BLOBS
    assert replacement_paths == {".gitignore"}
    assert all(
        not (_PROJECT_ROOT / relative_path).exists()
        for relative_path in recorded
        if relative_path not in replacement_paths
    )
    assert excluded["dogfood_and_live_tests"]["status"] == ("omitted_pending_rehome_review")
