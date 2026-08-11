"""Package-topology guards for the consolidated ``synthefy`` workspace."""

import ast
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 lane
    import tomli as tomllib


_ROOT = Path(__file__).resolve().parents[1]
_CLIENT = _ROOT / "libs" / "synthefy"
_OBSERVATION = (
    _ROOT
    / "docs"
    / "architecture"
    / "0001-consolidate-synthefy-source-tree-phase-2.json"
)


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def _requirements(values: list[str]) -> list[Requirement]:
    return [Requirement(value) for value in values]


def _named(requirements: list[Requirement], name: str) -> list[Requirement]:
    wanted = canonicalize_name(name)
    return [req for req in requirements if canonicalize_name(req.name) == wanted]


def _declared_version() -> str:
    tree = ast.parse((_CLIENT / "src" / "synthefy" / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            assert isinstance(node.value, ast.Constant)
            return str(node.value.value)
    raise AssertionError("synthefy.__version__ is not declared")


def test_project_identities_versions_and_build_backends_are_disjoint():
    root = _toml(_ROOT / "pyproject.toml")
    client = _toml(_CLIENT / "pyproject.toml")

    assert root["project"]["name"] == "synthefy-nori"
    assert root["project"]["version"] == "0.16.0"
    assert root["build-system"]["build-backend"] == "setuptools.build_meta"
    assert client["project"]["name"] == "synthefy"
    assert client["project"]["version"] == "7.0.0"
    assert _declared_version() == "7.0.0"
    assert client["build-system"] == {
        "requires": ["hatchling==1.27.0"],
        "build-backend": "hatchling.build",
    }
    assert client["dependency-groups"]["package"].count("hatchling==1.27.0") == 1


def test_uv_workspace_uses_the_lightweight_member_for_the_published_edge():
    root = _toml(_ROOT / "pyproject.toml")
    uv = root["tool"]["uv"]

    assert uv["workspace"]["members"] == ["libs/synthefy"]
    assert uv["sources"]["synthefy"] == {"workspace": True}
    assert uv["sources"]["torch"] == {"index": "pytorch-cu128"}
    assert root["tool"]["setuptools"]["packages"]["find"] == {
        "where": ["src"],
        "include": ["synthefy_nori*"],
    }


def test_published_dependencies_point_only_from_heavy_to_light():
    root = _toml(_ROOT / "pyproject.toml")["project"]
    client = _toml(_CLIENT / "pyproject.toml")["project"]
    root_requirements = _requirements(root["dependencies"])
    edge = _named(root_requirements, "synthefy")

    assert len(edge) == 1
    assert edge[0].specifier == SpecifierSet(">=7,<8")
    assert not edge[0].extras and edge[0].marker is None and edge[0].url is None
    assert Version(client["version"]) in edge[0].specifier

    client_optionals = client["optional-dependencies"]
    assert set(client_optionals) == {"aws", "forecasting", "text"}
    assert "local" not in client_optionals
    all_client_requirements = _requirements(
        client["dependencies"]
        + [value for values in client_optionals.values() for value in values]
    )
    assert not _named(all_client_requirements, "synthefy-nori")

    forbidden_base = {
        "boto3",
        "sentence-transformers",
        "synthefy-nori",
        "torch",
    }
    base_names = {canonicalize_name(req.name) for req in _requirements(client["dependencies"])}
    assert base_names.isdisjoint(forbidden_base)

    root_optionals = root["optional-dependencies"]
    forwarded_extras = {
        "forecasting": "forecasting",
        "text": "text",
        "timeseries": "forecasting",
    }
    for exposed_extra, client_extra in forwarded_extras.items():
        forwarded = _requirements(root_optionals[exposed_extra])
        assert len(forwarded) == 1
        assert canonicalize_name(forwarded[0].name) == "synthefy"
        assert forwarded[0].extras == {client_extra}
        assert forwarded[0].specifier == SpecifierSet(">=7,<8")


def test_namespaces_and_imports_do_not_create_a_base_runtime_cycle():
    root = _toml(_ROOT / "pyproject.toml")
    client = _toml(_CLIENT / "pyproject.toml")

    assert root["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "synthefy_nori*"
    ]
    assert client["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/synthefy"
    ]
    assert not (_ROOT / "src" / "synthefy").exists()
    assert not (_CLIENT / "src" / "synthefy_nori").exists()

    offenders = []
    for path in (_CLIENT / "src" / "synthefy").rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(name == "synthefy_nori" or name.startswith("synthefy_nori.") for name in modules):
                offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders, f"base synthefy imports synthefy_nori at module scope: {offenders}"


def test_the_root_lock_is_the_only_lock_and_contains_both_editable_projects():
    assert (_ROOT / "uv.lock").is_file()
    assert not (_CLIENT / "uv.lock").exists()

    lock = _toml(_ROOT / "uv.lock")
    assert lock["requires-python"] == ">=3.10"
    packages = lock["package"]
    root_entries = [item for item in packages if item["name"] == "synthefy-nori"]
    client_entries = [item for item in packages if item["name"] == "synthefy"]
    assert len(root_entries) == len(client_entries) == 1
    assert root_entries[0]["source"] == {"editable": "."}
    assert client_entries[0]["source"] == {"editable": "libs/synthefy"}
    assert client_entries[0]["version"] == "7.0.0"
    assert "synthefy" in {dep["name"] for dep in root_entries[0]["dependencies"]}

    hatchling = [item for item in packages if item["name"] == "hatchling"]
    assert len(hatchling) == 1
    assert hatchling[0]["version"] == "1.27.0"


def test_python_floors_distinguish_workspace_development_from_the_client_artifact():
    root = _toml(_ROOT / "pyproject.toml")["project"]
    client = _toml(_CLIENT / "pyproject.toml")["project"]
    observation = json.loads(_OBSERVATION.read_text())

    assert root["requires-python"] == ">=3.10"
    assert client["requires-python"] == ">=3.9"
    assert "Programming Language :: Python :: 3.8" not in client["classifiers"]
    assert observation["workspace"]["development_python_floor"] == ">=3.10"
    compatibility = observation["python_compatibility"]
    assert compatibility["accepted_target"] == ">=3.9"
    assert compatibility["lightweight_distribution"]["declared_floor"] == ">=3.9"
    heavy = compatibility["heavy_distribution"]
    assert heavy["declared_floor"] == ">=3.10"
    assert heavy["target_status"] == "blocked_pending_explicit_decision"
    assert heavy["blocker"]["dependency"] == "torch==2.10.0"
    aws = compatibility["optional_backend_follow_ups"][0]
    assert aws["extra"] == "aws"
    assert aws["status"] == "upstream_python39_support_ended"
    gaps = {gap["id"]: gap for gap in observation["known_integration_gaps"]}
    assert set(gaps) == {
        "client-api-cleanup",
        "optional-extra-artifact-validation",
        "text-source-relocation",
    }
    assert observation["publication"]["status"] == "blocked"
