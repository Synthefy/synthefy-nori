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
_CLIENT_TSFEATURES = _CLIENT / "src" / "synthefy" / "nori_ts" / "tsfeatures"
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


def _is_type_checking_guard(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "TYPE_CHECKING"
        or isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
        and node.attr == "TYPE_CHECKING"
    )


class _ImportTimeVisitor(ast.NodeVisitor):
    """Collect imports executed while a module is imported."""

    def __init__(self) -> None:
        self.modules: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.modules.extend(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.modules.append(node.module or "")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            for statement in node.orelse:
                self.visit(statement)
            return
        if isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not):
            if _is_type_checking_guard(node.test.operand):
                for statement in node.body:
                    self.visit(statement)
                return
        self.generic_visit(node)


def _import_time_modules(path: Path) -> list[str]:
    visitor = _ImportTimeVisitor()
    visitor.visit(ast.parse(path.read_text()))
    return visitor.modules


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
        "scikit-learn",
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

    client_forbidden = {
        "boto3",
        "botocore",
        "datasets",
        "gluonts",
        "sentence_transformers",
        "statsmodels",
        "synthefy_nori",
        "torch",
    }
    client_offenders = []
    for path in (_CLIENT / "src" / "synthefy").rglob("*.py"):
        if any(name.split(".", 1)[0] in client_forbidden for name in _import_time_modules(path)):
            client_offenders.append(str(path.relative_to(_ROOT)))
    assert not client_offenders, (
        "base synthefy imports a heavy or optional dependency at module scope: "
        f"{client_offenders}"
    )

    facade_offenders = []
    for path in (_ROOT / "src" / "synthefy_nori").rglob("*.py"):
        modules = _import_time_modules(path)
        if any(
            name == "synthefy" or name.startswith("synthefy.nori_client") for name in modules
        ):
            facade_offenders.append(str(path.relative_to(_ROOT)))
    assert not facade_offenders, (
        f"synthefy_nori imports the lightweight client facade at module scope: {facade_offenders}"
    )


def test_tabular_preparation_has_one_v7_implementation_owner():
    canonical_path = _CLIENT / "src" / "synthefy" / "featurize.py"
    client_path = _CLIENT / "src" / "synthefy" / "nori_client.py"
    legacy_path = _ROOT / "src" / "synthefy_nori" / "featurize.py"

    canonical_tree = ast.parse(canonical_path.read_text())
    client_tree = ast.parse(client_path.read_text())
    legacy_tree = ast.parse(legacy_path.read_text())
    canonical_defs = {
        node.name for node in canonical_tree.body if isinstance(node, ast.FunctionDef)
    }
    client_defs = {
        node.name for node in client_tree.body if isinstance(node, ast.FunctionDef)
    }
    duplicate_helpers = {
        "_has_encodable_columns",
        "_numeric_categories_to_values",
        "_featurize_frames",
        "align_and_featurize",
    }

    assert duplicate_helpers <= canonical_defs
    assert duplicate_helpers.isdisjoint(client_defs)
    assert not any(isinstance(node, ast.FunctionDef) for node in legacy_tree.body)

    builder = next(
        node
        for node in client_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_nori_request"
    )
    canonical_calls = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_align_and_featurize"
    ]
    assert len(canonical_calls) == 1
    stacklevel = next(
        keyword.value
        for keyword in canonical_calls[0].keywords
        if keyword.arg == "_warning_stacklevel"
    )
    assert isinstance(stacklevel, ast.Constant) and stacklevel.value == 5


def test_import_time_scan_descends_guards_but_skips_deferred_imports(tmp_path):
    module = tmp_path / "guarded_imports.py"
    module.write_text(
        "from typing import TYPE_CHECKING\n"
        "try:\n"
        "    import guarded_dependency\n"
        "except ImportError:\n"
        "    pass\n"
        "if TYPE_CHECKING:\n"
        "    import type_only_dependency\n"
        "if not TYPE_CHECKING:\n"
        "    import runtime_dependency\n"
        "def deferred():\n"
        "    import deferred_dependency\n"
    )

    modules = set(_import_time_modules(module))

    assert {"typing", "guarded_dependency", "runtime_dependency"} <= modules
    assert {"type_only_dependency", "deferred_dependency"}.isdisjoint(modules)


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
