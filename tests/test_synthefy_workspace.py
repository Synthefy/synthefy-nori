"""Package-topology guards for the consolidated ``synthefy`` workspace."""

import ast
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
    assert root["project"]["version"] == "0.19.0"
    assert root["build-system"]["build-backend"] == "setuptools.build_meta"
    assert client["project"]["name"] == "synthefy"
    assert client["project"]["version"] == "7.0.4"
    assert _declared_version() == "7.0.4"
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
    assert "torch" not in uv["sources"]
    assert uv["constraint-dependencies"] == ["torch<2.9"]
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
        "datasets",
        "gluonts",
        "joblib",
        "scipy",
        "scikit-learn",
        "sentence-transformers",
        "statsmodels",
        "synthefy-nori",
        "torch",
    }
    base_names = {canonicalize_name(req.name) for req in _requirements(client["dependencies"])}
    assert base_names.isdisjoint(forbidden_base)

    forecasting = _requirements(client_optionals["forecasting"])
    assert {
        canonicalize_name(requirement.name): str(requirement.specifier)
        for requirement in forecasting
    } == {
        "datasets": ">=2.0",
        "gluonts": ">=0.16",
        "joblib": ">=1.1",
        "scipy": ">=1.13",
        "statsmodels": ">=0.14",
    }

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
        "joblib",
        "scipy",
        "sentence_transformers",
        "statsmodels",
        "synthefy_nori",
        "torch",
    }
    client_offenders = []
    for path in (_CLIENT / "src" / "synthefy").rglob("*.py"):
        if path.is_relative_to(_CLIENT_TSFEATURES):
            continue
        if any(name.split(".", 1)[0] in client_forbidden for name in _import_time_modules(path)):
            client_offenders.append(str(path.relative_to(_ROOT)))
    assert not client_offenders, (
        "base synthefy imports a heavy or forecasting-only dependency at module scope: "
        f"{client_offenders}"
    )

    tsfeature_forbidden = {
        "boto3",
        "botocore",
        "sentence_transformers",
        "sklearn",
        "synthefy_nori",
        "torch",
    }
    tsfeature_offenders = []
    for path in _CLIENT_TSFEATURES.rglob("*.py"):
        if any(
            name.split(".", 1)[0] in tsfeature_forbidden
            for name in _import_time_modules(path)
        ):
            tsfeature_offenders.append(str(path.relative_to(_ROOT)))
    assert not tsfeature_offenders, (
        "model-free time-series preparation imports a heavy/client backend: "
        f"{tsfeature_offenders}"
    )
    assert _import_time_modules(
        _CLIENT / "src" / "synthefy" / "nori_ts" / "__init__.py"
    ) == ["synthefy.nori_ts.core"]

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
    canonical_helpers = {
        "_has_encodable_columns",
        "_numeric_categories_to_values",
        "align_and_featurize",
    }

    assert canonical_helpers <= canonical_defs
    assert canonical_helpers.isdisjoint(client_defs)
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


def test_time_series_forecaster_has_one_lightweight_implementation_owner():
    canonical = _CLIENT_TSFEATURES
    canonical_core = _CLIENT / "src" / "synthefy" / "nori_ts" / "core.py"
    legacy = _ROOT / "src" / "synthefy_nori" / "nori_ts" / "tsfeatures"
    legacy_core = _ROOT / "src" / "synthefy_nori" / "nori_ts" / "core.py"

    assert {path.name for path in canonical.glob("*.py")} == {
        "__init__.py",
        "auto_features.py",
        "basic_features.py",
        "data_preparation.py",
        "feature_generator_base.py",
        "feature_transformer.py",
        "ts_dataframe.py",
    }
    assert {path.name for path in legacy.glob("*.py")} == {
        "__init__.py",
        "auto_features.py",
        "basic_features.py",
        "data_preparation.py",
        "feature_generator_base.py",
        "feature_transformer.py",
        "ts_dataframe.py",
    }
    core_modules = set(_import_time_modules(canonical_core))
    assert "synthefy.nori_client" in core_modules
    assert "synthefy.nori_ts.tsfeatures" in core_modules
    assert not any(name.startswith("synthefy_nori") for name in core_modules)
    legacy_tree = ast.parse(legacy_core.read_text())
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in legacy_tree.body
    )
    assert _import_time_modules(legacy_core) == ["synthefy.nori_ts.core"]
    facade = (legacy / "__init__.py").read_text()
    assert '_CANONICAL_PACKAGE = "synthefy.nori_ts.tsfeatures"' in facade
    assert 'exc.name not in {"synthefy.nori_ts", _CANONICAL_PACKAGE}' in facade


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
    assert lock["requires-python"] == ">=3.9"
    packages = lock["package"]
    root_entries = [item for item in packages if item["name"] == "synthefy-nori"]
    client_entries = [item for item in packages if item["name"] == "synthefy"]
    assert len(root_entries) == len(client_entries) == 1
    assert root_entries[0]["source"] == {"editable": "."}
    assert client_entries[0]["source"] == {"editable": "libs/synthefy"}
    assert client_entries[0]["version"] == "7.0.4"
    assert "synthefy" in {dep["name"] for dep in root_entries[0]["dependencies"]}

    hatchling = [item for item in packages if item["name"] == "hatchling"]
    assert len(hatchling) == 1
    assert hatchling[0]["version"] == "1.27.0"

    forecasting = client_entries[0]["optional-dependencies"]["forecasting"]
    assert {dependency["name"] for dependency in forecasting} == {
        "datasets",
        "gluonts",
        "joblib",
        "scipy",
        "statsmodels",
    }
    metadata = client_entries[0]["metadata"]["requires-dist"]
    assert any(
        requirement["name"] == "joblib"
        and requirement["marker"] == "extra == 'forecasting'"
        and requirement["specifier"] == ">=1.1"
        for requirement in metadata
    )
    assert any(
        requirement["name"] == "scipy"
        and requirement["marker"] == "extra == 'forecasting'"
        and requirement["specifier"] == ">=1.13"
        for requirement in metadata
    )


def test_python_floors_preserve_public_python39_compatibility():
    root = _toml(_ROOT / "pyproject.toml")["project"]
    client = _toml(_CLIENT / "pyproject.toml")["project"]

    assert (
        "eval-type-backport>=0.2; python_version < '3.10'"
        in root["dependencies"]
    )
    assert root["requires-python"] == ">=3.9"
    assert client["requires-python"] == ">=3.9"
    assert "Programming Language :: Python :: 3.9" in root["classifiers"]
    assert "Programming Language :: Python :: 3.9" in client["classifiers"]
    assert "Programming Language :: Python :: 3.8" not in client["classifiers"]

def test_living_docs_name_supported_install_paths(monkeypatch):
    readme = (_ROOT / "README.md").read_text()
    assert "pip install synthefy" in readme
    assert 'pip install "synthefy[aws]"' in readme
    assert "pip install synthefy-nori" in readme
    assert 'pip install "synthefy[forecasting]"' in readme
    assert 'pip install "synthefy-nori[forecasting]"' in readme
    assert 'SynthefyNoriClient(mode="remote", model="nori-30m")' in readme
    assert "`auto` is\nnot supported." in readme
    assert "require an explicit `model=`; there is no default model." in readme
    assert "NoriRegressor(memory_policy=" not in readme

    monkeypatch.setenv("SYNTHEFY_NORI_API_KEY", "test")
    from synthefy import SynthefyNoriClient

    client = SynthefyNoriClient(model="nori-30m")
    assert client.mode == "remote"
    client.close()

    living_paths = (
        _ROOT / "README.md",
        _ROOT / "AGENTS.md",
        _ROOT / "src" / "synthefy_nori" / "nori_ts" / "__init__.py",
    )
    assert not [path for path in living_paths if "client-sync" in path.read_text()]
