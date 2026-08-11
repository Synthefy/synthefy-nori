#!/usr/bin/env python3
"""Probe cleanly installed ``synthefy`` distributions and their extras.

This script runs under the isolated interpreter receiving the built wheels.  It
rejects accidental workspace imports, exercises both import orders, and verifies
that uninstalling either distribution leaves the other namespace intact.  Its
extra probes verify dependency ownership without claiming that feature source has
already moved from ``synthefy_nori`` into the lightweight client.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import sys
import sysconfig
import tempfile
from pathlib import Path


_DISTRIBUTIONS = {
    "synthefy": "synthefy",
    "synthefy_nori": "synthefy-nori",
}

_EXTRA_CASES = (
    "client-aws",
    "client-forecasting",
    "client-text",
    "nori-forecasting",
    "nori-text",
)


def _site_packages() -> Path:
    return Path(sysconfig.get_path("purelib")).resolve()


def _assert_absent(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        raise AssertionError(
            f"{module_name} is still importable after its distribution was removed"
        )
    distribution = _DISTRIBUTIONS[module_name]
    try:
        importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return
    raise AssertionError(f"{distribution} metadata remains after uninstall")


def _import_installed(module_name: str):
    distribution = _DISTRIBUTIONS[module_name]
    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve()
    site_packages = _site_packages()
    if not module_file.is_relative_to(site_packages):
        raise AssertionError(
            f"{module_name} imported from {module_file}, outside clean environment {site_packages}"
        )
    actual_version = importlib.metadata.version(distribution)
    if getattr(module, "__version__", actual_version) != actual_version:
        raise AssertionError(f"{module_name}.__version__ does not match {actual_version}")
    return module


def _assert_not_loaded(*module_names: str) -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(f"{root}.") for root in module_names)
    )
    if loaded:
        raise AssertionError(f"optional modules loaded eagerly: {loaded}")


def _assert_not_importable(*module_names: str) -> None:
    importable = sorted(
        module_name
        for module_name in module_names
        if importlib.util.find_spec(module_name) is not None
    )
    if importable:
        raise AssertionError(f"unrelated optional modules are installed: {importable}")


def _import_required(*module_names: str) -> None:
    for module_name in module_names:
        importlib.import_module(module_name)


def _probe_client_extra(extra: str) -> None:
    _assert_absent("synthefy_nori")
    _import_installed("synthefy")

    optional_modules = {
        "aws": ("boto3", "botocore"),
        "forecasting": ("datasets", "gluonts", "statsmodels"),
        "text": ("sentence_transformers", "torch"),
    }
    all_optional = {name for names in optional_modules.values() for name in names}
    _assert_not_loaded(*sorted(all_optional))
    _assert_not_importable(*sorted(all_optional.difference(optional_modules[extra])))

    if extra == "aws":
        from synthefy.nori_client import _load_aws_sdk

        boto3, config = _load_aws_sdk()
        if boto3.__name__ != "boto3" or config.__name__ != "Config":
            raise AssertionError("the AWS extra did not load boto3 and botocore.Config")
    elif extra == "forecasting":
        _import_required("datasets", "gluonts", "statsmodels")
    else:
        from sentence_transformers import SentenceTransformer

        if not callable(SentenceTransformer):
            raise AssertionError("SentenceTransformer is not callable")

    print(f"synthefy[{extra}] owns its dependencies without installing synthefy-nori")


def _probe_nori_extra(extra: str) -> None:
    _assert_not_loaded("synthefy")
    _import_installed("synthefy_nori")
    _assert_not_loaded("synthefy")

    unrelated = {
        "forecasting": ("boto3", "botocore", "sentence_transformers"),
        "text": ("boto3", "botocore", "datasets", "gluonts", "statsmodels"),
    }
    _assert_not_importable(*unrelated[extra])

    if extra == "forecasting":
        _import_required("datasets", "gluonts", "statsmodels")
        from synthefy_nori.nori_ts import NoriTSForecaster

        if not callable(NoriTSForecaster):
            raise AssertionError("NoriTSForecaster is not callable")
    else:
        from sentence_transformers import SentenceTransformer
        from synthefy_nori.text_features import MultimodalPreprocessor

        if not callable(SentenceTransformer) or not callable(MultimodalPreprocessor):
            raise AssertionError("the text feature entry points are not callable")

    print(f"synthefy-nori[{extra}] forwards to the current feature entry point")


def probe_extra(case: str) -> None:
    owner, extra = case.split("-", 1)
    if owner == "client":
        _probe_client_extra(extra)
    else:
        _probe_nori_extra(extra)


def _assert_import_cause(loader_name: str, expected_name: str) -> None:
    from synthefy import nori_client

    loader = getattr(nori_client, loader_name)
    try:
        loader()
    except ModuleNotFoundError as exc:
        error = exc
        missing = exc
    except ImportError as exc:
        error = exc
        missing = exc.__cause__
    else:
        raise AssertionError(
            f"{loader_name} unexpectedly succeeded without a working synthefy_nori"
        )
    if not isinstance(missing, ModuleNotFoundError) or missing.name != expected_name:
        raise AssertionError(
            f"{loader_name} reported {missing!r}; expected "
            f"ModuleNotFoundError({expected_name!r})"
        ) from error


def _assert_all_loader_causes(expected_name: str) -> None:
    for loader_name in ("_load_local_predict", "_load_local_regressor"):
        sys.modules.pop("synthefy_nori", None)
        _assert_import_cause(loader_name, expected_name)


def _probe_missing_import_causes() -> None:
    """Keep top-level absence distinguishable from a transitive package failure.

    The frozen client currently wraps both errors with its legacy install hint.  The
    original ``ModuleNotFoundError.name`` remains available through ``__cause__``;
    this gate records that fact without claiming the user-facing behavior is fixed.
    """
    _assert_all_loader_causes("synthefy_nori")

    with tempfile.TemporaryDirectory(prefix="broken-synthefy-nori-") as temp_dir:
        package = Path(temp_dir) / "synthefy_nori"
        package.mkdir()
        (package / "__init__.py").write_text(
            "raise ModuleNotFoundError(\n"
            "    \"No module named 'sentinel_transitive_dep'\",\n"
            "    name='sentinel_transitive_dep',\n"
            ")\n"
        )
        sys.path.insert(0, temp_dir)
        importlib.invalidate_caches()
        sys.modules.pop("synthefy_nori", None)
        try:
            _assert_all_loader_causes("sentinel_transitive_dep")
        finally:
            sys.modules.pop("synthefy_nori", None)
            sys.path.remove(temp_dir)
            importlib.invalidate_caches()


def probe_both(order: str) -> None:
    module_order = {
        "client-first": ("synthefy", "synthefy_nori"),
        "nori-first": ("synthefy_nori", "synthefy"),
    }[order]
    for module_name in module_order:
        _import_installed(module_name)
    print(f"installed distributions import cleanly in {order} order")


def probe_client_only() -> None:
    _assert_absent("synthefy_nori")
    _import_installed("synthefy")
    _probe_missing_import_causes()
    print("synthefy remains healthy after synthefy-nori uninstall")


def probe_nori_only() -> None:
    _assert_absent("synthefy")
    _import_installed("synthefy_nori")
    print("synthefy_nori remains importable after synthefy uninstall")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="state", required=True)
    both = subparsers.add_parser("both", help="probe a co-installed environment")
    both.add_argument("--order", choices=("client-first", "nori-first"), required=True)
    subparsers.add_parser("client-only", help="probe after synthefy-nori is uninstalled")
    subparsers.add_parser("nori-only", help="probe after synthefy is uninstalled")
    extra = subparsers.add_parser("extra", help="probe one isolated optional extra")
    extra.add_argument("--case", choices=_EXTRA_CASES, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.state == "both":
        probe_both(args.order)
    elif args.state == "client-only":
        probe_client_only()
    elif args.state == "nori-only":
        probe_nori_only()
    else:
        probe_extra(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
