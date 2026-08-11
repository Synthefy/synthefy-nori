#!/usr/bin/env python3
"""Probe the lifecycle of cleanly installed ``synthefy`` distribution wheels.

This script runs under the isolated interpreter receiving the built wheels.  It
rejects accidental workspace imports, exercises both import orders, and verifies
that uninstalling either distribution leaves the other namespace intact.
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
    "synthefy": ("synthefy", "7.0.0"),
    "synthefy_nori": ("synthefy-nori", "0.16.0"),
}


def _site_packages() -> Path:
    return Path(sysconfig.get_path("purelib")).resolve()


def _assert_absent(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        raise AssertionError(
            f"{module_name} is still importable after its distribution was removed"
        )
    distribution, _ = _DISTRIBUTIONS[module_name]
    try:
        importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return
    raise AssertionError(f"{distribution} metadata remains after uninstall")


def _import_installed(module_name: str):
    distribution, expected_version = _DISTRIBUTIONS[module_name]
    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve()
    site_packages = _site_packages()
    if not module_file.is_relative_to(site_packages):
        raise AssertionError(
            f"{module_name} imported from {module_file}, outside clean environment {site_packages}"
        )
    actual_version = importlib.metadata.version(distribution)
    if actual_version != expected_version:
        raise AssertionError(
            f"{distribution} version is {actual_version!r}, expected {expected_version!r}"
        )
    if getattr(module, "__version__", expected_version) != expected_version:
        raise AssertionError(f"{module_name}.__version__ does not match {expected_version}")
    return module


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
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.state == "both":
        probe_both(args.order)
    elif args.state == "client-only":
        probe_client_only()
    else:
        probe_nori_only()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
