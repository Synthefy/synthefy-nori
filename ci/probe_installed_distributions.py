#!/usr/bin/env python3
"""Probe cleanly installed ``synthefy`` distributions and their extras.

This script runs under the isolated interpreter receiving the built wheels.  It
rejects accidental workspace imports, exercises both import orders, and verifies
that uninstalling either distribution leaves the other namespace intact.  Its
extra probes verify dependency and implementation ownership across the package
boundary.
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

import pandas as pd


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

_CLIENT_LAZY_MODULES = (
    "boto3",
    "botocore",
    "datasets",
    "gluonts",
    "joblib",
    "scipy",
    "sklearn",
    "sentence_transformers",
    "statsmodels",
    "synthefy_nori",
    "torch",
)


def _site_packages() -> Path:
    return Path(sysconfig.get_path("purelib")).resolve()


def _assert_absent(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        raise AssertionError(f"{module_name} is still importable after its distribution was removed")
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
        raise AssertionError(f"{module_name} imported from {module_file}, outside clean environment {site_packages}")
    actual_version = importlib.metadata.version(distribution)
    if getattr(module, "__version__", actual_version) != actual_version:
        raise AssertionError(f"{module_name}.__version__ does not match {actual_version}")
    return module


def _assert_not_loaded(*module_names: str) -> None:
    loaded = sorted(
        name for name in sys.modules if any(name == root or name.startswith(f"{root}.") for root in module_names)
    )
    if loaded:
        raise AssertionError(f"optional modules loaded eagerly: {loaded}")


def _assert_not_importable(*module_names: str) -> None:
    importable = sorted(
        module_name for module_name in module_names if importlib.util.find_spec(module_name) is not None
    )
    if importable:
        raise AssertionError(f"unrelated optional modules are installed: {importable}")


def _import_required(*module_names: str) -> None:
    for module_name in module_names:
        importlib.import_module(module_name)


def _canonical_featurizer():
    from synthefy.featurize import align_and_featurize

    if not callable(align_and_featurize) or align_and_featurize.__module__ != "synthefy.featurize":
        raise AssertionError("synthefy does not own the canonical tabular featurizer")
    return align_and_featurize


def _canonical_tsfeatures():
    from synthefy.nori_ts import tsfeatures

    frame = tsfeatures.TimeSeriesDataFrame.from_data_frame(
        pd.DataFrame(
            {
                "item_id": [0, 0],
                "timestamp": pd.date_range("2021-01-01", periods=2, freq="h"),
                "target": [1.0, 2.0],
            }
        )
    )
    horizon = tsfeatures.generate_test_X(frame, prediction_length=1, freq="h")
    if len(horizon) != 1 or not horizon["target"].isna().all():
        raise AssertionError("canonical time-series preparation did not execute")
    return tsfeatures


def _assert_legacy_tsfeatures_are_canonical():
    canonical = _canonical_tsfeatures()
    legacy = importlib.import_module("synthefy_nori.nori_ts.tsfeatures")
    if legacy.__all__ != canonical.__all__:
        raise AssertionError("historical and canonical time-series exports differ")
    for public_name in canonical.__all__:
        if getattr(legacy, public_name) is not getattr(canonical, public_name):
            raise AssertionError(f"historical {public_name} is not canonical")
    for module_name in (
        "auto_features",
        "basic_features",
        "data_preparation",
        "feature_generator_base",
        "feature_transformer",
        "ts_dataframe",
    ):
        canonical_module = importlib.import_module(f"synthefy.nori_ts.tsfeatures.{module_name}")
        historical_module = importlib.import_module(f"synthefy_nori.nori_ts.tsfeatures.{module_name}")
        if historical_module is not canonical_module:
            raise AssertionError(f"historical deep module {module_name} is not canonical")
    return canonical


def _probe_client_extra(extra: str) -> None:
    _assert_absent("synthefy_nori")
    _import_installed("synthefy")
    _canonical_featurizer()

    optional_modules = {
        "aws": ("boto3", "botocore"),
        "forecasting": ("datasets", "gluonts", "joblib", "scipy", "statsmodels"),
        "text": ("joblib", "scipy", "sentence_transformers", "sklearn", "torch"),
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
        _import_required("datasets", "gluonts", "joblib", "scipy", "statsmodels")
        _canonical_tsfeatures()
    else:
        from sentence_transformers import SentenceTransformer
        from synthefy.text_features import MultimodalPreprocessor

        if not callable(SentenceTransformer) or not callable(MultimodalPreprocessor):
            raise AssertionError("the client text entry points are not callable")

    print(f"synthefy[{extra}] owns its dependencies without installing synthefy-nori")


def _probe_nori_extra(extra: str) -> None:
    _import_installed("synthefy_nori")
    selected_modules = {
        "forecasting": ("datasets", "gluonts", "statsmodels"),
        "text": ("sentence_transformers",),
    }
    _assert_not_loaded(*selected_modules[extra])

    unrelated = {
        "forecasting": ("boto3", "botocore", "sentence_transformers"),
        "text": (
            "boto3",
            "botocore",
            "datasets",
            "gluonts",
            "statsmodels",
        ),
    }
    _assert_not_importable(*unrelated[extra])

    if extra == "forecasting":
        _import_required("datasets", "gluonts", "joblib", "scipy", "statsmodels")
        from synthefy_nori.nori_ts import NoriTSForecaster

        if not callable(NoriTSForecaster):
            raise AssertionError("NoriTSForecaster is not callable")
        _assert_legacy_tsfeatures_are_canonical()
    else:
        from sentence_transformers import SentenceTransformer
        from synthefy.text_features import MultimodalPreprocessor as CanonicalPreprocessor
        from synthefy_nori.text_features import MultimodalPreprocessor as LegacyPreprocessor

        if (
            not callable(SentenceTransformer)
            or not callable(CanonicalPreprocessor)
            or LegacyPreprocessor is not CanonicalPreprocessor
        ):
            raise AssertionError("the text feature entry points are not callable")

    print(f"synthefy-nori[{extra}] forwards to the current feature entry point")


def probe_extra(case: str) -> None:
    owner, extra = case.split("-", 1)
    if owner == "client":
        _probe_client_extra(extra)
    else:
        _probe_nori_extra(extra)


def _assert_import_cause(loader_name: str, expected_name: str, *, expect_wrapped: bool) -> None:
    from synthefy import nori_client

    loader = getattr(nori_client, loader_name)
    try:
        loader()
    except ModuleNotFoundError as exc:
        error = exc
        missing = exc
        wrapped = False
    except ImportError as exc:
        error = exc
        missing = exc.__cause__
        wrapped = True
    else:
        raise AssertionError(f"{loader_name} unexpectedly succeeded without a working synthefy_nori")
    if not isinstance(missing, ModuleNotFoundError) or missing.name != expected_name:
        raise AssertionError(
            f"{loader_name} reported {missing!r}; expected ModuleNotFoundError({expected_name!r})"
        ) from error
    if wrapped is not expect_wrapped:
        disposition = "wrapped" if wrapped else "unwrapped"
        raise AssertionError(
            f"{loader_name} left {expected_name!r} {disposition}; expect_wrapped={expect_wrapped}"
        ) from error


def _assert_all_loader_causes(expected_name: str, *, expect_wrapped: bool) -> None:
    for loader_name in ("_load_local_predict", "_load_local_regressor"):
        sys.modules.pop("synthefy_nori", None)
        _assert_import_cause(loader_name, expected_name, expect_wrapped=expect_wrapped)


def _probe_missing_import_causes() -> None:
    """Wrap only top-level absence; preserve transitive package failures.

    A missing optional ``synthefy_nori`` package gets the actionable install hint.
    A dependency missing *inside* that package must retain its original exception
    so the hint cannot misdiagnose an already-installed package.
    """
    _assert_all_loader_causes("synthefy_nori", expect_wrapped=True)

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
            _assert_all_loader_causes("sentinel_transitive_dep", expect_wrapped=False)
        finally:
            sys.modules.pop("synthefy_nori", None)
            sys.path.remove(temp_dir)
            importlib.invalidate_caches()


def probe_both(order: str) -> None:
    if order == "client-first":
        _import_installed("synthefy")
        _assert_not_loaded(*_CLIENT_LAZY_MODULES)
        _import_installed("synthefy_nori")
    else:
        _import_installed("synthefy_nori")
        _import_installed("synthefy")
    canonical = _canonical_featurizer()
    from synthefy_nori.featurize import align_and_featurize as legacy

    if legacy is not canonical:
        raise AssertionError("synthefy_nori does not re-export the canonical featurizer")
    _assert_legacy_tsfeatures_are_canonical()
    print(f"installed distributions import cleanly in {order} order")


def probe_client_only() -> None:
    _assert_absent("synthefy_nori")
    _import_installed("synthefy")
    _canonical_featurizer()
    _canonical_tsfeatures()
    _probe_missing_import_causes()
    print("synthefy remains healthy after synthefy-nori uninstall")


def probe_nori_only() -> None:
    """The heavy wheel remains installed but needs its declared base dependency."""
    _assert_absent("synthefy")
    if importlib.util.find_spec("synthefy_nori") is None:
        raise AssertionError("synthefy_nori files disappeared after synthefy uninstall")
    try:
        importlib.import_module("synthefy_nori")
    except ModuleNotFoundError as exc:
        if exc.name != "synthefy":
            raise AssertionError(f"synthefy_nori failed for {exc.name!r}, expected its required synthefy edge") from exc
    else:
        raise AssertionError("synthefy_nori imported without its required synthefy dependency")
    print("synthefy_nori remains installed and reports its missing synthefy dependency")


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
