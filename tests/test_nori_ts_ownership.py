"""Ownership and compatibility guards for shared time-series preparation."""

from __future__ import annotations

import importlib
import sys

import pandas as pd
import pytest


pytest.importorskip("datasets")
pytest.importorskip("gluonts")
pytest.importorskip("joblib")
pytest.importorskip("scipy")
pytest.importorskip("statsmodels")

import synthefy.nori_ts.tsfeatures as canonical
import synthefy_nori.nori_ts.tsfeatures as legacy


_DEEP_MODULES = (
    "auto_features",
    "basic_features",
    "data_preparation",
    "feature_generator_base",
    "feature_transformer",
    "ts_dataframe",
)
_PUBLIC = (
    "TimeSeriesDataFrame",
    "generate_test_X",
    "FeatureGenerator",
    "FeatureTransformer",
    "RunningIndexFeature",
    "CalendarFeature",
    "AdditionalCalendarFeature",
    "AutoSeasonalFeature",
)


def test_v7_legacy_exports_and_deep_modules_are_canonical():
    assert legacy.__all__ == canonical.__all__ == list(_PUBLIC)
    assert all(getattr(legacy, name) is getattr(canonical, name) for name in _PUBLIC)
    for name in _DEEP_MODULES:
        canonical_module = importlib.import_module(
            f"synthefy.nori_ts.tsfeatures.{name}"
        )
        historical_module = importlib.import_module(
            f"synthefy_nori.nori_ts.tsfeatures.{name}"
        )
        assert historical_module is canonical_module


def test_heavy_forecaster_core_still_consumes_the_historical_facade():
    from synthefy_nori.nori_ts import core

    assert core.TimeSeriesDataFrame is legacy.TimeSeriesDataFrame
    assert core.FeatureTransformer is legacy.FeatureTransformer


def test_facade_falls_back_only_when_the_canonical_owner_is_missing(monkeypatch):
    real_import_module = importlib.import_module

    def missing_canonical(name, package=None):
        if name == "synthefy.nori_ts.tsfeatures":
            raise ModuleNotFoundError(
                "No module named 'synthefy.nori_ts.tsfeatures'",
                name="synthefy.nori_ts.tsfeatures",
            )
        return real_import_module(name, package)

    for name in _DEEP_MODULES:
        sys.modules.pop(f"synthefy_nori.nori_ts.tsfeatures.{name}", None)
    monkeypatch.setattr(importlib, "import_module", missing_canonical)
    try:
        fallback = importlib.reload(legacy)
        assert fallback.TimeSeriesDataFrame.__module__ == (
            "synthefy_nori.nori_ts.tsfeatures.ts_dataframe"
        )
        frame = fallback.TimeSeriesDataFrame.from_data_frame(
            pd.DataFrame(
                {
                    "item_id": [0, 0],
                    "timestamp": pd.date_range("2021-01-01", periods=2, freq="h"),
                    "target": [1.0, 2.0],
                }
            )
        )
        assert list(frame.item_ids) == [0]
    finally:
        for name in _DEEP_MODULES:
            sys.modules.pop(f"synthefy_nori.nori_ts.tsfeatures.{name}", None)
        monkeypatch.undo()
        importlib.reload(legacy)

    assert legacy.TimeSeriesDataFrame is canonical.TimeSeriesDataFrame


def test_facade_does_not_mask_a_transitive_canonical_import_failure(monkeypatch):
    real_import_module = importlib.import_module

    def broken_canonical(name, package=None):
        if name == "synthefy.nori_ts.tsfeatures":
            raise ModuleNotFoundError(
                "No module named 'sentinel_ts_dependency'",
                name="sentinel_ts_dependency",
            )
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", broken_canonical)
    try:
        with pytest.raises(ModuleNotFoundError) as caught:
            importlib.reload(legacy)
        assert caught.value.name == "sentinel_ts_dependency"
    finally:
        monkeypatch.undo()
        importlib.reload(legacy)
