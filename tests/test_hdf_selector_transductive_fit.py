"""HighDimFeatureSelector.fit_on_test: the projection is fitted on context + test features (no labels)."""

import numpy as np
import pytest

from synthefy_nori.inference.predictor import NoriPredictor
from synthefy_nori.inference.preprocess import HighDimFeatureSelector


def _shifted_split(seed=0, n_train=400, n_test=100, n_features=40, n_shifted=10, shift=8.0):
    rng = np.random.default_rng(seed)
    x_train = rng.standard_normal((n_train, n_features))
    x_test = rng.standard_normal((n_test, n_features))
    # A block of features whose test values lie far outside the context range (time-varying covariates).
    x_test[:, :n_shifted] += shift
    return x_train.astype(np.float32), x_test.astype(np.float32)


def _selector(**kw):
    return HighDimFeatureSelector(strategy="svd_all", svd_components=8, n_features_threshold=16, **kw)


def test_flag_defaults_off_and_is_stored():
    assert _selector().fit_on_test is False
    assert _selector(fit_on_test=True).fit_on_test is True


def test_inductive_helper_fits_on_context_plus_test_when_enabled(monkeypatch):
    x_train, x_test = _shifted_split()
    seen = {}
    orig_fit = HighDimFeatureSelector.fit

    def recording_fit(self, x, categorical_features, seed, *, y=None, **kwargs):
        seen["rows"] = len(x)
        seen["y"] = y
        return orig_fit(self, x, categorical_features, seed, y=y, **kwargs)

    monkeypatch.setattr(HighDimFeatureSelector, "fit", recording_fit)
    step = _selector(fit_on_test=True)
    NoriPredictor._fit_transform_step_inductive(None, step, x_train, x_test, [], 0, y_train=np.zeros(len(x_train)))
    assert seen["rows"] == len(x_train) + len(x_test)
    assert seen["y"] is None  # labels are never part of the transductive fit

    seen.clear()
    step = _selector(fit_on_test=False)
    NoriPredictor._fit_transform_step_inductive(None, step, x_train, x_test, [], 0, y_train=np.zeros(len(x_train)))
    assert seen["rows"] == len(x_train)


def test_transductive_projection_reconstructs_shifted_test_rows_better():
    x_train, x_test = _shifted_split()
    ctx_only, both = _selector(), _selector()
    ctx_only.fit(x_train, [], 0)
    both.fit(np.vstack([x_train, x_test]), [], 0)
    assert not ctx_only.passthrough_ and not both.passthrough_

    def recon_error(sel):
        comps = np.asarray(sel.svd_model_.components_, dtype=np.float64)
        x = x_test.astype(np.float64)
        proj = x @ comps.T @ comps
        return float(np.mean((x - proj) ** 2))

    # The context-only basis cannot represent the shifted block; the joint basis can.
    assert recon_error(both) < 0.5 * recon_error(ctx_only)


def test_disabled_flag_is_byte_identical_to_previous_behaviour():
    x_train, x_test = _shifted_split()
    a, b = _selector(), _selector(fit_on_test=False)
    a.fit(x_train, [], 0)
    b.fit(x_train, [], 0)
    xa, _ = a.transform(x_test)
    xb, _ = b.transform(x_test)
    np.testing.assert_array_equal(xa, xb)


def test_shipped_configs_enable_transductive_fit():
    """Every shipped inference config that carries the selector turns the transductive fit on."""
    import json
    from pathlib import Path

    cfg_dir = Path(__file__).resolve().parents[1] / "src" / "synthefy_nori" / "configs"
    checked = 0
    for path in sorted(cfg_dir.glob("*.json")):
        data = json.loads(path.read_text())
        members = data if isinstance(data, list) else [data]
        for m in members:
            if isinstance(m, dict) and "HighDimFeatureSelector" in m:
                assert m["HighDimFeatureSelector"].get("fit_on_test") is True, f"{path.name} lacks fit_on_test"
                checked += 1
    assert checked >= 8  # at least the 8 default ensemble members
