"""The SVD rank must scale with rows, not merely be bounded by them.

Regression guard for the defect where ``min(svd_components, p-1, n-1)`` returned the
MAXIMUM available rank exactly when rows were scarcest: on a 1901-feature x 190-row
table it yielded 189, i.e. an orthogonal rotation into the full row space that keeps
every noise direction rather than a reduction.
"""

import pathlib

import numpy as np
import pytest

from synthefy_nori.inference.preprocess import HighDimFeatureSelector


def _fit(x, **kw):
    sel = HighDimFeatureSelector(strategy="svd_all", n_features_threshold=256, **kw)
    sel.fit_transform(x, categorical_features=[], seed=0)
    return sel


def _rank(sel):
    """Components actually retained, or None when the selector never fitted an SVD.

    Read off the fitted model rather than a bookkeeping attribute so the assertion
    covers the rank the data is really projected onto.
    """
    model = getattr(sel, "svd_model_", None)
    if model is None or getattr(model, "components_", None) is None:
        return None
    return int(model.components_.shape[0])


def test_short_wide_table_is_reduced_not_rotated():
    """190 rows x 1901 features: the old rule kept 189 components (all of them)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(190, 1901))
    sel = _fit(x, svd_components=512, svd_rows_per_component=3)
    k = _rank(sel)
    assert k is not None
    assert k <= 190 // 3, f"rank {k} not tied to rows (expected <= {190 // 3})"
    assert k < min(x.shape) - 1, "rank equals full row space -- a rotation, not a reduction"


def test_tall_table_still_capped_by_svd_components():
    """The rows term must not remove the cap: uncapped min(p, n//3) scored -0.045."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6234, 2000))
    sel = _fit(x, svd_components=512, svd_rows_per_component=3)
    assert _rank(sel) == 512, f"expected the 512 cap to bind, got {_rank(sel)}"


def test_default_is_the_previous_behaviour():
    """Opt-in by design: the bare constructor must not change any existing caller.
    A fixed low cap is known to REGRESS genuinely high-rank wide data, and the evidence
    for this rule is low-rank spectral tables only -- so it ships in the config, not
    here."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 400))
    sel = _fit(x, svd_components=256)
    assert _rank(sel) == 256, f"default changed behaviour: got {_rank(sel)}"


def test_rows_per_component_one_restores_previous_behaviour():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(190, 1901))
    sel = _fit(x, svd_components=512, svd_rows_per_component=1)
    assert _rank(sel) == 189, f"expected the legacy 189, got {_rank(sel)}"


def test_narrow_table_is_untouched_by_the_gate():
    """p <= n_features_threshold: the selector must not fire at all (144/145 of the
    tracked suite), so this change is a no-op on the standard benchmarks."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5000, 10))
    sel = _fit(x, svd_components=512)
    assert sel.passthrough_, "selector fired on a narrow table"


@pytest.mark.parametrize("n", [2, 3, 7])
def test_tiny_row_counts_stay_valid(n):
    """RamanBench has tables with 7 train rows; k must remain >= 1."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(n, 400))
    sel = _fit(x, svd_components=512)
    k = _rank(sel)
    assert k is None or k >= 1


def test_shipped_config_enables_the_rule():
    """The production config is where this actually ships."""
    import json

    from synthefy_nori.training.config import package_config_path

    cfg = json.loads(pathlib.Path(package_config_path("default_inference.json")).read_text())
    sels = [m["HighDimFeatureSelector"] for m in cfg if "HighDimFeatureSelector" in m]
    assert sels, "no HighDimFeatureSelector in the shipped config"
    for h in sels:
        assert h.get("svd_rows_per_component") == 3, h
        assert h.get("svd_components") == 512, h
