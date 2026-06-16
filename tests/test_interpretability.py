"""Interpretability adapters: sklearn-estimator compliance (no weights needed) +
an end-to-end shapiq/PDP/feature-selection smoke test behind a stub predictor."""

import numpy as np
import pytest

from synthefy_nori import NoriRegressor


def test_regressor_is_sklearn_estimator_and_clones():
    from sklearn.base import clone, is_regressor

    m = NoriRegressor(model_path="local.pt", augmentations=("yj",))
    assert is_regressor(m)                       # RegressorMixin -> PDP/SFS treat it as regressor
    params = m.get_params()
    assert params["model_path"] == "local.pt"
    c = clone(m)                                 # required by SequentialFeatureSelector/cross_val
    assert c.get_params()["model_path"] == "local.pt"
    assert c.inference_config.endswith("reg_allordinal_poly10_adaptive_svd256.json")


def test_imputation_explainer_requires_shapiq_or_runs():
    """If shapiq is installed, the adapter explains a prediction from a stub
    regressor (no model weights / GPU needed); otherwise it raises a clear hint."""
    from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    w = np.array([2.0, -1.0, 0.5, 0.0])

    class _StubRegressor:                        # behaves like a fitted regressor
        def predict(self, x):
            return np.asarray(x) @ w

    try:
        import shapiq  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="shapiq"):
            get_nori_imputation_explainer(_StubRegressor(), X)
        pytest.skip("shapiq not installed")

    ex = get_nori_imputation_explainer(_StubRegressor(), X, index="SV", max_order=1)
    sv = ex.explain(X[:1], budget=32)
    vals = np.asarray(sv.get_n_order_values(1) if hasattr(sv, "get_n_order_values") else sv.values)
    # feature 3 has zero weight -> smallest |attribution|; feature 0 the largest.
    assert np.argmin(np.abs(vals)) == 3
    assert np.argmax(np.abs(vals)) == 0


def test_pdp_and_feature_selection_wrappers_plumbing():
    """The PDP / feature-selection adapters are sklearn passthroughs — validate
    their plumbing with a fast sklearn estimator (model weights not needed; the
    NoriRegressor sklearn-compliance is covered by the clone test above)."""
    import matplotlib
    matplotlib.use("Agg")
    from sklearn.linear_model import LinearRegression

    from synthefy_nori.interpretability.feature_selection import feature_selection
    from synthefy_nori.interpretability.pdp import partial_dependence_plots

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4)); y = X @ np.array([3.0, 0.0, -2.0, 0.0]) + rng.normal(scale=0.1, size=60)
    est = LinearRegression().fit(X, y)

    disp = partial_dependence_plots(est, X, features=[0, 2], kind="average")
    assert disp is not None

    res = feature_selection(est, X, y, n_features_to_select=2, cv=3, feature_names=list("abcd"))
    assert len(res.selected_indices) == 2
    assert set(res.selected_indices) == {0, 2}            # the two signal features
    assert res.selected_names == ["a", "c"]
    assert res.selected_score_mean >= res.baseline_score_mean - 0.05
