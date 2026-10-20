"""Interpretability adapters: sklearn-estimator compliance (no weights needed) +
an end-to-end shapiq/PDP/feature-selection smoke test behind a stub predictor."""

import numpy as np
import pytest

from synthefy_nori import NoriRegressor


def test_regressor_is_sklearn_estimator_and_clones():
    from sklearn.base import clone, is_regressor

    m = NoriRegressor(model_path="local.pt", augmentations=("yj",))
    assert is_regressor(m)  # RegressorMixin -> PDP/SFS treat it as regressor
    params = m.get_params()
    assert params["model_path"] == "local.pt"
    c = clone(m)  # required by SequentialFeatureSelector/cross_val
    assert c.get_params()["model_path"] == "local.pt"
    assert c.inference_config is None


def test_imputation_explainer_requires_shapiq_or_runs():
    """If shapiq is installed, the adapter explains a prediction from a stub
    regressor (no model weights / GPU needed); otherwise it raises a clear hint."""
    from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    w = np.array([2.0, -1.0, 0.5, 0.0])

    class _StubRegressor:  # behaves like a fitted regressor
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


class _FrameFittedRegressor:
    """Behave like a NoriRegressor fitted on a DataFrame: refuse arrays.

    Predicts 2 * x for numbers plus 3 for each row whose nation is "Dutch", and
    declares the integer column ``team`` categorical.
    """

    feature_names_in_ = np.asarray(["x", "nation", "team"], dtype=object)

    def get_params(self):
        return {"categorical_columns": ["team"]}

    def predict(self, X):
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        assert X["x"].dtype == np.float64 and X["team"].dtype == np.int64
        return 2.0 * X["x"].to_numpy() + 3.0 * (X["nation"] == "Dutch").to_numpy()


def _get_frame_with_text():
    """Return background rows with a float, a text and an integer-coded column."""
    import pandas as pd

    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "nation": ["British", "British", "British", "Dutch", "German", "British"],
            "team": [7, 7, 9, 9, 7, 9],
        }
    )


def test_predict_function_rebuilds_the_fitted_frame_with_its_dtypes():
    from synthefy_nori.interpretability.shapiq import _get_predict_function

    background = _get_frame_with_text()
    predict = _get_predict_function(_FrameFittedRegressor(), background)
    rows = np.asarray([[1.0, "Dutch", 7], [2.0, "British", 9]], dtype=object)
    assert predict(rows).tolist() == [5.0, 4.0]


def test_background_keeps_text_and_stays_float_when_numeric():
    from synthefy_nori.interpretability.shapiq import _get_background_array

    with_text = _get_background_array(_get_frame_with_text())
    assert with_text.dtype == object and with_text[3, 1] == "Dutch"
    assert _get_background_array(np.ones((3, 2), dtype=int)).dtype == np.float64


def test_text_and_declared_columns_are_categorical_features():
    from synthefy_nori.interpretability.shapiq import _get_categorical_feature_indices

    indices = _get_categorical_feature_indices(_FrameFittedRegressor(), _get_frame_with_text())
    assert indices == [1, 2]
    numeric_only = _get_categorical_feature_indices(object(), np.ones((3, 2)))
    assert numeric_only == []


def test_explainer_handles_a_text_column_and_a_frame_fitted_model():
    """Shapley values over a text column add up to the prediction minus the baseline."""
    pytest.importorskip("shapiq")
    from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

    background = _get_frame_with_text()
    model = _FrameFittedRegressor()
    explainer = get_nori_imputation_explainer(model, background, index="SV", max_order=1)
    row = np.asarray([[4.0, "Dutch", 9]], dtype=object)
    sv = explainer.explain(row, budget=8)
    values = {key[0]: value for key, value in sv.dict_values.items() if len(key) == 1}
    prediction = model.predict(_get_frame_with_text().iloc[[0]].assign(x=4.0, nation="Dutch", team=9))[0]
    assert sv.baseline_value + sum(values.values()) == pytest.approx(prediction)
    # The text feature moved the prediction by its full effect (British -> Dutch).
    assert values[1] == pytest.approx(3.0)


def test_pdp_and_feature_selection_wrappers_plumbing():
    """The PDP / feature-selection adapters are sklearn passthroughs — validate
    their plumbing with a fast sklearn estimator (model weights not needed; the
    NoriRegressor sklearn-compliance is covered by the clone test above)."""
    matplotlib = pytest.importorskip("matplotlib")  # PDP needs it; skip in core CI (dev extra only)
    matplotlib.use("Agg")
    from sklearn.linear_model import LinearRegression

    from synthefy_nori.interpretability.feature_selection import feature_selection
    from synthefy_nori.interpretability.pdp import partial_dependence_plots

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    y = X @ np.array([3.0, 0.0, -2.0, 0.0]) + rng.normal(scale=0.1, size=60)
    est = LinearRegression().fit(X, y)

    disp = partial_dependence_plots(est, X, features=[0, 2], kind="average")
    assert disp is not None

    res = feature_selection(est, X, y, n_features_to_select=2, cv=3, feature_names=list("abcd"))
    assert len(res.selected_indices) == 2
    assert set(res.selected_indices) == {0, 2}  # the two signal features
    assert res.selected_names == ["a", "c"]
    assert res.selected_score_mean >= res.baseline_score_mean - 0.05
