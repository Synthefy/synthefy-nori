from pathlib import Path

import pytest

from synthefy_nori import NoriRegressor, config_path


def test_config_path_points_to_bundled_file():
    path = Path(config_path("reg_default_noretrieval.json"))
    assert path.name == "reg_default_noretrieval.json"
    assert path.exists()


def test_regressor_uses_default_regression_config():
    model = NoriRegressor(model_path="local.pt")
    assert model.model_path == "local.pt"
    assert model.inference_config.endswith("reg_allordinal_poly10_adaptive_svd256.json")


def test_predict_rejects_unsupported_output_types():
    # output_type is validated before the checkpoint is loaded, so these paths
    # exercise the TabPFN-contract guardrails without any model weights.
    model = NoriRegressor(model_path="local.pt")

    # 'main' is part of the TabPFN contract but unsupported here.
    with pytest.raises(NotImplementedError):
        model.predict([[0.0, 1.0]], output_type="main")

    with pytest.raises(ValueError):
        model.predict([[0.0, 1.0]], output_type="bogus")

    with pytest.raises(ValueError):
        model.predict([[0.0, 1.0]], output_type="mean", quantiles=[0.5])


def test_distribution_outputs_require_fit_and_valid_levels():
    # 'quantiles'/'full' are supported but need fit() first; the fit guard and
    # the quantile-level validation both fire before any weights are loaded.
    model = NoriRegressor(model_path="local.pt")

    for output_type in ("quantiles", "full"):
        with pytest.raises(ValueError):
            model.predict([[0.0, 1.0]], output_type=output_type, quantiles=[0.5])

    # Once "fit", an empty/out-of-range quantiles list is rejected.
    model.X_train_ = [[0.0, 1.0]]
    model.y_train_ = [0.0]
    model.y_mean_, model.y_std_ = 0.0, 1.0
    with pytest.raises(ValueError):
        model.predict([[0.0, 1.0]], output_type="quantiles", quantiles=[])
    with pytest.raises(ValueError):
        model.predict([[0.0, 1.0]], output_type="quantiles", quantiles=[1.5])
