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

    for output_type in ("quantiles", "main", "full"):
        with pytest.raises(NotImplementedError):
            model.predict([[0.0, 1.0]], output_type=output_type)

    with pytest.raises(ValueError):
        model.predict([[0.0, 1.0]], output_type="bogus")

    with pytest.raises(ValueError):
        model.predict([[0.0, 1.0]], output_type="mean", quantiles=[0.5])
