from pathlib import Path

from synthefy_tabular import SynthefyTabularRegressor, config_path


def test_config_path_points_to_bundled_file():
    path = Path(config_path("reg_default_noretrieval.json"))
    assert path.name == "reg_default_noretrieval.json"
    assert path.exists()


def test_regressor_uses_default_regression_config():
    model = SynthefyTabularRegressor(model_path="local.pt")
    assert model.model_path == "local.pt"
    assert model.inference_config.endswith("reg_allordinal_poly10_noretrieval.json")
