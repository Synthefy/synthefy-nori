from pathlib import Path

import pytest

from synthefy_nori import NoriRegressor, config_path


def test_config_path_points_to_bundled_file():
    path = Path(config_path("default_inference.json"))
    assert path.name == "default_inference.json"
    assert path.exists()


def test_regressor_uses_default_regression_config():
    model = NoriRegressor(model_path="local.pt")
    assert model.model_path == "local.pt"
    assert model.inference_config.endswith("default_inference.json")


def test_regressor_stores_model_variant_verbatim():
    # stored as-is so sklearn clone/get_params round-trips (BaseEstimator contract)
    model = NoriRegressor(model="nori-30m")
    assert model.model == "nori-30m"
    assert model.get_params()["model"] == "nori-30m"
    # Stored verbatim (sklearn contract; no __init__ validation). There is no default:
    # fitting/predicting without a model= (or model_path) raises at checkpoint load.
    assert NoriRegressor().model is None


def test_predict_without_model_or_path_raises_require_model():
    # Neither model= nor model_path -> the require-model guard raises at checkpoint load,
    # before any network call. Covers the public NoriRegressor.fit/predict entry point.
    with pytest.raises(ValueError, match=r"requires model="):
        NoriRegressor().fit([[0.0, 1.0], [1.0, 0.0]], [0.0, 1.0]).predict([[0.5, 0.5]])


def test_resolve_model_path_threads_variant_to_download(monkeypatch):
    from synthefy_nori import api

    seen = {}

    def fake_download(*, model=None, token=None):
        seen["model"] = model
        return "/tmp/resolved.pt"

    monkeypatch.setattr("synthefy_nori.hf.download_checkpoint", fake_download)
    # variant flows through to the download
    assert api._resolve_model_path(None, None, "nori-30m") == "/tmp/resolved.pt"
    assert seen["model"] == "nori-30m"
    # an explicit local checkpoint still wins over the variant
    assert api._resolve_model_path("/my/ckpt.pt", None, "nori-30m") == "/my/ckpt.pt"


def test_predict_rejects_unsupported_output_types():
    # output_type is validated before the checkpoint is loaded, so these paths
    # exercise the guardrails without any model weights.
    model = NoriRegressor(model_path="local.pt")

    # "main" is a recognized output_type name that Nori does not implement.
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
