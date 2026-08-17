from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from synthefy_nori import NoriRegressor, config_path
from synthefy_nori import api
from synthefy.nori_client import _resolve_text_device


def _set_accelerators(monkeypatch, *, cuda=False, mps=False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_default_device_prefers_cuda_over_mps(monkeypatch):
    _set_accelerators(monkeypatch, cuda=True, mps=True)
    assert api._default_device() == torch.device("cuda:0")


def test_default_device_uses_mps_when_cuda_is_unavailable(monkeypatch):
    _set_accelerators(monkeypatch, mps=True)
    assert api._default_device() == torch.device("mps")


def test_default_device_falls_back_to_cpu(monkeypatch):
    _set_accelerators(monkeypatch)
    assert api._default_device() == torch.device("cpu")


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_local_and_client_auto_device_policies_match(
    monkeypatch, cuda, mps, expected
):
    _set_accelerators(monkeypatch, cuda=cuda, mps=mps)
    assert api._default_device().type == expected
    assert _resolve_text_device(None) == expected


def test_explicit_device_does_not_probe_auto_detection(monkeypatch):
    monkeypatch.setattr(
        api,
        "_default_device",
        lambda: pytest.fail("an explicit device must skip automatic detection"),
    )
    assert api._as_device("cpu") == torch.device("cpu")


def test_explicit_mps_fails_early_when_pytorch_lacks_support(monkeypatch):
    _set_accelerators(monkeypatch)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)

    with pytest.raises(RuntimeError, match="PyTorch build does not include MPS"):
        api._as_device("mps")


def test_explicit_mps_is_accepted_when_available(monkeypatch):
    _set_accelerators(monkeypatch, mps=True)
    assert api._as_device("mps") == torch.device("mps")


def test_fit_resolves_and_reuses_device_for_named_text_encoder(monkeypatch):
    captured = {}

    class FakePreprocessor:
        def __init__(self, text_columns, **kwargs):
            captured.update(kwargs)
            self.text_columns_ = list(text_columns)

        def fit_transform(self, frame):
            return np.zeros((len(frame), 2), dtype=np.float32)

    monkeypatch.setattr(api, "_as_device", lambda device: torch.device("mps"))
    monkeypatch.setattr(api, "MultimodalPreprocessor", FakePreprocessor)

    model = NoriRegressor(model_path="local.pt", text_columns=["review"])
    model.fit(pd.DataFrame({"review": ["good", "bad"]}), [1.0, 0.0])

    assert model.device_ == torch.device("mps")
    assert model.text_device_ == torch.device("mps")
    assert captured["device"] == torch.device("mps")


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
