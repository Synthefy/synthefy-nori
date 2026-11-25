"""--icl-filter-model shorthands resolve to a checkpoint without any network access."""

import json

import pytest
import torch

import synthefy_nori.hf as hf
from synthefy_nori.training.config import package_config_path
from synthefy_nori.training.trainer import NoriTrainer
from synthefy_nori.utils.loading import MissingFeatureDecoderError, build_model


@pytest.fixture
def recorded_checkpoint_download_calls(monkeypatch):
    calls = []

    def fake_download_checkpoint(repo_id=None, *, model=None, **kwargs):
        # Keep the real "a size is required" contract so a bare call still fails here.
        if model is None and repo_id is None:
            raise ValueError("download_checkpoint requires model= or an explicit repo_id=")
        calls.append({"model": model, "repo_id": repo_id})
        return f"/cache/{model or repo_id}/nori.pt"

    monkeypatch.setattr(hf, "download_checkpoint", fake_download_checkpoint)
    monkeypatch.setattr(hf, "download_limix", lambda: "/cache/limix.ckpt")
    return calls


@pytest.mark.parametrize("size", list(hf.NORI_MODELS))
def test_size_names_download_that_size(recorded_checkpoint_download_calls, size):
    assert NoriTrainer._resolve_icl_filter_path(size) == f"/cache/{size}/nori.pt"
    assert recorded_checkpoint_download_calls == [{"model": size, "repo_id": None}]


def test_raw_repo_id_is_downloaded_by_repo(recorded_checkpoint_download_calls):
    assert NoriTrainer._resolve_icl_filter_path("org/repo") == "/cache/org/repo/nori.pt"
    assert recorded_checkpoint_download_calls == [{"model": None, "repo_id": "org/repo"}]


def test_limix_and_local_paths_are_unchanged(recorded_checkpoint_download_calls, tmp_path):
    local = tmp_path / "filter.pt"
    local.write_bytes(b"")
    assert NoriTrainer._resolve_icl_filter_path("limix") == "/cache/limix.ckpt"
    assert NoriTrainer._resolve_icl_filter_path(str(local)) == str(local)
    assert recorded_checkpoint_download_calls == []


def _tiny_checkpoint(path, *, omit_feature_decoder):
    with open(package_config_path("model_base.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(nlayers=1, embed_dim=32, hid_dim=64, nhead=2, omit_feature_decoder=omit_feature_decoder)
    for sub_key in ("encoder_config_x", "encoder_config_y"):
        for field in ("embedding_size", "mask_embedding_size"):
            if field in config[sub_key]:
                config[sub_key][field] = 32
    torch.save({"config": config, "state_dict": build_model(config).state_dict()}, path)
    return str(path)


def _bare_trainer():
    trainer = object.__new__(NoriTrainer)
    trainer.device = torch.device("cpu")
    trainer.is_main = False
    return trainer


def test_filter_without_feature_decoder_fails_with_an_actionable_error(monkeypatch, tmp_path):
    # Nori 1.5 ships without the feature decoder that mask_prediction=True needs.
    checkpoint = _tiny_checkpoint(tmp_path / "no-decoder.pt", omit_feature_decoder=True)
    monkeypatch.setattr(hf, "download_checkpoint", lambda repo_id=None, *, model=None, **kwargs: checkpoint)

    with pytest.raises(ValueError, match=r"--icl-filter-model 'nori-1-5' cannot be used") as excinfo:
        _bare_trainer()._init_gpu_icl_filter("nori-1-5")
    message = str(excinfo.value)
    assert "'nori-6m', 'nori-30m' or 'nori-100m'" in message
    assert "mask_prediction" not in message  # internal detail stays in the chained cause
    assert isinstance(excinfo.value.__cause__, MissingFeatureDecoderError)


def test_filter_with_feature_decoder_loads_frozen(tmp_path):
    trainer = _bare_trainer()
    trainer._init_gpu_icl_filter(_tiny_checkpoint(tmp_path / "with-decoder.pt", omit_feature_decoder=False))
    model = trainer._icl_filter_model
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())


def test_missing_feature_decoder_error_is_still_a_value_error():
    # Callers that already catch ValueError from load_model keep working.
    assert issubclass(MissingFeatureDecoderError, ValueError)
