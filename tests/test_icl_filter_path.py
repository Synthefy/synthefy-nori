"""--icl-filter-model shorthands resolve to a checkpoint without any network access."""

import pytest

import synthefy_nori.hf as hf
from synthefy_nori.training.trainer import NoriTrainer


@pytest.fixture
def downloads(monkeypatch):
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


def test_hf_is_an_alias_for_the_base_checkpoint(downloads):
    assert NoriTrainer._resolve_icl_filter_path("hf") == "/cache/nori-6m/nori.pt"
    assert downloads == [{"model": "nori-6m", "repo_id": None}]


@pytest.mark.parametrize("size", list(hf.NORI_MODELS))
def test_size_names_download_that_size(downloads, size):
    assert NoriTrainer._resolve_icl_filter_path(size) == f"/cache/{size}/nori.pt"
    assert downloads == [{"model": size, "repo_id": None}]


def test_raw_repo_id_is_downloaded_by_repo(downloads):
    assert NoriTrainer._resolve_icl_filter_path("org/repo") == "/cache/org/repo/nori.pt"
    assert downloads == [{"model": None, "repo_id": "org/repo"}]


def test_limix_and_local_paths_are_unchanged(downloads, tmp_path):
    local = tmp_path / "filter.pt"
    local.write_bytes(b"")
    assert NoriTrainer._resolve_icl_filter_path("limix") == "/cache/limix.ckpt"
    assert NoriTrainer._resolve_icl_filter_path(str(local)) == str(local)
    assert downloads == []
