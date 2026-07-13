from synthefy_nori import hf


def test_hf_defaults_are_public_strings():
    assert hf.DEFAULT_MODEL_REPO_ID == "Synthefy/Nori"
    assert hf.DEFAULT_CHECKPOINT_FILENAME.endswith(".pt")


def test_model_variant_registry_resolution():
    # friendly variant names -> HF repo ids
    assert hf.resolve_model_repo("nori-30m") == "Synthefy/Nori-30M"
    assert hf.resolve_model_repo("nori") == hf.DEFAULT_MODEL_REPO_ID       # default 6M base
    assert hf.resolve_model_repo("nori-6m") == hf.DEFAULT_MODEL_REPO_ID    # explicit base alias
    assert hf.resolve_model_repo(None) == hf.DEFAULT_MODEL_REPO_ID         # None -> default base
    assert hf.resolve_model_repo("Synthefy/Custom-Repo") == "Synthefy/Custom-Repo"  # raw id passes through
    assert {"nori", "nori-6m", "nori-30m"} <= set(hf.NORI_MODELS)


def test_download_checkpoint_model_overrides_repo(monkeypatch):
    seen = {}

    def fake_hub_download(repo_id, filename, **kwargs):
        seen["repo_id"] = repo_id
        return "/tmp/nori.pt"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)
    hf.download_checkpoint(model="nori-30m", token=False)
    assert seen["repo_id"] == "Synthefy/Nori-30M"
