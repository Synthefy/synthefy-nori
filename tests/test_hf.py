import pytest

from synthefy_nori import hf


def test_hf_defaults_are_public_strings():
    assert hf.DEFAULT_MODEL_REPO_ID == "Synthefy/Nori"
    assert hf.DEFAULT_CHECKPOINT_FILENAME.endswith(".pt")


def test_model_variant_registry_resolution():
    # friendly variant names -> HF repo ids
    assert hf.resolve_model_repo("nori-30m") == "Synthefy/Nori-30M"
    assert hf.resolve_model_repo("nori-100m") == "Synthefy/Nori-100M"
    assert hf.resolve_model_repo("nori-6m") == hf.DEFAULT_MODEL_REPO_ID  # ~6M base
    # A size is required -- None and a bare "nori" both raise (there is no default).
    for missing in (None, "nori"):
        with pytest.raises(ValueError, match=r"model is required"):
            hf.resolve_model_repo(missing)
    assert hf.resolve_model_repo("Synthefy/Custom-Repo") == "Synthefy/Custom-Repo"  # raw id passes through
    assert set(hf.NORI_MODELS) == {"nori-6m", "nori-30m", "nori-100m"}
    assert "nori" not in hf.NORI_MODELS


def test_thinking_variant_is_rejected_not_downloaded():
    # Thinking is hosted-API only: resolve_model_repo must raise a clear error rather than fall
    # through to a raw-repo lookup that 404s with an opaque message.
    for name in (
        "nori-30m-thinking",
        "nori-30m-thinking-medium",
        "synthefy/nori-30m-thinking-high",
    ):
        assert hf._is_thinking_model(name)
        with pytest.raises(ValueError, match=r"Thinking.*hosted Synthefy API"):
            hf.resolve_model_repo(name)
    # non-thinking selectors are unaffected
    assert not hf._is_thinking_model("nori-30m")
    assert hf.resolve_model_repo("nori-30m") == "Synthefy/Nori-30M"


def test_download_checkpoint_rejects_thinking_variant():
    # The guard fires through the download entry point too (before any network call).
    with pytest.raises(ValueError, match=r"Thinking.*hosted Synthefy API"):
        hf.download_checkpoint(model="nori-30m-thinking-medium", token=False)


def test_download_checkpoint_requires_a_size():
    # No model= and no repo_id -> raise (there is no default), before any network call.
    with pytest.raises(ValueError, match=r"requires model="):
        hf.download_checkpoint(token=False)


def test_download_checkpoint_model_overrides_repo(monkeypatch):
    seen = {}

    def fake_hub_download(repo_id, filename, **kwargs):
        seen["repo_id"] = repo_id
        return "/tmp/nori.pt"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)
    hf.download_checkpoint(model="nori-30m", token=False)
    assert seen["repo_id"] == "Synthefy/Nori-30M"


@pytest.fixture
def hub_downloads(monkeypatch):
    seen = []

    def fake_hub_download(repo_id, filename, **kwargs):
        seen.append(repo_id)
        return f"/cache/{repo_id}/{filename}"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hub_download)
    # download_checkpoint also fetches the config.json sidecar from the same repo, so
    # assert on the set of repos touched rather than on the number of calls.
    return seen


def test_download_cli_defaults_to_base_repo(hub_downloads, capsys):
    hf.download_cli([])
    assert set(hub_downloads) == {"Synthefy/Nori"}
    assert capsys.readouterr().out.strip() == "/cache/Synthefy/Nori/nori.pt"


@pytest.mark.parametrize("size", list(hf.NORI_MODELS))
def test_download_cli_model_selects_size(hub_downloads, size):
    hf.download_cli(["--model", size])
    assert set(hub_downloads) == {hf.NORI_MODELS[size]}


def test_download_cli_repo_id_still_works(hub_downloads):
    hf.download_cli(["--repo-id", "org/repo"])
    assert set(hub_downloads) == {"org/repo"}


@pytest.mark.parametrize(
    "argv",
    [["--model", "nori-30m", "--repo-id", "org/repo"], ["--model", "nori-7b"]],
)
def test_download_cli_rejects_bad_model_args(hub_downloads, argv):
    with pytest.raises(SystemExit):
        hf.download_cli(argv)
    assert hub_downloads == []
