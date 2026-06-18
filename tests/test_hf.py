from synthefy_nori import hf


def test_hf_defaults_are_public_strings():
    assert hf.DEFAULT_MODEL_REPO_ID == "Synthefy/Nori"
    assert hf.DEFAULT_CHECKPOINT_FILENAME.endswith(".pt")
