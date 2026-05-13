from synthefy_tabular import hf


def test_hf_defaults_are_public_strings():
    assert hf.DEFAULT_MODEL_REPO_ID == "Synthefy/synthefy-tabular"
    assert hf.DEFAULT_CHECKPOINT_FILENAME.endswith(".pt")
