"""Smoke tests for the shipped text-features example.

The fast test guards the synthetic-dataset builder and the public API surface the
example relies on (no checkpoint / encoder download). The slow test runs the
example end-to-end against the real checkpoint + MiniLM encoder and asserts the
text column actually helps — it is skipped when sentence-transformers is absent.
"""
import importlib.util
import pathlib

import pytest

EXAMPLE = (pathlib.Path(__file__).resolve().parents[1]
           / "examples" / "text_features_synthetic.py")


def _load():
    spec = importlib.util.spec_from_file_location("text_features_synthetic", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_dataset_shapes_and_columns():
    module = _load()
    df, y = module.make_dataset(20, seed=0)
    assert list(df.columns) == ["x1", "x2", "brand", "review"]
    assert len(df) == len(y) == 20
    assert df["review"].str.startswith("Customer review:").all()
    assert set(df["brand"]).issubset({"acme", "globex", "initech"})


@pytest.mark.slow
def test_text_example_text_helps_end_to_end():
    pytest.importorskip("sentence_transformers")
    module = _load()
    r_tab, r_text = module.run(n_train=150, n_test=50, svd_dim=16, seed=0)
    # the review's sentiment drives a large target term the tabular columns
    # cannot see, so the text feature must improve held-out R2
    assert r_text > r_tab
