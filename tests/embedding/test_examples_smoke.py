"""Smoke tests for the shipped embedding examples.

These exist because the examples were previously never executed by CI. That gap
let an internal-only ``NoriRegressor(compile_model=...)`` kwarg ship in a public
example, which crashes at construction on the public package. The fast test
guards the exact public construction API the examples/README rely on; the slow
tests import each example module and run its own ``extract_embeddings`` end-to-end
on tiny data against the real checkpoint.
"""

import importlib.util
import pathlib

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from synthefy_nori import NoriEmbedding, NoriRegressor

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples"
EXAMPLE_NAMES = ["embedding_synthetic", "embedding_tabarena"]


def _load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_construction_api_the_examples_use():
    """The exact public construction the examples/README rely on must not depend
    on any internal-only kwarg. Regression guard for the ``compile_model`` leak.

    Fast: construction is lazy (no checkpoint download), so this runs in the
    default (non-slow) suite and blocks the merge before an example can ship
    with an internal-only constructor argument again.
    """
    NoriRegressor(model="nori-30m")
    NoriEmbedding(n_fold=5, shuffle=True, random_state=0, model=NoriRegressor(model="nori-30m"))


@pytest.mark.slow
@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_example_imports(name):
    """Each shipped example imports cleanly against the public package."""
    assert hasattr(_load_example(name), "extract_embeddings")


@pytest.mark.slow
@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_example_extract_embeddings_end_to_end(name):
    """Run the example's own ``extract_embeddings`` on tiny synthetic data against
    the real checkpoint — exercises the actual shipped example code path."""
    module = _load_example(name)
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(60, 5)).astype(np.float32)
    y = (np.sin(X[:, 0]) + X[:, 1] ** 2).astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    Z_train, Z_test, native = module.extract_embeddings(X_train, y_train, X_test)

    assert Z_train.shape[0] == len(X_train)
    assert Z_test.shape[0] == len(X_test)
    assert Z_train.shape[1] == Z_test.shape[1] > 0
    assert np.asarray(native).shape[0] == len(X_test)
