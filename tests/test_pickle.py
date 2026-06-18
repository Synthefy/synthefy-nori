"""Regression tests for stdlib-pickle support (GitHub issue #45).

A fitted ``NoriRegressor`` used to fail ``pickle.dumps`` because the
preprocessing pipeline built ``FunctionTransformer`` steps from *local lambdas*
in ``RebalanceFeatureDistribution._set``. Stdlib pickle can only serialize
functions by reference, so any framework that persists models via stdlib pickle
(AutoGluon, joblib's default, multiprocessing/Ray) could not save the model.

The fix replaces those lambdas with module-level helpers. The fast tests below
exercise the exact class that owned the lambdas without needing model weights;
the ``slow`` test covers the full fit -> pickle -> predict round-trip from the
issue's reproduction.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from synthefy_nori.inference import preprocess
from synthefy_nori.inference.preprocess import RebalanceFeatureDistribution


def _sample(n: int = 64, d: int = 5) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, d)).astype(np.float32)


def test_function_transformer_helpers_are_module_level():
    """The helpers must be importable (not local lambdas) to pickle by reference.

    This is the root-cause guard: if someone reintroduces a local lambda in a
    FunctionTransformer step, the round-trip tests below break, but this also
    documents the contract that these names exist at module scope.
    """
    x = _sample()
    for fn in (preprocess._inf_to_nan, preprocess._identity,
               preprocess._shift_to_nonnegative, preprocess._add_epsilon):
        assert fn.__module__ == preprocess.__name__
        assert "<lambda>" not in fn.__qualname__
        # Each helper pickles by reference and survives a round-trip.
        assert pickle.loads(pickle.dumps(fn)) is fn

    np.testing.assert_array_equal(preprocess._identity(x), x)
    # _inf_to_nan maps +/-inf to NaN (its raison d'etre) while leaving finite
    # values untouched, so a later imputer can fill them.
    assert np.all(np.isnan(preprocess._inf_to_nan(np.array([np.inf, -np.inf]))))
    np.testing.assert_array_equal(preprocess._inf_to_nan(x), x)
    assert np.all(preprocess._shift_to_nonnegative(x) >= 0.0)


# worker_tag configs whose pipelines build FunctionTransformer steps. The first
# three carry the lambdas that caused the bug; ``None`` covers the standalone
# identity transformer branch.
@pytest.mark.parametrize(
    ("worker_tags", "svd_tag"),
    [
        (["logNormal"], None),
        (["power"], None),
        (["quantile_uniform_5"], "svd"),  # svd branch builds its own transformers
        ([None], None),
    ],
)
def test_fitted_rebalance_pickles_and_round_trips(worker_tags, svd_tag):
    x = _sample()
    rebalance = RebalanceFeatureDistribution(worker_tags=worker_tags, svd_tag=svd_tag)
    rebalance.fit(x, categorical_features=[], seed=0)

    expected = rebalance.worker.transform(x)

    # The actual fix: stdlib pickle (not cloudpickle) must succeed.
    restored = pickle.loads(pickle.dumps(rebalance))

    np.testing.assert_allclose(restored.worker.transform(x), expected, rtol=1e-6, atol=1e-6)


def test_rebalance_with_categorical_features_pickles():
    """The categorical/continuous split chooses different branches in ``_set``."""
    x = _sample()
    rebalance = RebalanceFeatureDistribution(worker_tags=["logNormal"], svd_tag="svd")
    rebalance.fit(x, categorical_features=[0, 2], seed=7)

    expected = rebalance.worker.transform(x)
    restored = pickle.loads(pickle.dumps(rebalance))
    np.testing.assert_allclose(restored.worker.transform(x), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.slow
def test_fitted_regressor_pickles_and_predicts_identically():
    """End-to-end reproduction from issue #45: fit, stdlib-pickle, predict.

    Requires the public checkpoint (downloaded or via
    ``SYNTHEFY_NORI_TEST_CHECKPOINT``), so it carries the ``slow`` marker like
    the other e2e tests.
    """
    from synthefy_nori import NoriRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 5)).astype(np.float32)
    w = rng.normal(size=5).astype(np.float32)
    y = (X @ w).astype(np.float32)
    X_test = rng.normal(size=(32, 5)).astype(np.float32)

    checkpoint = os.environ.get("SYNTHEFY_NORI_TEST_CHECKPOINT")
    kwargs = {"model_path": checkpoint} if checkpoint else {}

    model = NoriRegressor(**kwargs)
    model.fit(X, y)
    expected = model.predict(X_test)

    blob = pickle.dumps(model)  # used to raise AttributeError: Can't pickle local object
    restored = pickle.loads(blob)

    np.testing.assert_allclose(restored.predict(X_test), expected, rtol=1e-5, atol=1e-5)
