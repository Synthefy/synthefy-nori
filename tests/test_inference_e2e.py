"""End-to-end inference tests.

These tests download the default HuggingFace checkpoint and run a real forward
pass through the model. They are deselected by default (the ``slow`` marker)
because they:

* require network access to fetch the public ``Synthefy/synthefy-tabular``
  checkpoint (no token required; one is used automatically if present in the
  environment, only to raise anonymous rate limits), and
* take ~15s on CPU for the regression case.

Run explicitly with::

    pytest -m slow

Override the checkpoint location without touching HF by setting
``SYNTHEFY_TABULAR_TEST_CHECKPOINT=/abs/path/to/checkpoint.pt`` in the
environment.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


pytestmark = pytest.mark.slow


def _checkpoint_kwargs():
    local = os.environ.get("SYNTHEFY_TABULAR_TEST_CHECKPOINT")
    if local:
        return {"model_path": local}
    # The default repo is public, so anonymous download works. Any HF token in
    # the environment is picked up automatically (it only affects rate limits).
    return {}


def test_regressor_recovers_linear_signal():
    from synthefy_tabular import SynthefyTabularRegressor

    rng = np.random.default_rng(0)
    n_train, n_test, d = 200, 50, 4
    X_train = rng.normal(size=(n_train, d)).astype(np.float32)
    true_w = rng.normal(size=d).astype(np.float32)
    y_train = (X_train @ true_w + rng.normal(scale=0.1, size=n_train)).astype(np.float32)
    X_test = rng.normal(size=(n_test, d)).astype(np.float32)
    y_truth = X_test @ true_w

    model = SynthefyTabularRegressor(**_checkpoint_kwargs())
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    assert pred.shape == (n_test,)
    assert np.all(np.isfinite(pred))
    assert pred.std() > 1e-6, "predictions are degenerate"

    corr = float(np.corrcoef(pred, y_truth)[0, 1])
    assert corr > 0.8, f"expected strong correlation with linear truth, got {corr:.3f}"


def test_regressor_output_types_match_tabpfn_contract():
    from synthefy_tabular import SynthefyTabularRegressor

    rng = np.random.default_rng(0)
    n_train, n_test, d = 200, 50, 4
    X_train = rng.normal(size=(n_train, d)).astype(np.float32)
    true_w = rng.normal(size=d).astype(np.float32)
    y_train = (X_train @ true_w + rng.normal(scale=0.1, size=n_train)).astype(np.float32)
    X_test = rng.normal(size=(n_test, d)).astype(np.float32)

    model = SynthefyTabularRegressor(**_checkpoint_kwargs())
    model.fit(X_train, y_train)

    # All three supported point estimates return a finite per-row vector.
    for output_type in ("mean", "median", "mode"):
        pred = model.predict(X_test, output_type=output_type)
        assert pred.shape == (n_test,), output_type
        assert np.all(np.isfinite(pred)), output_type
