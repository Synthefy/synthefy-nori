"""End-to-end inference tests.

These tests download the default HuggingFace checkpoint and run a real forward
pass through the model. They are deselected by default (the ``slow`` marker)
because they:

* require network access to fetch the public ``Synthefy/Nori``
  checkpoint (no token required; one is used automatically if present in the
  environment, only to raise anonymous rate limits), and
* take ~15s on CPU for the regression case.

Run explicitly with::

    pytest -m slow

Override the checkpoint location without touching HF by setting
``SYNTHEFY_NORI_TEST_CHECKPOINT=/abs/path/to/checkpoint.pt`` in the
environment.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch

from synthefy_nori import NoriRegressor


pytestmark = pytest.mark.slow


def _checkpoint_kwargs():
    local = os.environ.get("SYNTHEFY_NORI_TEST_CHECKPOINT")
    if local:
        return {"model_path": local}
    # model= is now required (the bare default was retired). "nori-6m" resolves to
    # the public HF repo, so anonymous download works. Any HF token in the
    # environment is picked up automatically (it only affects rate limits).
    return {"model": "nori-6m"}


def _linear_signal_data():
    rng = np.random.default_rng(0)
    n_train, n_test, d = 200, 50, 4
    X_train = rng.normal(size=(n_train, d)).astype(np.float32)
    true_w = rng.normal(size=d).astype(np.float32)
    y_train = (X_train @ true_w + rng.normal(scale=0.1, size=n_train)).astype(np.float32)
    X_test = rng.normal(size=(n_test, d)).astype(np.float32)
    return X_train, y_train, X_test, X_test @ true_w


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS-capable macOS host"
)
def test_default_devices_run_nori_and_minilm_on_mps():
    """One smoke covers automatic model and named-text placement end to end."""
    X_train = pd.DataFrame(
        {
            "amount": np.linspace(1.0, 8.0, 8, dtype=np.float32),
            "description": [f"customer transaction {i}" for i in range(8)],
        }
    )
    y_train = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    X_test = pd.DataFrame(
        {"amount": [2.5, 7.5], "description": ["small order", "large order"]}
    )

    model = NoriRegressor(
        **_checkpoint_kwargs(), text_columns=["description"], svd_dim=4
    ).fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert model.device_ == torch.device("mps")
    assert predictions.shape == (2,)
    assert np.all(np.isfinite(predictions))
    assert model._predictor.mix_precision is False


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS-capable macOS host"
)
def test_mps_regressor_matches_cpu_quality():
    """MPS must preserve CPU task quality, not merely return finite numbers."""
    X_train, y_train, X_test, y_truth = _linear_signal_data()

    cpu_model = NoriRegressor(**_checkpoint_kwargs(), device="cpu").fit(X_train, y_train)
    cpu_predictions = cpu_model.predict(X_test)
    cpu_corr = float(np.corrcoef(cpu_predictions, y_truth)[0, 1])
    cpu_rmse = float(np.sqrt(np.mean((cpu_predictions - y_truth) ** 2)))

    mps_model = NoriRegressor(**_checkpoint_kwargs()).fit(X_train, y_train)
    mps_predictions = mps_model.predict(X_test)
    mps_corr = float(np.corrcoef(mps_predictions, y_truth)[0, 1])
    mps_rmse = float(np.sqrt(np.mean((mps_predictions - y_truth) ** 2)))

    assert mps_model.device_ == torch.device("mps")
    assert mps_model._predictor.mix_precision is False
    assert mps_corr > 0.8, f"expected strong correlation with linear truth, got {mps_corr:.3f}"
    assert mps_corr >= cpu_corr - 0.02, f"MPS correlation {mps_corr:.3f} trails CPU {cpu_corr:.3f}"
    assert mps_rmse <= cpu_rmse * 1.1, f"MPS RMSE {mps_rmse:.3f} exceeds CPU {cpu_rmse:.3f}"


def test_regressor_recovers_linear_signal():
    X_train, y_train, X_test, y_truth = _linear_signal_data()

    model = NoriRegressor(**_checkpoint_kwargs())
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    assert pred.shape == (len(X_test),)
    assert np.all(np.isfinite(pred))
    assert pred.std() > 1e-6, "predictions are degenerate"

    corr = float(np.corrcoef(pred, y_truth)[0, 1])
    assert corr > 0.8, f"expected strong correlation with linear truth, got {corr:.3f}"


def test_regressor_output_types_match_tabpfn_contract():
    rng = np.random.default_rng(0)
    n_train, n_test, d = 200, 50, 4
    X_train = rng.normal(size=(n_train, d)).astype(np.float32)
    true_w = rng.normal(size=d).astype(np.float32)
    y_train = (X_train @ true_w + rng.normal(scale=0.1, size=n_train)).astype(np.float32)
    X_test = rng.normal(size=(n_test, d)).astype(np.float32)

    model = NoriRegressor(**_checkpoint_kwargs())
    model.fit(X_train, y_train)

    # All three supported point estimates return a finite per-row vector.
    for output_type in ("mean", "median", "mode"):
        pred = model.predict(X_test, output_type=output_type)
        assert pred.shape == (n_test,), output_type
        assert np.all(np.isfinite(pred)), output_type


def test_regressor_distribution_outputs():
    rng = np.random.default_rng(0)
    n_train, n_test, d = 200, 50, 4
    X_train = rng.normal(size=(n_train, d)).astype(np.float32)
    true_w = rng.normal(size=d).astype(np.float32)
    y_train = (X_train @ true_w + rng.normal(scale=0.1, size=n_train)).astype(np.float32)
    X_test = rng.normal(size=(n_test, d)).astype(np.float32)

    model = NoriRegressor(**_checkpoint_kwargs())
    model.fit(X_train, y_train)

    # output_type="full": a per-row quantile function plus the matching levels.
    dist = model.predict(X_test, output_type="full")
    Q, taus, mean = dist["quantiles"], dist["taus"], dist["mean"]
    K = Q.shape[1]
    assert Q.shape == (n_test, K)
    assert taus.shape == (K,)
    assert mean.shape == (n_test,)
    assert np.all(np.isfinite(Q))
    # Quantiles are sorted to a valid (monotone non-decreasing) quantile function.
    assert np.all(np.diff(Q, axis=1) >= -1e-6)
    assert taus[0] > 0.0 and taus[-1] < 1.0

    # output_type="full" mean matches the default point estimate. The two come
    # from independent forward passes under mixed precision, so they agree only
    # up to GPU/autocast nondeterminism rather than bit-for-bit.
    np.testing.assert_allclose(mean, model.predict(X_test, output_type="mean"), rtol=5e-3, atol=5e-3)

    # output_type="quantiles": shape (n_levels, n_samples), ordered across levels.
    levels = [0.1, 0.5, 0.9]
    qs = model.predict(X_test, output_type="quantiles", quantiles=levels)
    assert qs.shape == (len(levels), n_test)
    assert np.all(np.isfinite(qs))
    assert np.all(qs[0] <= qs[1] + 1e-6) and np.all(qs[1] <= qs[2] + 1e-6)
