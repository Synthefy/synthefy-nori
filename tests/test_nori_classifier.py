"""End-to-end tests for the re-exposed NoriClassifier.

These download the default HuggingFace checkpoint and run a real forward pass
through the model's classification head, so they carry the ``slow`` marker (the
same convention as ``test_inference_e2e.py``). They are deselected by default.

Run explicitly with::

    pytest -m slow tests/test_nori_classifier.py

Override the checkpoint location without touching HF by setting
``SYNTHEFY_NORI_TEST_CHECKPOINT=/abs/path/to/checkpoint.pt``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


pytestmark = pytest.mark.slow


def _checkpoint_kwargs():
    local = os.environ.get("SYNTHEFY_NORI_TEST_CHECKPOINT")
    if local:
        return {"model_path": local}
    return {}


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC for binary labels (no sklearn dependency)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    assert n_pos > 0 and n_neg > 0, "AUROC needs both classes present"
    sum_pos_ranks = ranks[labels == 1].sum()
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _separable_binary(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_train, n_test, d = 200, 50, 3
    X_train = rng.normal(size=(n_train, d)).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(np.int64)
    X_test = rng.normal(size=(n_test, d)).astype(np.float32)
    y_test = (X_test[:, 0] > 0).astype(np.int64)
    return X_train, y_train, X_test, y_test


def test_classifier_predict_proba_is_valid_and_separates():
    from synthefy_nori import NoriClassifier

    X_train, y_train, X_test, y_test = _separable_binary()

    model = NoriClassifier(device="cpu", **_checkpoint_kwargs())
    model.fit(X_train, y_train)

    assert model.n_classes_ == 2
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))

    proba = model.predict_proba(X_test)
    assert proba.shape == (X_test.shape[0], 2)
    assert np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    auroc = _auroc(proba[:, 1], y_test)
    assert auroc > 0.7, f"expected easy separation, got AUROC={auroc:.3f}"

    preds = model.predict(X_test)
    assert preds.shape == (X_test.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_infer_classification_task_returns_proba():
    from synthefy_nori import infer

    X_train, y_train, X_test, y_test = _separable_binary(seed=1)

    proba = infer(X_train, y_train, X_test, task="classification",
                  device="cpu", **_checkpoint_kwargs())
    assert proba.shape == (X_test.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert _auroc(proba[:, 1], y_test) > 0.7


def test_regressor_still_works_unchanged():
    """Smoke test proving the classification work did not break regression."""
    from synthefy_nori import NoriRegressor

    rng = np.random.default_rng(0)
    n_train, n_test, d = 200, 50, 4
    X_train = rng.normal(size=(n_train, d)).astype(np.float32)
    true_w = rng.normal(size=d).astype(np.float32)
    y_train = (X_train @ true_w + rng.normal(scale=0.1, size=n_train)).astype(np.float32)
    X_test = rng.normal(size=(n_test, d)).astype(np.float32)
    y_truth = X_test @ true_w

    model = NoriRegressor(device="cpu", **_checkpoint_kwargs())
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    assert pred.shape == (n_test,)
    assert np.all(np.isfinite(pred))
    corr = float(np.corrcoef(pred, y_truth)[0, 1])
    assert corr > 0.8, f"regression regressed: corr={corr:.3f}"
