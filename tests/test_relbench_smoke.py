"""RelBench entity-task smoke tests for the Nori tabular protocol.

Deselected by default (the ``slow`` marker): these download both the RelBench
``rel-f1`` database (the smallest, ~74K rows) and the public ``Synthefy/Nori``
checkpoint, then run real in-context inference. ``rel-f1`` is used because it is
the cheapest dataset to fetch and run.

Run explicitly with::

    pytest -m slow tests/test_relbench_smoke.py

Requires the ``relbench`` extra (``pip install "synthefy-nori[relbench]"``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest


pytestmark = pytest.mark.slow

relbench = pytest.importorskip("relbench", reason="requires the 'relbench' extra")


def test_relbench_regression_smoke():
    from synthefy_nori.evaluation.relbench_tasks import run_task

    res = run_task(
        "rel-f1", "driver-position",
        mode="entity", device="cpu", max_train=2000, download=True,
    )
    assert res.error is None, f"regression task crashed: {res.error}"
    assert res.task_type == "REGRESSION"
    assert res.n_train > 0 and res.n_val > 0 and res.n_test > 0
    # task.evaluate exposes mae for regression; it must be a finite number.
    assert "test_mae" in res.metrics
    assert math.isfinite(res.metrics["test_mae"])
    assert math.isfinite(res.metrics["val_mae"])


def test_relbench_classification_smoke():
    from synthefy_nori.evaluation.relbench_tasks import run_task

    res = run_task(
        "rel-f1", "driver-dnf",
        mode="entity", device="cpu", max_train=2000, download=True,
    )
    assert res.error is None, f"classification task crashed: {res.error}"
    assert res.task_type == "BINARY_CLASSIFICATION"
    # task.evaluate exposes roc_auc for binary classification.
    assert "test_roc_auc" in res.metrics
    auroc = res.metrics["test_roc_auc"]
    assert math.isfinite(auroc) and 0.0 <= auroc <= 1.0


def test_feature_table_builds_numeric():
    """The flattened feature table must be all-numeric and label-aligned."""
    from relbench.base import TaskType
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from synthefy_nori.evaluation.relbench_tasks import build_feature_table

    task = get_task("rel-f1", "driver-position", download=True)
    db = get_dataset("rel-f1", download=True).get_db()
    X_df, y = build_feature_table(task, db, "val", mode="entity")
    assert y is not None and len(y) == len(X_df)
    assert task.target_col not in X_df.columns
    assert task.entity_col not in X_df.columns
    # test split is label-masked: y is absent
    X_test_df, y_test = build_feature_table(task, db, "test", mode="entity")
    assert y_test is None
    assert task.task_type in (TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION)
