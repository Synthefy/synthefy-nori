"""Boundary tests for legacy synthetic learnability filters."""

from __future__ import annotations

import numpy as np
import sklearn.ensemble

from synthefy_nori.training import data_generator as dg


def _row_id_data(n_rows: int, n_features: int = 4):
    row_ids = np.arange(n_rows, dtype=np.float64)
    X = np.column_stack(
        [
            row_ids,
            *[np.sin(row_ids / (feature + 2.0)) for feature in range(n_features - 1)],
        ]
    )
    y = row_ids * 0.25 + np.cos(row_ids)
    return X, y


def test_extra_trees_filter_fits_context_and_scores_query_with_15_rows(monkeypatch):
    instances = []

    class SplitSpyExtraTrees:
        def __init__(self, **kwargs):
            del kwargs
            self.fit_ids = None
            self.score_ids = None
            instances.append(self)

        def fit(self, X, y):
            del y
            self.fit_ids = X[:, 0].astype(np.int64)
            return self

        def score(self, X, y):
            del y
            self.score_ids = X[:, 0].astype(np.int64)
            return 1.0

    monkeypatch.setattr(sklearn.ensemble, "ExtraTreesRegressor", SplitSpyExtraTrees)
    X, y = _row_id_data(64)

    assert dg._check_learnability(X, y, context_rows=15)
    assert len(instances) == 1
    np.testing.assert_array_equal(instances[0].fit_ids, np.arange(15))
    assert np.all(instances[0].score_ids >= 15)


def test_scaling_filter_uses_context_ladder_and_fixed_query(monkeypatch):
    instances = []

    class ScalingSpyExtraTrees:
        def __init__(self, **kwargs):
            del kwargs
            self.fit_ids = None
            self.score_ids = None
            instances.append(self)

        def fit(self, X, y):
            del y
            self.fit_ids = X[:, 0].astype(np.int64)
            return self

        def score(self, X, y):
            del y
            self.score_ids = X[:, 0].astype(np.int64)
            return len(self.fit_ids) / 100.0

    monkeypatch.setattr(sklearn.ensemble, "ExtraTreesRegressor", ScalingSpyExtraTrees)
    X, y = _row_id_data(80)

    assert dg._check_icl_scaling(X, y, context_rows=40)
    assert len(instances) >= 2
    for model in instances:
        assert np.all(model.fit_ids < 40)
        assert np.all(model.score_ids >= 40)
    expected_query_ids = np.sort(instances[0].score_ids)
    for model in instances[1:]:
        np.testing.assert_array_equal(np.sort(model.score_ids), expected_query_ids)


def test_legacy_icl_filter_preserves_context_query_sides(monkeypatch):
    captured = {}

    class SplitSpyModel:
        def __call__(self, x, y, eval_pos, y_type, task_type):
            assert y_type == "reg"
            assert task_type == "reg"
            captured["row_ids"] = x[0, :, 0].numpy().astype(np.int64)
            captured["eval_pos"] = eval_pos
            return {"reg_output": y[:, eval_pos:].unsqueeze(-1)}

    monkeypatch.setattr(dg, "_get_icl_filter_model", lambda _path: SplitSpyModel())
    X, y = _row_id_data(80)

    assert dg._check_learnability_icl(
        X,
        y,
        "fake-model",
        context_rows=40,
        max_context=16,
        max_query=12,
    )
    split = captured["eval_pos"]
    assert split == 16
    assert np.all(captured["row_ids"][:split] < 40)
    assert np.all(captured["row_ids"][split:] >= 40)


def test_generate_batch_omitted_context_keeps_legacy_filter_mode(monkeypatch):
    seen_context_rows = []

    def tree_episode(n_samples, n_features, task_type, n_classes, rng, context_rows=None, **_kwargs):
        del task_type, n_classes, rng
        row_ids = np.arange(n_samples, dtype=np.float64)
        X = np.column_stack([row_ids + feature for feature in range(n_features)])
        return {
            "X": X,
            "y": row_ids.astype(np.float32),
            "filtered": False,
            "meta": {"generator_family": "tree_prior"},
        }

    def learnability_filter(X, y, task_type, *, reg_min_score=0.10, context_rows=None, **_kwargs):
        del X, y, task_type, reg_min_score
        seen_context_rows.append(context_rows)
        return True

    monkeypatch.setattr(dg, "_generate_tree_prior_episode", tree_episode)
    monkeypatch.setattr(dg, "_check_learnability", learnability_filter)

    X, y, n_classes = dg.generate_batch(
        1,
        64,
        4,
        "reg",
        rng=np.random.default_rng(0),
        tree_prior_prob=1.0,
        learnability_filter=True,
        filter_max_retries=0,
    )

    assert X.shape == (1, 64, 4)
    assert y.shape == (1, 64)
    assert n_classes is None
    assert seen_context_rows == [None]
