"""Best-competitor diagnostics must agree with each metric's direction."""

import numpy as np
import pandas as pd
import pytest

from synthefy_nori.evaluation.analysis import EvalAnalyzer


def _build_comparison_results(metric, scores):
    return pd.DataFrame(
        [
            {"source": "audit", "dataset": "example", "model": model, "task_type": "regression", metric: score}
            for model, score in scores.items()
        ]
    )


@pytest.mark.parametrize("metric", ["rmse", "mae", "log_loss", "ece", "latency_ms"])
@pytest.mark.parametrize("focus_score,expected_delta", [(2.0, -1.0), (0.5, 0.5)])
def test_best_competitor_and_advantage_respect_lower_is_better(metric, focus_score, expected_delta):
    analyzer = EvalAnalyzer(_build_comparison_results(metric, {"focus": focus_score, "best": 1.0, "worst": 3.0}))

    row = analyzer.comparison_vs_best_other("focus", "regression", metric).iloc[0]

    assert row["best_other_model"] == "best"
    assert row["best_other_score"] == 1.0
    assert row["delta_vs_best_other"] == expected_delta


@pytest.mark.parametrize("metric", ["r2", "auc", "accuracy", "spearman_ic"])
def test_best_competitor_preserves_higher_is_better(metric):
    analyzer = EvalAnalyzer(_build_comparison_results(metric, {"focus": 0.8, "best": 0.9, "worst": 0.2}))

    row = analyzer.comparison_vs_best_other("focus", "regression", metric).iloc[0]

    assert row["best_other_model"] == "best"
    assert row["best_other_score"] == 0.9
    assert row["delta_vs_best_other"] == pytest.approx(-0.1)


def test_best_competitor_ignores_nonfinite_scores():
    analyzer = EvalAnalyzer(
        _build_comparison_results("rmse", {"focus": 2.0, "valid": 1.0, "inf": np.inf, "nan": np.nan})
    )

    row = analyzer.comparison_vs_best_other("focus", "regression", "rmse").iloc[0]

    assert row["best_other_model"] == "valid"
    assert row["delta_vs_best_other"] == -1.0
