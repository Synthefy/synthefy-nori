"""Offline tests for the evaluation runner's predictive-distribution metrics."""

import numpy as np
import pytest
from scipy.stats import norm

from synthefy_nori.evaluation import runner
from synthefy_nori.evaluation.datasets import DatasetEntry
from synthefy_nori.evaluation.models import ModelEntry
from synthefy_nori.evaluation.runner import DIST_METRIC_KEYS, EvalRunner, compute_dist_metrics

# The released checkpoints' 999-level grid.
TAUS = np.arange(1, 1000) / 1000.0


def _gaussian_bank(mu, sigma):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), mu.shape)
    return mu[:, None] + sigma[:, None] * norm.ppf(TAUS)[None, :]


def test_crps_matches_the_gaussian_closed_form():
    y = np.linspace(-3.0, 3.0, 61)
    mu, sigma = np.full_like(y, 0.5), 2.0
    z = (y - mu) / sigma
    exact = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

    metrics = compute_dist_metrics(y, _gaussian_bank(mu, sigma), TAUS)

    assert metrics["crps"] == pytest.approx(exact.mean(), rel=1e-4)
    assert metrics["crps_rel"] == pytest.approx(metrics["crps"] / y.std())


def test_calibrated_gaussian_hits_nominal_coverage_width_and_pit():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20_000)

    metrics = compute_dist_metrics(y, _gaussian_bank(np.zeros_like(y), 1.0), TAUS)

    for level in (50, 80, 90, 95):
        assert metrics[f"cov{level}"] == pytest.approx(level / 100, abs=0.01)
    assert metrics["width90_rel"] == pytest.approx(2 * norm.ppf(0.95) / y.std(), rel=1e-3)
    assert metrics["pit_ks"] < 0.02
    # E[CRPS] for a standard normal forecast of standard normal data is 1/sqrt(pi).
    assert metrics["crps"] == pytest.approx(1 / np.sqrt(np.pi), rel=0.02)


def test_miscalibration_shows_up_in_coverage_and_pit():
    rng = np.random.default_rng(1)
    y = rng.normal(size=5_000)

    overconfident = compute_dist_metrics(y, _gaussian_bank(np.zeros_like(y), 0.5), TAUS)
    biased = compute_dist_metrics(y, _gaussian_bank(np.full_like(y, 1.0), 1.0), TAUS)
    calibrated = compute_dist_metrics(y, _gaussian_bank(np.zeros_like(y), 1.0), TAUS)

    assert overconfident["cov90"] < 0.65
    assert biased["pit_ks"] > 0.3
    assert calibrated["crps"] < min(overconfident["crps"], biased["crps"])


def test_crps_is_independent_of_the_row_chunk_size(monkeypatch):
    rng = np.random.default_rng(3)
    y = rng.normal(size=1_000)
    bank = _gaussian_bank(rng.normal(size=1_000), rng.uniform(0.5, 2.0, size=1_000))
    whole = compute_dist_metrics(y, bank, TAUS)
    monkeypatch.setattr(runner, "_DIST_ROW_CHUNK", 7)  # uneven last chunk on purpose
    assert compute_dist_metrics(y, bank, TAUS) == pytest.approx(whole, rel=1e-12)


def test_crossed_quantiles_are_sorted_before_scoring():
    rng = np.random.default_rng(4)
    y = rng.normal(size=200)
    bank = _gaussian_bank(np.zeros_like(y), 1.0)
    crossed = bank.copy()
    crossed[::2] = crossed[::2, ::-1]  # reverse every other row
    assert compute_dist_metrics(y, crossed, TAUS) == pytest.approx(compute_dist_metrics(y, bank, TAUS))


def test_coverage_counts_exact_interval_membership():
    # Each row's central 90% interval is [-1, 1]; three targets fall inside, one outside.
    bank = np.tile(np.interp(TAUS, [0.05, 0.95], [-1.0, 1.0]), (4, 1))
    metrics = compute_dist_metrics([0.0, 0.9, -1.0, 2.0], bank, TAUS)
    assert metrics["cov90"] == pytest.approx(0.75)


def test_non_finite_rows_are_dropped_and_degenerate_inputs_are_nan():
    bank = _gaussian_bank(np.zeros(4), 1.0)
    bank[1, 10] = np.nan
    finite = compute_dist_metrics([0.1, 5.0, np.nan, -0.2], bank, TAUS)
    only_rows_0 = compute_dist_metrics([0.1, -0.2], bank[[0, 3]], TAUS)
    assert finite == pytest.approx(only_rows_0)

    assert all(np.isnan(v) for v in compute_dist_metrics([1.0], bank[:1], TAUS).values())
    constant = compute_dist_metrics([1.0, 1.0, 1.0], bank[:3], TAUS)
    assert np.isnan(constant["crps_rel"]) and np.isnan(constant["width90_rel"])
    assert np.isfinite(constant["crps"])


@pytest.mark.parametrize(
    ("quantiles", "taus"),
    [
        (np.zeros((3, 5)), TAUS),  # wrong number of columns
        (np.zeros((3, 3)), np.array([0.1, 0.1, 0.9])),  # not strictly increasing
        (np.zeros((3, 2)), np.array([0.0, 0.5])),  # level outside (0, 1)
    ],
)
def test_bad_shapes_or_levels_raise(quantiles, taus):
    with pytest.raises(ValueError):
        compute_dist_metrics(np.zeros(3), quantiles, taus)


# Runner integration (fake wrapper, no checkpoint)


class _FakeWrapper:
    """Point predictions are y=x0; the distribution is N(x0, 1) on the 999-level grid."""

    device = "cpu"

    def __init__(self, quantile_collapse="mean", has_quantile_head=True):
        self.quantile_collapse = quantile_collapse
        self.has_quantile_head = has_quantile_head
        self.calls = []

    @property
    def name(self):
        return "fake"

    def predict_regression(self, X_train, y_train, X_test):
        self.calls.append("point")
        return X_test[:, 0] + 0.25  # deliberately differs from the distribution mean

    def predict_distribution(self, X_train, y_train, X_test):
        self.calls.append("distribution")
        if not self.has_quantile_head:
            raise NotImplementedError("predict_distribution needs the pinball checkpoint")
        return _gaussian_bank(X_test[:, 0], 1.0), TAUS, X_test[:, 0]

    def cleanup(self):
        pass


def _dataset():
    rng = np.random.default_rng(2)
    X_train, X_test = rng.normal(size=(50, 2)), rng.normal(size=(400, 2))
    return DatasetEntry(
        name="toy",
        source="unit",
        task_type="regression",
        X_train=X_train,
        y_train=X_train[:, 0],
        X_test=X_test,
        y_test=X_test[:, 0] + rng.normal(size=400),
    )


def _evaluate(tmp_path, wrapper, distribution_metrics):
    runner = EvalRunner(
        model_registry=None,
        dataset_registry=None,
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        verbose=False,
        skip_on_error=False,
        distribution_metrics=distribution_metrics,
    )
    entry = ModelEntry(name="fake", wrapper=wrapper, model_type="custom")
    return runner, runner._eval_one(entry, _dataset(), prefix="")


def test_default_runner_is_point_only(tmp_path):
    wrapper = _FakeWrapper()
    _, result = _evaluate(tmp_path, wrapper, distribution_metrics=False)
    assert wrapper.calls == ["point"]
    assert set(result.metrics) == {"r2", "rmse", "mae"}


def test_distribution_run_adds_metrics_from_one_call(tmp_path):
    wrapper = _FakeWrapper()
    _, result = _evaluate(tmp_path, wrapper, distribution_metrics=True)
    assert wrapper.calls == ["distribution"]
    assert set(DIST_METRIC_KEYS) <= set(result.metrics)
    assert result.metrics["cov90"] == pytest.approx(0.9, abs=0.05)
    # Under quantile_collapse="mean" the point metrics come from the distribution mean.
    y = _dataset().y_test
    assert result.metrics["mae"] == pytest.approx(np.mean(np.abs(y - _dataset().X_test[:, 0])))


def test_non_mean_collapse_keeps_predict_regression_for_point_metrics(tmp_path):
    wrapper = _FakeWrapper(quantile_collapse="median")
    _, result = _evaluate(tmp_path, wrapper, distribution_metrics=True)
    assert wrapper.calls == ["distribution", "point"]
    ds = _dataset()
    assert result.metrics["mae"] == pytest.approx(np.mean(np.abs(ds.y_test - (ds.X_test[:, 0] + 0.25))))
    assert "cov90" in result.metrics


def test_checkpoint_without_quantile_head_falls_back_to_point_metrics(tmp_path):
    wrapper = _FakeWrapper(has_quantile_head=False)
    _, result = _evaluate(tmp_path, wrapper, distribution_metrics=True)
    assert result.error is None
    assert wrapper.calls == ["distribution", "point"]
    assert set(result.metrics) == {"r2", "rmse", "mae"}


def test_distribution_results_are_cached_separately_and_round_trip(tmp_path):
    point_runner, point_result = _evaluate(tmp_path, _FakeWrapper(), distribution_metrics=False)
    dist_runner, dist_result = _evaluate(tmp_path, _FakeWrapper(), distribution_metrics=True)
    args = ("fake", "toy", "unit", "regression")
    assert point_runner._cache_key(*args) != dist_runner._cache_key(*args)

    point_runner._cache_put(point_result)
    # A point-only cached result must not answer a distribution run...
    assert dist_runner._cache_get(*args) is None
    # ...and a cached distribution result restores every distribution metric.
    dist_runner._cache_put(dist_result)
    cached = dist_runner._cache_get(*args)
    assert cached.metrics == pytest.approx(dist_result.metrics)
