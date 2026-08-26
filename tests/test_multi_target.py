from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.base import clone

import synthefy_nori.api as api_module
from synthefy_nori import MultiTargetPredictionPolicy, NoriRegressor
from synthefy_nori.multi_target import (
    cdf_from_quantile_bank,
    quantiles_from_levels,
)


class _FakePredictor:
    regression_head = "pinball"
    regression_quantiles = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
    quantile_collapse = "mean"
    bar_point_estimator = "mean"
    memory_policy = None
    context_cache_entries = 1
    memory_report_ = {"rung": "no_cache"}

    def predict(self, X_train, y_train, X_test, return_distribution=False):
        center = np.asarray(X_test, dtype=np.float64)[:, 0]
        if return_distribution:
            return center[:, None] + np.asarray([-1.0, 0.0, 1.0])[None, :]
        return center


class _FakeVinecop:
    fitted_pits = None

    @classmethod
    def from_data(cls, pits, controls=None):
        instance = cls()
        instance.fitted_pits = np.asarray(pits)
        return instance

    def simulate(self, n, seeds=None):
        rng = np.random.default_rng(seeds[0])
        shared = rng.random((n, 1))
        return np.repeat(shared, self.fitted_pits.shape[1], axis=1)


@pytest.fixture
def fake_runtime(monkeypatch):
    predictor = _FakePredictor()
    monkeypatch.setattr(NoriRegressor, "_get_predictor", lambda self: predictor)
    families = SimpleNamespace(
        **{
            name: name
            for name in (
                "indep",
                "gaussian",
                "student",
                "clayton",
                "gumbel",
                "frank",
                "joe",
                "bb1",
                "bb6",
                "bb7",
                "bb8",
                "tawn",
            )
        }
    )
    fake_pyvine = SimpleNamespace(
        BicopFamily=families,
        FitControlsVinecop=lambda **kwargs: kwargs,
        Vinecop=_FakeVinecop,
    )
    monkeypatch.setattr(api_module, "pyvinecopulib", fake_pyvine)
    return predictor


def _data(n=8):
    X = np.column_stack([np.linspace(-1.0, 1.0, n), np.arange(n) % 2])
    Y = np.column_stack([2.0 * X[:, 0], -X[:, 0] + X[:, 1]])
    return X, Y


def test_policy_is_strict_and_prediction_override_is_partial():
    with pytest.raises(ValidationError):
        MultiTargetPredictionPolicy(n_draw=10)
    fitted = MultiTargetPredictionPolicy(n_draws=20, copula_cv=3)
    merged = fitted.merge_prediction_override(MultiTargetPredictionPolicy(n_draws=7))
    assert merged.n_draws == 7
    assert merged.copula_cv == 3
    with pytest.raises(ValueError, match="fit-dependent"):
        fitted.merge_prediction_override(MultiTargetPredictionPolicy(copula_cv=4))
    with pytest.raises(ValidationError, match="cannot be combined"):
        MultiTargetPredictionPolicy(
            autoregressive_n_orders=2,
            autoregressive_orders=[[0, 1], [1, 0]],
        )
    with pytest.raises(ValidationError, match="duplicate"):
        MultiTargetPredictionPolicy(autoregressive_orders=[[0, 1], [0, 1]])
    with pytest.raises(ValidationError):
        MultiTargetPredictionPolicy(random_state=-1)
    with pytest.raises(ValidationError):
        MultiTargetPredictionPolicy(random_state=2**32)


def test_quantile_bank_cdf_round_trip_and_duplicate_values():
    taus = np.asarray([0.1, 0.5, 0.9])
    bank = np.asarray([[0.0, 1.0, 2.0], [3.0, 3.0, 5.0]])
    levels = np.asarray([[0.2, 0.7], [0.5, 0.8]])
    values = quantiles_from_levels(bank, taus, levels)
    recovered = np.column_stack([cdf_from_quantile_bank(bank, taus, values[:, column]) for column in range(2)])
    assert np.allclose(recovered[0], levels[0])
    assert recovered[1, 0] == pytest.approx(0.5)
    assert recovered[1, 1] == pytest.approx(0.8)


def test_quantile_inverse_preserves_native_probability_levels_and_clamps_tails():
    taus = np.asarray([0.1, 0.5, 0.9])
    bank = np.asarray([[0.0, 1.0, 10.0]])
    levels = np.asarray([[0.0, 0.1, 0.5, 0.9, 1.0]])

    values = quantiles_from_levels(bank, taus, levels)

    assert np.array_equal(values, [[0.0, 0.0, 1.0, 10.0, 10.0]])
    assert cdf_from_quantile_bank(bank, taus, np.asarray([-1.0]))[0] == 0.0
    assert cdf_from_quantile_bank(bank, taus, np.asarray([10.0]))[0] == 1.0
    assert cdf_from_quantile_bank(bank, taus, np.asarray([11.0]))[0] == 1.0

    constant_bank = np.asarray([[4.0, 4.0, 4.0]])
    assert cdf_from_quantile_bank(constant_bank, taus, np.asarray([3.9]))[0] == 0.0
    assert cdf_from_quantile_bank(constant_bank, taus, np.asarray([4.0]))[0] == 1.0


def test_cdf_preserves_both_sides_of_an_interior_quantile_tie():
    taus = np.asarray([0.1, 0.3, 0.6, 0.9])
    bank = np.asarray([[0.0, 1.0, 1.0, 2.0]])

    values = np.asarray([0.5, 1.0, 1.5])
    cdf = np.asarray([cdf_from_quantile_bank(bank, taus, np.asarray([value]))[0] for value in values])

    assert np.allclose(cdf, [0.2, 0.6, 0.75])


def test_independent_samples_are_deterministic_and_share_one_runtime(fake_runtime):
    X, Y = _data()
    regressor = NoriRegressor(
        model_path="unused",
        multi_target_prediction_strategy="independent",
        multi_target_prediction_policy={"n_draws": 6, "random_state": 42},
    ).fit(X, Y)

    first = regressor.predict(X[:3], output_type="samples")
    second = regressor.predict(X[:3], output_type="samples")
    assert first.shape == (3, 6, 2)
    assert np.array_equal(first, second)
    assert all(marginal._predictor is fake_runtime for marginal in regressor.marginal_estimators_)
    assert regressor.nori_calls_ == 2
    assert len(regressor.memory_report_) == 2


def test_copula_is_default_and_uses_cross_fitted_pits(fake_runtime):
    pd = pytest.importorskip("pandas")
    X, Y = _data(10)
    target_frame = pd.DataFrame(Y, columns=["left", "right"])
    regressor = NoriRegressor(
        model_path="unused",
        multi_target_prediction_policy={"n_draws": 5, "copula_cv": 2},
    ).fit(X, target_frame)

    assert regressor.multi_target_prediction_strategy_ == "copula"
    assert regressor.copula_.fitted_pits.shape == Y.shape
    assert set(regressor.copula_fold_indices_) == {0, 1}
    assert regressor.output_names_.tolist() == ["left", "right"]
    samples = regressor.predict(X[:4], output_type="samples")
    assert samples.shape == (4, 5, 2)
    assert np.all(np.isfinite(samples))


def test_autoregressive_supports_joint_mean_and_draw_override(fake_runtime):
    X, Y = _data()
    regressor = NoriRegressor(
        model_path="unused",
        multi_target_prediction_strategy="autoregressive",
        multi_target_prediction_policy={
            "n_draws": 6,
            "random_state": 3,
            "autoregressive_n_orders": 2,
        },
    ).fit(X, Y)

    samples = regressor.predict(
        X[:2],
        output_type="samples",
        multi_target_prediction_policy=MultiTargetPredictionPolicy(n_draws=4),
    )
    mean = regressor.predict(X[:2])
    assert samples.shape == (2, 4, 2)
    assert mean.shape == (2, 2)
    assert len(regressor.target_orders_) == 2


def test_autoregressive_expanded_queries_are_chunked(fake_runtime, monkeypatch):
    X, Y = _data()
    monkeypatch.setattr(api_module, "MAX_AUTOREGRESSIVE_EXPANDED_ROWS_PER_CALL", 2)
    regressor = NoriRegressor(
        model_path="unused",
        multi_target_prediction_strategy="autoregressive",
        multi_target_prediction_policy={
            "n_draws": 4,
            "autoregressive_orders": [[0, 1]],
        },
    ).fit(X, Y)

    samples = regressor.predict(X[:2], output_type="samples")

    assert samples.shape == (2, 4, 2)
    assert regressor.nori_calls_ == 5  # one first-target call plus four bounded chunks


@pytest.mark.parametrize("strategy", ["independent", "autoregressive"])
def test_multi_target_refreshes_memory_policy_after_fit(fake_runtime, strategy):
    X, Y = _data()
    policy = {"n_draws": 2}
    if strategy == "autoregressive":
        policy["autoregressive_orders"] = [[0, 1]]
    regressor = NoriRegressor(
        model_path="unused",
        memory_policy="exact",
        multi_target_prediction_strategy=strategy,
        multi_target_prediction_policy=policy,
    ).fit(X, Y)
    regressor.memory_policy = "off"

    regressor.predict(X[:2], output_type="samples")

    children = (
        regressor.marginal_estimators_
        if strategy == "independent"
        else [child for chain in regressor.autoregressive_chains_ for child in chain]
    )
    assert all(child.memory_policy == "off" for child in children)


def test_autoregressive_explicit_orders_are_used_exactly(fake_runtime):
    X, Y = _data()
    regressor = NoriRegressor(
        model_path="unused",
        multi_target_prediction_strategy="autoregressive",
        multi_target_prediction_policy={
            "n_draws": 4,
            "autoregressive_orders": [[1, 0], [0, 1]],
        },
    ).fit(X, Y)

    assert regressor.target_orders_ == [(1, 0), (0, 1)]
    assert regressor.predict(X[:2], output_type="samples").shape == (2, 4, 2)


def test_autoregressive_automatic_orders_cap_at_unique_permutations(fake_runtime):
    X, Y = _data()
    regressor = NoriRegressor(
        model_path="unused",
        multi_target_prediction_strategy="autoregressive",
        multi_target_prediction_policy={"autoregressive_n_orders": 8},
    ).fit(X, Y)

    assert len(regressor.target_orders_) == 2
    assert len(set(regressor.target_orders_)) == 2


def test_autoregressive_rejects_incomplete_explicit_order(fake_runtime):
    X, Y = _data()
    with pytest.raises(ValueError, match="complete permutation"):
        NoriRegressor(
            model_path="unused",
            multi_target_prediction_strategy="autoregressive",
            multi_target_prediction_policy={"autoregressive_orders": [[0, 0]]},
        ).fit(X, Y)


def test_scalar_path_and_clone_contract_remain_unchanged(fake_runtime):
    X, Y = _data()
    policy = MultiTargetPredictionPolicy(n_draws=4)
    regressor = NoriRegressor(
        model_path="unused",
        multi_target_prediction_strategy="independent",
        multi_target_prediction_policy=policy,
    )
    cloned = clone(regressor)
    assert cloned.multi_target_prediction_policy is not policy
    assert cloned.multi_target_prediction_policy == policy

    scalar = regressor.fit(X, Y[:, 0]).predict(X[:3])
    assert scalar.shape == (3,)
    assert not regressor._multi_target_active_


def test_multi_target_rejects_unsupported_contracts(fake_runtime):
    X, Y = _data()
    with pytest.raises(ValueError, match="large_context_policy"):
        NoriRegressor(
            model_path="unused",
            large_context_policy="random",
            multi_target_prediction_strategy="independent",
        ).fit(X, Y)

    regressor = NoriRegressor(
        model_path="unused",
        multi_target_prediction_strategy="independent",
    ).fit(X, Y)
    with pytest.raises(ValueError, match="output_type='mean'"):
        regressor.predict(X[:2], output_type="full")
    with pytest.raises(ValueError, match="same number of rows"):
        regressor.fit(X, Y[:-1])
