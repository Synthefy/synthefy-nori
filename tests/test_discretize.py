"""Unit tests for synthefy_nori.discretize + the discretize=/categorical_levels= API wiring."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from synthefy_nori.api import NoriRegressor
from synthefy_nori.discretize import (
    DISCRETIZE_METHOD_DESCRIPTIONS,
    DISCRETIZE_METHODS,
    cell_masses,
    discretize_predictions,
    prior_match,
    snap_to_levels,
    target_levels,
)


def _bank_from_normal(mu, sigma, k=99):
    """Quantile bank rows of N(mu, sigma) at taus = i/(k+1)."""
    taus = (np.arange(k) + 1.0) / (k + 1.0)
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    Q = mu[:, None] + sigma[:, None] * norm.ppf(taus)[None, :]
    return Q, taus


class TestSnapToLevels:
    def test_nearest(self):
        levels = np.array([3.0, 4.0, 5.0, 6.0])
        assert snap_to_levels([5.37, 3.9, 100.0, -7.0], levels).tolist() == [5.0, 4.0, 6.0, 3.0]

    def test_tie_goes_up(self):
        # same convention as NoriPredictor._maybe_snap_discrete_y
        assert snap_to_levels([4.5], np.array([4.0, 5.0])).tolist() == [5.0]

    def test_nan_propagates(self):
        out = snap_to_levels([np.nan, 4.1], np.array([3.0, 4.0, 5.0]))
        assert np.isnan(out[0]) and out[1] == 4.0

    def test_infinities_snap_to_extremes(self):
        out = snap_to_levels([np.inf, -np.inf], np.array([3.0, 4.0, 5.0]))
        assert out.tolist() == [5.0, 3.0]

    def test_single_level(self):
        assert snap_to_levels([1.2, -9.0], np.array([7.0])).tolist() == [7.0, 7.0]


class TestCellMasses:
    def test_rows_sum_to_one(self):
        Q, taus = _bank_from_normal([5.0, 4.2], [0.5, 2.0])
        P = cell_masses(Q, taus, np.array([3.0, 4.0, 5.0, 6.0, 7.0]))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)

    def test_tight_distribution_concentrates(self):
        Q, taus = _bank_from_normal([5.0], [0.05])
        P = cell_masses(Q, taus, np.array([3.0, 4.0, 5.0, 6.0]))
        assert P[0].argmax() == 2
        assert P[0, 2] > 0.99

    def test_mass_below_and_above_bank(self):
        Q, taus = _bank_from_normal([0.0], [1.0])
        P = cell_masses(Q, taus, np.array([-100.0, 0.0, 100.0]))
        assert P[0, 1] > 0.95

    def test_single_level_single_cell(self):
        Q, taus = _bank_from_normal([5.0], [1.0])
        P = cell_masses(Q, taus, np.array([7.0]))
        assert P.shape == (1, 1) and P[0, 0] == 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            cell_masses(np.zeros((2, 5)), np.linspace(0.1, 0.9, 4), np.array([0.0, 1.0]))

    def test_empty_rows(self):
        taus = np.linspace(0.01, 0.99, 9)
        P = cell_masses(np.empty((0, 9)), taus, np.array([1.0, 2.0, 3.0]))
        assert P.shape == (0, 3)


class TestDiscretizePredictions:
    def test_mode_vs_mean_on_skewed_posterior(self):
        # two-lobe bank: 60% of mass near level 7, 40% near level 3 ->
        # mean lands mid-lattice (bad for accuracy), mode picks 7.
        taus = (np.arange(99) + 1.0) / 100.0
        row = np.where(taus < 0.4, 3.0, 7.0) + np.linspace(0, 0.01, 99)
        Q = row[None, :]
        levels = np.array([3.0, 5.0, 7.0])
        mode = discretize_predictions("map-cell", levels, Q=Q, taus=taus)
        assert mode.tolist() == [7.0]
        snapped_mean = discretize_predictions("snap-mean", levels, point=Q.mean(axis=1))
        assert snapped_mean.tolist() == [5.0]  # mean ~5.4 snaps to the wrong level

    def test_median_cell_crosses_half(self):
        taus = (np.arange(99) + 1.0) / 100.0
        row = np.where(taus < 0.4, 3.0, 7.0) + np.linspace(0, 0.01, 99)
        out = discretize_predictions(
            "median-cell", np.array([3.0, 5.0, 7.0]), Q=row[None, :], taus=taus)
        assert out.tolist() == [7.0]

    def test_unsorted_duplicate_levels_normalized(self):
        Q, taus = _bank_from_normal([7.0], [0.05])
        out = discretize_predictions(
            "map-cell", np.array([7.0, 3.0, 5.0, 3.0]), Q=Q, taus=taus)
        assert out.tolist() == [7.0]

    def test_single_level_cell_method(self):
        Q, taus = _bank_from_normal([5.0], [1.0])
        out = discretize_predictions("map-cell", np.array([7.0]), Q=Q, taus=taus)
        assert out.tolist() == [7.0]

    def test_expected_level_matches_posterior_mean(self):
        taus = (np.arange(99) + 1.0) / 100.0
        row = np.where(taus < 0.4, 3.0, 7.0) + np.linspace(0, 0.01, 99)
        out = discretize_predictions(
            "expected-level", np.array([3.0, 5.0, 7.0]), Q=row[None, :], taus=taus)
        # ~40% mass at 3, ~60% at 7 -> expectation ~ 5.4 (continuous, off-lattice)
        assert 5.0 < out[0] < 5.8
        assert out[0] not in (3.0, 5.0, 7.0)

    def test_prior_match_matches_train_frequencies(self):
        y_train = np.array([3.0] * 6 + [5.0] * 3 + [7.0] * 3)  # 50/25/25
        point = np.linspace(0, 10, 8)
        out = discretize_predictions(
            "prior-match", np.array([3.0, 5.0, 7.0]), point=point, y_train=y_train)
        vals, counts = np.unique(out, return_counts=True)
        assert vals.tolist() == [3.0, 5.0, 7.0]
        assert counts.tolist() == [4, 2, 2]  # 8 rows at 50/25/25
        # lowest-ranked points get the lowest levels
        assert out[np.argsort(point)[0]] == 3.0
        assert out[np.argsort(point)[-1]] == 7.0

    def test_prior_match_requires_y_train(self):
        with pytest.raises(ValueError, match="y_train"):
            discretize_predictions("prior-match", np.array([1.0, 2.0]),
                                   point=np.array([1.0]))

    def test_prior_match_direct_zero_overlap_raises(self):
        with pytest.raises(ValueError, match="no training values"):
            prior_match(np.array([1.0]), np.array([10.0, 20.0]), np.array([1.0, 2.0]))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown discretize"):
            discretize_predictions("banana", np.array([1.0]), point=np.array([1.0]))

    def test_missing_inputs_raise(self):
        with pytest.raises(ValueError, match="requires point"):
            discretize_predictions("snap-mean", np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="requires the quantile bank"):
            discretize_predictions("map-cell", np.array([1.0, 2.0]))


class TestTargetLevels:
    def test_unique_sorted_finite(self):
        levels = target_levels([5, 3, 5, np.nan, 4, 3])
        assert levels.tolist() == [3.0, 4.0, 5.0]

    def test_many_levels_warns(self):
        with pytest.warns(UserWarning, match="distinct values"):
            target_levels(np.arange(1000, dtype=float))

    def test_all_nan_raises(self):
        with pytest.raises(ValueError):
            target_levels([np.nan, np.inf])


class TestPackageSurface:
    def test_descriptions_cover_all_methods(self):
        assert set(DISCRETIZE_METHOD_DESCRIPTIONS) == set(DISCRETIZE_METHODS)
        assert all(len(v) > 40 for v in DISCRETIZE_METHOD_DESCRIPTIONS.values())


    def test_submodule_attribute_access(self):
        # docs promise `synthefy_nori.discretize.snap_to_levels` works
        import synthefy_nori

        assert synthefy_nori.discretize.snap_to_levels is snap_to_levels


class TestPredictCategoricalAPI:
    """API wiring without a real checkpoint (distribution/point paths stubbed)."""

    def _fitted(self, monkeypatch):
        model = NoriRegressor()
        X = np.zeros((6, 2), dtype=np.float32)
        y = np.array([3.0, 3.0, 5.0, 5.0, 7.0, 7.0])
        model.fit(X, y)
        Q, taus = _bank_from_normal([6.9, 3.1, 5.0], [0.1, 0.1, 0.1])
        monkeypatch.setattr(
            NoriRegressor, "_predict_distribution",
            lambda self, X, *, output_type, quantiles:
                {"quantiles": Q, "taus": taus, "mean": Q.mean(axis=1)},
        )
        return model

    def test_map_cell_default_via_levels_only(self, monkeypatch):
        # categorical_levels alone activates with the default (map-cell) strategy
        model = self._fitted(monkeypatch)
        out = model.predict(np.zeros((3, 2), dtype=np.float32),
                            categorical_levels=[3.0, 5.0, 7.0])
        assert out.tolist() == [7.0, 3.0, 5.0]

    def test_snap_mean_forces_mean_collapse(self, monkeypatch):
        model = self._fitted(monkeypatch)
        seen = {}

        def fake_point(self, X, *, quantile_collapse, bar_point_estimator):
            seen["collapse"] = (quantile_collapse, bar_point_estimator)
            return np.array([6.6, 2.0, 4.9])

        monkeypatch.setattr(NoriRegressor, "_predict_point", fake_point)
        out = model.predict(np.zeros((3, 2), dtype=np.float32),
                            discretize="snap-mean")
        assert out.tolist() == [7.0, 3.0, 5.0]
        # snap-mean must snap the MEAN even if the regressor was configured
        # with a different quantile_collapse
        assert seen["collapse"] == ("mean", "mean")

    def test_snap_median_forces_median_collapse(self, monkeypatch):
        model = self._fitted(monkeypatch)
        seen = {}

        def fake_point(self, X, *, quantile_collapse, bar_point_estimator):
            seen["collapse"] = (quantile_collapse, bar_point_estimator)
            return np.array([6.6, 2.0, 4.9])

        monkeypatch.setattr(NoriRegressor, "_predict_point", fake_point)
        model.predict(np.zeros((3, 2), dtype=np.float32),
                      discretize="snap-median")
        assert seen["collapse"] == ("median", "median")

    def test_categorical_levels_override(self, monkeypatch):
        model = self._fitted(monkeypatch)
        # supply a lattice richer than the fitted y (e.g. known 1-9 scale)
        out = model.predict(np.zeros((3, 2), dtype=np.float32),
                            categorical_levels=np.arange(1.0, 10.0))
        assert out.tolist() == [7.0, 3.0, 5.0]

    def test_discretize_alone_activates(self, monkeypatch):
        # per review: passing discretize= (or categorical_levels=) alone is
        # enough -- no separate flag needed
        model = self._fitted(monkeypatch)
        out = model.predict(np.zeros((3, 2), dtype=np.float32), discretize="median-cell")
        assert set(out.tolist()) <= {3.0, 5.0, 7.0}
        out = model.predict(np.zeros((3, 2), dtype=np.float32),
                            categorical_levels=[3.0, 5.0, 7.0])
        assert set(out.tolist()) <= {3.0, 5.0, 7.0}

    def test_infer_helper_implication(self, monkeypatch):
        # the one-shot helper routes discretize= through the same implication
        from synthefy_nori.api import infer

        Q, taus = _bank_from_normal([6.9, 3.1, 5.0], [0.1, 0.1, 0.1])
        monkeypatch.setattr(
            NoriRegressor, "_predict_distribution",
            lambda self, X, *, output_type, quantiles:
                {"quantiles": Q, "taus": taus, "mean": Q.mean(axis=1)},
        )
        out = infer(np.zeros((6, 2), dtype=np.float32),
                    np.array([3.0, 3.0, 5.0, 5.0, 7.0, 7.0]),
                    np.zeros((3, 2), dtype=np.float32),
                    discretize="map-cell")
        assert out.tolist() == [7.0, 3.0, 5.0]

    def test_flag_is_gone(self, monkeypatch):
        # categorical_target was removed entirely (PR review) -- passing it
        # must fail loudly, not be silently swallowed
        model = self._fitted(monkeypatch)
        with pytest.raises(TypeError):
            model.predict(np.zeros((3, 2), dtype=np.float32), categorical_target=True)

    def test_rejects_output_type_combo(self, monkeypatch):
        model = self._fitted(monkeypatch)
        with pytest.raises(ValueError, match="categorical output"):
            model.predict(np.zeros((3, 2), dtype=np.float32),
                          discretize="map-cell", output_type="median")

    def test_unknown_strategy_raises(self, monkeypatch):
        model = self._fitted(monkeypatch)
        with pytest.raises(ValueError, match="Unknown discretize"):
            model.predict(np.zeros((3, 2), dtype=np.float32), discretize="banana")

    def test_bar_checkpoint_gets_actionable_error(self, monkeypatch):
        model = self._fitted(monkeypatch)

        def raise_ni(self, X, *, output_type, quantiles):
            raise NotImplementedError("bar_distribution")

        monkeypatch.setattr(NoriRegressor, "_predict_distribution", raise_ni)
        with pytest.raises(NotImplementedError, match="snap-mean"):
            model.predict(np.zeros((3, 2), dtype=np.float32), discretize="map-cell")

    def test_requires_fit(self):
        with pytest.raises(ValueError, match="fit"):
            NoriRegressor().predict(np.zeros((1, 2), dtype=np.float32),
                                    discretize="map-cell")

    def test_expected_level_via_predict(self, monkeypatch):
        model = self._fitted(monkeypatch)
        out = model.predict(np.zeros((3, 2), dtype=np.float32),
                            discretize="expected-level")
        assert out.shape == (3,)  # continuous, near the per-row bank means

    def test_prior_match_via_predict(self, monkeypatch):
        model = self._fitted(monkeypatch)
        monkeypatch.setattr(
            NoriRegressor, "_predict_point",
            lambda self, X, *, quantile_collapse, bar_point_estimator:
                np.array([6.6, 2.0, 4.9]))
        out = model.predict(np.zeros((3, 2), dtype=np.float32),
                            discretize="prior-match")
        # fitted y is 2x each of {3,5,7} -> 3 rows get one of each, rank-ordered
        assert sorted(out.tolist()) == [3.0, 5.0, 7.0]
        assert out.tolist() == [7.0, 3.0, 5.0]

    @pytest.mark.parametrize("method", DISCRETIZE_METHODS[:2])
    def test_outputs_are_on_lattice(self, monkeypatch, method):
        model = self._fitted(monkeypatch)
        out = model.predict(np.zeros((3, 2), dtype=np.float32), discretize=method)
        assert set(out.tolist()) <= {3.0, 5.0, 7.0}


class TestSklearnEcosystem:
    """Estimator-level categorical params: clone/get_params/CV reachability."""

    def _patch_distribution(self, monkeypatch):
        def fake_dist(self, X, *, output_type, quantiles):
            n = np.asarray(X).shape[0]
            taus = (np.arange(99) + 1.0) / 100.0
            Q = 5.0 + 0.1 * norm.ppf(taus)[None, :] * np.ones((n, 1))
            return {"quantiles": Q, "taus": taus, "mean": Q.mean(axis=1)}

        monkeypatch.setattr(NoriRegressor, "_predict_distribution", fake_dist)

    def test_constructor_params_drive_predict(self, monkeypatch):
        self._patch_distribution(monkeypatch)
        model = NoriRegressor(discretize="map-cell")
        model.fit(np.zeros((6, 2), dtype=np.float32),
                  np.array([3.0, 3.0, 5.0, 5.0, 7.0, 7.0]))
        out = model.predict(np.zeros((4, 2), dtype=np.float32))
        assert out.tolist() == [5.0, 5.0, 5.0, 5.0]

    def test_per_call_override_wins(self, monkeypatch):
        self._patch_distribution(monkeypatch)
        model = NoriRegressor(discretize="median-cell")
        model.fit(np.zeros((6, 2), dtype=np.float32),
                  np.array([3.0, 3.0, 5.0, 5.0, 7.0, 7.0]))
        monkeypatch.setattr(
            NoriRegressor, "_predict_point",
            lambda self, X, *, quantile_collapse, bar_point_estimator:
                np.array([5.37] * np.asarray(X).shape[0]))
        out = model.predict(np.zeros((2, 2), dtype=np.float32),
                            discretize="map-cell")
        assert set(out.tolist()) <= {3.0, 5.0, 7.0}  # per-call strategy override

    def test_clone_roundtrip(self):
        from sklearn.base import clone

        model = NoriRegressor(discretize="median-cell",
                              categorical_levels=[1.0, 2.0, 3.0])
        params = clone(model).get_params()
        assert params["discretize"] == "median-cell"
        assert params["categorical_levels"] == [1.0, 2.0, 3.0]

    def test_gridsearch_over_discretize(self, monkeypatch):
        from sklearn.model_selection import GridSearchCV

        self._patch_distribution(monkeypatch)
        X = np.zeros((12, 2), dtype=np.float32)
        y = np.tile([3.0, 5.0, 7.0], 4)
        gs = GridSearchCV(
            NoriRegressor(),
            {"discretize": ["map-cell", "median-cell"]},
            scoring="accuracy", cv=2, error_score="raise")
        gs.fit(X, y)
        assert gs.best_params_["discretize"] in ("map-cell", "median-cell")
