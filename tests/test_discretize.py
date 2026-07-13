"""Unit tests for synthefy_nori.discretize + the categorical_target API wiring."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from synthefy_nori.api import NoriRegressor
from synthefy_nori.discretize import (
    DISCRETIZE_METHODS,
    cell_masses,
    discretize_predictions,
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

    def test_map_cell_default(self, monkeypatch):
        model = self._fitted(monkeypatch)
        out = model.predict(np.zeros((3, 2), dtype=np.float32), categorical_target=True)
        assert out.tolist() == [7.0, 3.0, 5.0]

    def test_snap_mean_forces_mean_collapse(self, monkeypatch):
        model = self._fitted(monkeypatch)
        seen = {}

        def fake_point(self, X, *, quantile_collapse, bar_point_estimator):
            seen["collapse"] = (quantile_collapse, bar_point_estimator)
            return np.array([6.6, 2.0, 4.9])

        monkeypatch.setattr(NoriRegressor, "_predict_point", fake_point)
        out = model.predict(np.zeros((3, 2), dtype=np.float32),
                            categorical_target=True, discretize="snap-mean")
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
                      categorical_target=True, discretize="snap-median")
        assert seen["collapse"] == ("median", "median")

    def test_categorical_levels_override(self, monkeypatch):
        model = self._fitted(monkeypatch)
        # supply a lattice richer than the fitted y (e.g. known 1-9 scale)
        out = model.predict(np.zeros((3, 2), dtype=np.float32),
                            categorical_target=True,
                            categorical_levels=np.arange(1.0, 10.0))
        assert out.tolist() == [7.0, 3.0, 5.0]

    def test_discretize_without_flag_raises(self, monkeypatch):
        model = self._fitted(monkeypatch)
        with pytest.raises(ValueError, match="require categorical_target=True"):
            model.predict(np.zeros((3, 2), dtype=np.float32), discretize="median-cell")
        with pytest.raises(ValueError, match="require categorical_target=True"):
            model.predict(np.zeros((3, 2), dtype=np.float32),
                          categorical_levels=[1.0, 2.0])

    def test_rejects_output_type_combo(self, monkeypatch):
        model = self._fitted(monkeypatch)
        with pytest.raises(ValueError, match="categorical_target"):
            model.predict(np.zeros((3, 2), dtype=np.float32),
                          categorical_target=True, output_type="median")

    def test_unknown_strategy_raises(self, monkeypatch):
        model = self._fitted(monkeypatch)
        with pytest.raises(ValueError, match="Unknown discretize"):
            model.predict(np.zeros((3, 2), dtype=np.float32),
                          categorical_target=True, discretize="banana")

    def test_bar_checkpoint_gets_actionable_error(self, monkeypatch):
        model = self._fitted(monkeypatch)

        def raise_ni(self, X, *, output_type, quantiles):
            raise NotImplementedError("bar_distribution")

        monkeypatch.setattr(NoriRegressor, "_predict_distribution", raise_ni)
        with pytest.raises(NotImplementedError, match="snap-mean"):
            model.predict(np.zeros((3, 2), dtype=np.float32), categorical_target=True)

    def test_requires_fit(self):
        with pytest.raises(ValueError, match="fit"):
            NoriRegressor().predict(np.zeros((1, 2), dtype=np.float32),
                                    categorical_target=True)

    @pytest.mark.parametrize("method", DISCRETIZE_METHODS[:2])
    def test_outputs_are_on_lattice(self, monkeypatch, method):
        model = self._fitted(monkeypatch)
        out = model.predict(np.zeros((3, 2), dtype=np.float32),
                            categorical_target=True, discretize=method)
        assert set(out.tolist()) <= {3.0, 5.0, 7.0}
