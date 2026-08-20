from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from synthefy_nori.api import NoriRegressor
from synthefy_nori.evaluation.models import NoriWrapper
from synthefy_nori.inference.predictor import NoriPredictor


class _DistributionPredictor:
    regression_head = "pinball"
    regression_quantiles = (0.05, 0.4, 0.95)

    def predict(self, X_train, y_train, X_test, *, return_distribution=False):
        assert return_distribution
        return np.tile(np.array([-2.0, 0.25, 3.0]), (len(X_test), 1))


def test_regressor_returns_exact_checkpoint_quantile_levels(monkeypatch):
    regressor = NoriRegressor().fit(
        np.arange(8, dtype=np.float32).reshape(4, 2),
        np.array([-1.0, 1.0, -1.0, 1.0]),
    )
    predictor = _DistributionPredictor()
    monkeypatch.setattr(regressor, "_get_predictor", lambda: predictor)

    result = regressor.predict(
        np.zeros((2, 2), dtype=np.float32),
        output_type="full",
    )

    np.testing.assert_array_equal(result["taus"], predictor.regression_quantiles)
    assert result["quantiles"].shape == (2, 3)
    np.testing.assert_allclose(result["mean"], 0.6375)


def test_evaluation_wrapper_returns_exact_checkpoint_quantile_levels():
    wrapper = NoriWrapper("test", "/unused.pt", device="cpu")
    wrapper._reg_predictor = _DistributionPredictor()

    _, taus, mean = wrapper.predict_distribution(
        np.zeros((4, 2), dtype=np.float32),
        np.array([-1.0, 1.0, -1.0, 1.0]),
        np.zeros((2, 2), dtype=np.float32),
    )

    np.testing.assert_array_equal(taus, (0.05, 0.4, 0.95))
    np.testing.assert_allclose(mean, 0.6375)


def test_legacy_checkpoint_quantile_grid_falls_back_from_head_width():
    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.model = SimpleNamespace(num_reg_quantiles=2)

    assert predictor.regression_head == "pinball"
    assert predictor.regression_quantiles == pytest.approx((1.0 / 3.0, 2.0 / 3.0))


def test_legacy_scalar_head_is_inferred_as_mse():
    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.model = SimpleNamespace(num_reg_quantiles=1)

    assert predictor.regression_head == "mse"


@pytest.mark.parametrize("mode", ["median", "trimmed_mean"])
def test_two_quantile_collapse_uses_both_values(mode):
    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.quantile_collapse = mode
    bank = torch.tensor([[1.0, 5.0], [-3.0, 7.0]])

    collapsed = predictor._apply_quantile_collapse(bank)

    torch.testing.assert_close(collapsed, torch.tensor([3.0, 2.0]))
    assert torch.isfinite(collapsed).all()


@pytest.mark.parametrize("mode", ["mean", "qdist", "qdist_simple"])
def test_predictive_mean_uses_nonuniform_checkpoint_levels(mode):
    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.model = SimpleNamespace(
        num_reg_quantiles=3,
        regression_loss="pinball",
        regression_quantiles=(0.1, 0.2, 0.9),
    )
    predictor.quantile_collapse = mode

    collapsed = predictor._apply_quantile_collapse(
        torch.tensor([[0.0, 10.0, 20.0]])
    )

    # Constant tails plus trapezoidal interpolation:
    # .1*0 + .1*(0+10)/2 + .7*(10+20)/2 + .1*20 = 13.
    torch.testing.assert_close(collapsed, torch.tensor([13.0]))


def test_qdist_exponential_path_uses_checkpoint_levels():
    tau = tuple(np.arange(1, 9, dtype=np.float64) / 10.0)
    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.model = SimpleNamespace(
        num_reg_quantiles=8,
        regression_loss="pinball",
        regression_quantiles=tau,
    )
    predictor.quantile_collapse = "qdist"
    bank = torch.full((1, 8), np.log(2.0), dtype=torch.float64)
    bank[0, 0] = 0.0

    collapsed = predictor._apply_quantile_collapse(bank)

    # lambda_L=1 for q(.1)=0, q(.2)=log(2); the right tail is flat.
    expected = (
        0.1 * -1.0
        + 0.1 * np.log(2.0) / 2.0
        + 0.6 * np.log(2.0)
        + 0.2 * np.log(2.0)
    )
    torch.testing.assert_close(collapsed, torch.tensor([expected], dtype=torch.float64))


def test_predictor_preserves_dense_grid_for_bfloat16_predictions():
    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.model = SimpleNamespace(
        num_reg_quantiles=999,
        regression_loss="pinball",
        regression_quantiles=tuple(
            np.arange(1, 1000, dtype=np.float64) / 1000.0
        ),
    )
    predictor.quantile_collapse = "mean"
    bank = torch.linspace(-2.0, 3.0, 999, dtype=torch.bfloat16).unsqueeze(0)

    collapsed = predictor._apply_quantile_collapse(bank)

    assert collapsed.dtype == torch.float32
    torch.testing.assert_close(collapsed, torch.tensor([0.5]), atol=2e-3, rtol=0.0)


def test_distribution_rejects_scalar_loss_metadata(monkeypatch):
    regressor = NoriRegressor().fit(
        np.arange(8, dtype=np.float32).reshape(4, 2),
        np.array([-1.0, 1.0, -1.0, 1.0]),
    )
    predictor = _DistributionPredictor()
    predictor.regression_head = "mse"
    monkeypatch.setattr(regressor, "_get_predictor", lambda: predictor)

    with pytest.raises(NotImplementedError, match="mse"):
        regressor.predict(np.zeros((1, 2), dtype=np.float32), output_type="full")
