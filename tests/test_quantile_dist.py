from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from synthefy_nori.model.quantile_dist import (
    quantile_dist_mean_batch,
    quantile_dist_mean_numpy,
    quantile_dist_mean_simple,
)


def _mid_area(q: np.ndarray, tau: np.ndarray) -> float:
    return float((0.5 * (q[:-1] + q[1:]) * np.diff(tau)).sum())


def test_exponential_left_tail_uses_the_anchored_integral():
    tau = np.arange(1, 9, dtype=np.float64) / 10.0
    q = np.full(8, math.log(2.0), dtype=np.float64)
    q[0] = 0.0
    # The first two points follow q(tau) = log(tau / 0.1), so lambda_L=1.
    expected = tau[0] * (q[0] - 1.0) + _mid_area(q, tau) + (1.0 - tau[-1]) * q[-1]

    actual = quantile_dist_mean_batch(
        torch.from_numpy(q).unsqueeze(0),
        tau,
        tail_outer_n=2,
    )

    assert actual.item() == pytest.approx(expected)


def test_exponential_right_tail_uses_the_anchored_integral():
    tau = np.arange(1, 9, dtype=np.float64) / 10.0
    q = np.zeros(8, dtype=np.float64)
    q[-1] = math.log(1.5)
    # The last two points follow q(tau) = log(0.3 / (1 - tau)), so lambda_R=1.
    expected = tau[0] * q[0] + _mid_area(q, tau) + (1.0 - tau[-1]) * (q[-1] + 1.0)

    actual = quantile_dist_mean_batch(
        torch.from_numpy(q).unsqueeze(0),
        tau,
        tail_outer_n=2,
    )

    assert actual.item() == pytest.approx(expected)


@pytest.mark.parametrize(
    "tau",
    [
        np.array([0.1, 0.2]),
        np.array([0.1, 0.1, 0.9]),
        np.array([0.0, 0.5, 0.9]),
        np.array([0.1, 0.5, np.nan]),
    ],
)
def test_quantile_mean_helpers_reject_invalid_tau_grids(tau):
    q_numpy = np.array([[0.0, 1.0, 2.0]])
    q_torch = torch.from_numpy(q_numpy)

    with pytest.raises(ValueError, match="tau"):
        quantile_dist_mean_numpy(q_numpy, tau)
    with pytest.raises(ValueError, match="tau"):
        quantile_dist_mean_simple(q_torch, torch.as_tensor(tau))
    with pytest.raises(ValueError, match="tau"):
        quantile_dist_mean_batch(q_torch, tau)


def test_torch_quantile_mean_rejects_integer_predictions():
    with pytest.raises(TypeError, match="floating-point"):
        quantile_dist_mean_simple(
            torch.tensor([[0, 1, 2]]),
            torch.tensor([0.1, 0.5, 0.9]),
        )


def test_dense_grid_with_bfloat16_predictions_integrates_in_float32():
    q = torch.linspace(-2.0, 3.0, 999, dtype=torch.bfloat16).unsqueeze(0)
    tau = torch.arange(1, 1000, dtype=torch.float64) / 1000.0

    result = quantile_dist_mean_simple(q, tau)

    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.tensor([0.5]), atol=2e-3, rtol=0.0)
