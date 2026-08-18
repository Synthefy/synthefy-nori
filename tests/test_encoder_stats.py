from __future__ import annotations

import pytest
import torch

from synthefy_nori.model.encoders import calc_mean, calc_std


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_calc_std_broadcast_matches_materialized_mean(dim):
    torch.manual_seed(0)
    actual_input = torch.randn(3, 5, 4, requires_grad=True)
    with torch.no_grad():
        actual_input[1, 2, 3] = torch.nan
    expected_input = actual_input.detach().clone().requires_grad_(True)

    expected_mean, expected_count = calc_mean(expected_input, dim)
    materialized_mean = torch.repeat_interleave(
        expected_mean.unsqueeze(dim),
        expected_input.shape[dim],
        dim=dim,
    )
    expected = torch.sqrt(
        torch.nansum(torch.square(materialized_mean - expected_input), dim=dim) / (expected_count - 1)
    )
    actual = calc_std(actual_input, dim)

    assert torch.equal(actual, expected)
    actual.nansum().backward()
    expected.nansum().backward()
    torch.testing.assert_close(
        actual_input.grad,
        expected_input.grad,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
