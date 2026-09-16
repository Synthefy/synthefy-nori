from __future__ import annotations

import pytest
import torch

from synthefy_nori.model.encoders import calc_mean, calc_std, drop_outliers


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


@pytest.mark.parametrize(("batch_size", "rows"), [(4, 10), (4, 4), (1, 10)])
def test_drop_outliers_matches_independent_episodes_and_reuses_bounds(batch_size, rows):
    generator = torch.Generator().manual_seed(17)
    offsets = torch.arange(batch_size).reshape(-1, 1, 1, 1) * 20
    x = torch.randn(batch_size, rows, 3, 2, generator=generator) + offsets
    x[:, 0, 0, 0] += 50
    x[:, 1, 1, 1] = torch.nan
    eval_pos = min(rows, 6)
    original = x.clone()

    actual, lower, upper = drop_outliers(x, std_sigma=1, eval_pos=eval_pos)
    independent = [drop_outliers(episode[None], std_sigma=1, eval_pos=eval_pos) for episode in x]
    for result, index in ((actual, 0), (lower, 1), (upper, 2)):
        torch.testing.assert_close(result, torch.cat([episode[index] for episode in independent]), equal_nan=True)
    assert lower.shape == upper.shape == (batch_size, 3, 2)
    torch.testing.assert_close(x, original, equal_nan=True)

    query = torch.randn(batch_size, 7, 3, 2, generator=generator) * 40 + offsets
    frozen, frozen_lower, frozen_upper = drop_outliers(query, lower=lower, upper=upper)
    independent_query = [
        drop_outliers(query[i : i + 1], lower=lower[i : i + 1], upper=upper[i : i + 1])[0] for i in range(batch_size)
    ]
    torch.testing.assert_close(frozen, torch.cat(independent_query))
    assert frozen_lower is lower
    assert frozen_upper is upper
