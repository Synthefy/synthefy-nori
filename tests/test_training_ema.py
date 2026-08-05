import torch

from synthefy_nori.training.trainer import _foreach_ema_update


def test_foreach_ema_matches_scalar_update_exactly():
    generator = torch.Generator().manual_seed(17)
    current = {
        "weight": torch.randn(7, 5, generator=generator),
        "bias": torch.randn(5, generator=generator),
        "wide": torch.randn(3, generator=generator, dtype=torch.float64),
        "counter": torch.tensor(9, dtype=torch.int64),
    }
    initial = {
        name: torch.randn_like(value) if torch.is_floating_point(value) else value - 1
        for name, value in current.items()
    }
    expected = {name: value.clone() for name, value in initial.items()}
    actual = {name: value.clone() for name, value in initial.items()}
    decay = 0.999

    for name, value in current.items():
        if torch.is_floating_point(value):
            expected[name].mul_(decay).add_(value, alpha=1.0 - decay)
        else:
            expected[name].copy_(value)
    _foreach_ema_update(actual, current, decay)

    for name in expected:
        assert torch.equal(actual[name], expected[name]), name
