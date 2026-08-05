import torch

from synthefy_nori.model.layer import RMSNorm


def _manual_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    output = x * x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return output if weight is None else output * weight


def test_rms_norm_defaults_to_historical_implementation():
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(4, 7, 128, generator=generator)
    norm = RMSNorm(128, eps=1e-5)

    assert norm.use_native is False
    assert torch.equal(norm(x), _manual_rms_norm(x, norm.weight, norm.eps))


def test_native_rms_norm_matches_manual_float32():
    generator = torch.Generator().manual_seed(92)
    x = torch.randn(4, 7, 128, generator=generator)
    norm = RMSNorm(128, eps=1e-5)
    norm.use_native = True

    expected = _manual_rms_norm(x, norm.weight, norm.eps)
    actual = norm(x)

    assert torch.equal(actual, expected)


def test_native_rms_norm_preserves_parameter_schema():
    affine = RMSNorm(32, eps=1e-5, elementwise_affine=True)
    non_affine = RMSNorm(32, eps=1e-5, elementwise_affine=False)

    assert tuple(affine.state_dict()) == ("weight",)
    assert non_affine.state_dict() == {}
