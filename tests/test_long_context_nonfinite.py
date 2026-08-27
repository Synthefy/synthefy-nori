"""Regression tests for issue #439 — long-context predictions collapsing to the
context target mean.

Two independent defects produced one silent failure:

1. ``QASSMaxScaling.forward`` built the ``log(n)`` attention scale by first
   materializing the context row count ``n`` in ``q.dtype``. Inference autocasts
   to fp16 on CUDA (the trainer uses bf16), and fp16's largest finite value is
   65504 — so any context of 65520+ rows became ``inf`` *before* ``log()`` ran.
   The inf flowed through ``base_scale`` into ``q`` and every prediction went
   NaN. The boundary is 65520, not 65536: fp16 round-to-nearest sends 65520 and
   up to inf, while 65505..65519 still round down to 65504.

2. ``NoriPredictor`` then replaced the NaNs with 0.0 and reported success. The
   predictor works in standardized target space, so a zeroed prediction
   denormalizes to exactly ``y_mean`` — a finite, plausible-looking *constant*
   that scores at chance (ROC AUC 0.5) with no error raised anywhere.

The fp16 tests below run on CPU: the overflow is a dtype cast, not a kernel or
device behavior, so they reproduce the original bug in milliseconds. The
end-to-end boundary check needs a real 65k-row forward and is marked ``slow``.
"""

import math

import numpy as np
import pytest
import torch

from synthefy_nori.inference.predictor import NoriPredictor
from synthefy_nori.model.layer import QASSMaxScaling

# fp16 max finite is 65504; round-to-nearest-even sends 65520+ to inf.
FP16_MAX_FINITE = 65504
FP16_OVERFLOW_START = 65520


# ---------------------------------------------------------------------------
# 1. Root cause: log(n) must be computed before the cast, not after.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key_len",
    [
        65519,  # last row count fp16 still rounds to a finite value
        FP16_OVERFLOW_START,  # first row count that overflowed fp16
        65535,
        65536,
        65537,  # the boundary named in the issue
        70000,  # a reported failing context
        100000,  # a reported failing context
    ],
)
@pytest.mark.parametrize("mode", ["log_only", "base_only", "full"])
def test_qassmax_finite_above_fp16_row_count_limit(key_len, mode):
    """A context longer than fp16 can represent must not poison the q scale."""
    m = QASSMaxScaling(num_heads=2, head_dim=4, qass_mode=mode).half()
    q = torch.randn(1, 8, 2, 4, dtype=torch.float16)

    out = m(q, key_len=key_len)

    assert torch.isfinite(out).all(), (
        f"qass_mode={mode} produced non-finite q at key_len={key_len}: "
        f"nan={int(torch.isnan(out).sum())} inf={int(torch.isinf(out).sum())}. "
        "log(n) is being taken after casting n to fp16."
    )


@pytest.mark.parametrize("key_len", [FP16_OVERFLOW_START, 70000, 100000])
def test_qassmax_scale_is_log_n_not_inf(key_len):
    """log_only mode is exactly q*log(n) — check the value, not just finiteness.

    Finiteness alone would pass if log(n) were clamped to the fp16 max instead
    of actually computed, which would silently over-sharpen attention.
    """
    m = QASSMaxScaling(num_heads=2, head_dim=4, qass_mode="log_only").half()
    q = torch.ones(1, 4, 2, 4, dtype=torch.float16)

    scale = m(q, key_len=key_len).float().unique()

    assert scale.numel() == 1
    assert scale.item() == pytest.approx(math.log(key_len), rel=1e-3)


@pytest.mark.parametrize("key_len", [2, 100, 1024, 8000, 60000, FP16_MAX_FINITE])
@pytest.mark.parametrize("mode", ["log_only", "base_only", "full"])
def test_qassmax_below_boundary_matches_exact_log(key_len, mode):
    """Below the boundary, behavior is unchanged within fp16 tolerance.

    Computing log(n) in double and casting the ~11-magnitude result costs at
    most one fp16 ulp versus the old fp16-log-of-fp16-n, so predictions on
    every context length that used to work must not move meaningfully.
    """
    m = QASSMaxScaling(num_heads=2, head_dim=4, qass_mode=mode).half()
    q = torch.randn(1, 8, 2, 4, dtype=torch.float16)

    got = m(q, key_len=key_len).float()

    # Reference: same math in float32, exact log — no fp16 anywhere.
    ref_m = QASSMaxScaling(num_heads=2, head_dim=4, qass_mode=mode)
    ref_m.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
    expected = ref_m(q.float(), key_len=key_len)

    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


def test_qassmax_respects_fp32_and_bf16():
    """The fix must not change the dtype of the scale it builds."""
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        m = QASSMaxScaling(num_heads=2, head_dim=4, qass_mode="log_only").to(dtype)
        q = torch.randn(1, 4, 2, 4, dtype=dtype)
        out = m(q, key_len=70000)
        assert out.dtype == dtype
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 2. A non-finite forward must never become a silent constant prediction.
# ---------------------------------------------------------------------------


class _Stub:
    """Minimal carrier for the real unbound method under test."""

    mix_precision = True
    _reject_nonfinite_output = NoriPredictor._reject_nonfinite_output


def test_finite_output_passes_through_untouched():
    out = torch.tensor([1.0, -2.5, 0.0])
    got = _Stub()._reject_nonfinite_output(out, path="cached", n_train=10, n_test=3)
    assert got is out


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_output_raises_instead_of_zero_filling(bad):
    out = torch.tensor([1.0, bad, 3.0])

    with pytest.raises(RuntimeError) as exc:
        _Stub()._reject_nonfinite_output(out, path="cached (resident_bf16)", n_train=93729, n_test=4096)

    msg = str(exc.value)
    # The message has to carry enough to act on: where, how big, and why a
    # zero-fill would have looked like a working run.
    assert "non-finite" in msg
    assert "n_context=93729" in msg and "n_query=4096" in msg
    assert "resident_bf16" in msg
    assert "context target mean" in msg


def test_partial_nonfinite_also_raises():
    """One bad row is enough. Those rows would silently become the mean."""
    out = torch.cat([torch.randn(4095), torch.tensor([float("nan")])])

    with pytest.raises(RuntimeError, match="1/4096 non-finite"):
        _Stub()._reject_nonfinite_output(out, path="plain chunked loop", n_train=70000, n_test=4096)


def test_there_is_no_env_opt_out(monkeypatch):
    """No environment variable may turn the raise back into a zero-fill.

    An opt-out here can only produce a number indistinguishable from a working
    prediction while carrying no signal, which is exactly the #439 failure mode.
    """
    for name in (
        "SYNTHEFY_ALLOW_NONFINITE_PREDICTIONS",
        "SYNTHEFY_ALLOW_NONFINITE",
        "SYNTHEFY_DISABLE_NONFINITE_CHECK",
    ):
        monkeypatch.setenv(name, "1")

    with pytest.raises(RuntimeError):
        _Stub()._reject_nonfinite_output(torch.tensor([float("nan")]), path="cached", n_train=70000, n_test=1)


# ---------------------------------------------------------------------------
# 3. End-to-end at the boundary (needs a GPU + the real checkpoint).
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("n_train", [65535, 65536, 65537, 70000])
def test_long_context_predictions_are_finite_and_varying(n_train):
    """The acceptance criterion: no NaN, and no collapse to a constant."""
    from synthefy_nori.hf import download_checkpoint
    from synthefy_nori.utils.loading import load_model

    device = torch.device("cuda")
    ckpt = download_checkpoint(model="nori-6m")
    model = load_model(ckpt, mask_prediction=False).to(device).eval()

    n_test, n_feat = 2048, 12
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_train + n_test, n_feat)).astype(np.float32)
    y = X[:, 0] * 0.8 + X[:, 1] * 0.3 + rng.standard_normal(n_train + n_test) * 0.1
    y = ((y - y[:n_train].mean()) / y[:n_train].std()).astype(np.float32)

    # enabled=True with no dtype is exactly what NoriPredictor does -> fp16.
    with torch.autocast(device_type="cuda", enabled=True), torch.inference_mode():
        out = model.forward_cached_regression(
            x=torch.from_numpy(X).to(device).unsqueeze(0),
            y=torch.from_numpy(y[:n_train]).to(device).unsqueeze(0),
            eval_pos=n_train,
            row_chunk_size=2048,
        )
    pred = out["reg_output"] if isinstance(out, dict) else out
    pred = pred.float().squeeze(0)
    if pred.dim() > 1:
        pred = pred.mean(dim=-1)

    assert torch.isfinite(pred).all(), f"non-finite predictions at n_train={n_train}"
    assert pred.std().item() > 1e-3, (
        f"predictions collapsed to a constant at n_train={n_train} "
        f"(std={pred.std().item():.3e}) — the #439 failure mode"
    )
