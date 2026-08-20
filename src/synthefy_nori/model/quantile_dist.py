"""Quantile-distribution point estimate decoders.

Convert K predicted quantiles (at τ levels) into a single point estimate
by treating them as a piecewise-linear quantile function and computing the analytical
mean of the implied distribution.

Two variants:
  quantile_dist_mean_batch  — sort + analytical mean with exp tail extrapolation
  quantile_dist_mean_simple — sort + analytical mean, no tail correction (faster)
"""

from __future__ import annotations

import numpy as np
import torch


def _enforce_monotone(q: torch.Tensor) -> torch.Tensor:
    """Sort quantiles along the last dim to enforce monotonicity."""
    return q.sort(dim=-1).values


def _validate_tau_numpy(tau: np.ndarray, quantile_count: int) -> np.ndarray:
    tau = np.asarray(tau)
    if tau.shape != (quantile_count,):
        raise ValueError(
            f"tau must have shape ({quantile_count},), got {tau.shape}"
        )
    if quantile_count < 1:
        raise ValueError("q must contain at least one quantile")
    if (
        not np.isfinite(tau).all()
        or np.any(tau <= 0.0)
        or np.any(tau >= 1.0)
        or np.any(np.diff(tau) <= 0.0)
    ):
        raise ValueError("tau must be finite, strictly increasing values in (0, 1)")
    return tau


def _validate_tau_torch(
    tau: torch.Tensor,
    quantile_count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if tau.shape != (quantile_count,):
        raise ValueError(
            f"tau must have shape ({quantile_count},), got {tuple(tau.shape)}"
        )
    if quantile_count < 1:
        raise ValueError("q must contain at least one quantile")
    if not tau.is_floating_point():
        raise TypeError("tau must be a floating-point tensor")
    # Validate before any lossy cast. A valid 999-level float64/float32 grid
    # has only ~506 distinct values in bfloat16, so checking after conversion
    # would reject the released model's grid.
    tau = tau.to(device=device)
    if (
        not bool(torch.isfinite(tau).all())
        or bool((tau <= 0.0).any())
        or bool((tau >= 1.0).any())
        or bool((torch.diff(tau) <= 0.0).any())
    ):
        raise ValueError("tau must be finite, strictly increasing values in (0, 1)")
    return tau


def _quantile_compute_dtype(q: torch.Tensor) -> torch.dtype:
    # Accumulating hundreds of narrow tau intervals in fp16/bf16 both loses
    # levels and needlessly rounds the mean. Keep fp32/fp64 inputs unchanged;
    # promote lower-precision prediction banks to float32 for integration.
    if q.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return q.dtype


def quantile_dist_mean_numpy(
    q: np.ndarray,
    tau: np.ndarray,
    *,
    enforce_monotone_first: bool = True,
) -> np.ndarray:
    """NumPy counterpart of :func:`quantile_dist_mean_simple`.

    The outer quantiles are extended as constants to 0 and 1, matching the
    point-estimate decoder used by inference.
    """
    q = np.asarray(q)
    if q.ndim == 0:
        raise ValueError("q must have a quantile axis")
    tau = _validate_tau_numpy(tau, q.shape[-1])
    if enforce_monotone_first:
        q = np.sort(q, axis=-1)
    if q.shape[-1] <= 1:
        return np.squeeze(q, axis=-1)

    left_area = tau[0] * q[..., 0]
    right_area = (1.0 - tau[-1]) * q[..., -1]
    mid_area = (
        0.5 * (q[..., :-1] + q[..., 1:]) * np.diff(tau)
    ).sum(axis=-1)
    return left_area + mid_area + right_area


def quantile_dist_mean_simple(
    q: torch.Tensor,
    tau: torch.Tensor,
    *,
    enforce_monotone_first: bool = True,
) -> torch.Tensor:
    """Analytical mean of the piecewise-linear quantile distribution.

    Given K quantile predictions q[..., k] at τ levels tau[k], the implied
    quantile function (inverse CDF) is piecewise linear between consecutive
    (τ, q) points. The mean is the area under that quantile function,
    computed as a trapezoidal integral: Σ 0.5*(q[k]+q[k+1])*(τ[k+1]-τ[k]).

    Boundaries: [0, τ_0] uses q[0]; [τ_{K-1}, 1] uses q[K-1].
    """
    if q.ndim == 0:
        raise ValueError("q must have a quantile axis")
    if not q.is_floating_point():
        raise TypeError("q must be a floating-point tensor")
    tau = _validate_tau_torch(
        tau,
        q.shape[-1],
        device=q.device,
    )
    compute_dtype = _quantile_compute_dtype(q)
    q = q.to(dtype=compute_dtype)
    tau = tau.to(dtype=compute_dtype)
    if enforce_monotone_first:
        q = _enforce_monotone(q)
    K = q.shape[-1]
    if K <= 1:
        return q.squeeze(-1)

    left_area = tau[0] * q[..., 0]
    right_area = (1.0 - tau[-1]) * q[..., -1]

    dτ = tau[1:] - tau[:-1]
    mid_vals = 0.5 * (q[..., :-1] + q[..., 1:])
    mid_area = (mid_vals * dτ).sum(dim=-1)

    return left_area + mid_area + right_area


def quantile_dist_mean_batch(
    q: torch.Tensor,
    tau_np: np.ndarray,
    *,
    enforce_monotone_first: bool = True,
    tail_outer_n: int = 20,
) -> torch.Tensor:
    """Analytical mean with exponential tail extrapolation.

    Like quantile_dist_mean_simple but fits exponential tails beyond the
    outermost τ levels for better extreme-value handling on heavy-tailed
    distributions.

    The left tail [0, τ_0] is modeled as q(τ) = q_0 - λ_L * ln(τ_0/τ),
    and the right tail [τ_{K-1}, 1] as q(τ) = q_{K-1} + λ_R * ln((1-τ_{K-1})/(1-τ)).
    λ is estimated from the slope of the outer `tail_outer_n` quantiles.
    """
    if q.ndim == 0:
        raise ValueError("q must have a quantile axis")
    if not q.is_floating_point():
        raise TypeError("q must be a floating-point tensor")
    tau_np = _validate_tau_numpy(tau_np, q.shape[-1])
    compute_dtype = _quantile_compute_dtype(q)
    q = q.to(dtype=compute_dtype)
    if enforce_monotone_first:
        q = _enforce_monotone(q)
    K = q.shape[-1]
    if K < 8:
        tau_t = torch.as_tensor(tau_np, device=q.device, dtype=compute_dtype)
        return quantile_dist_mean_simple(q, tau_t, enforce_monotone_first=False)

    tau_t = torch.as_tensor(tau_np, device=q.device, dtype=compute_dtype)

    dτ = tau_t[1:] - tau_t[:-1]
    mid_vals = 0.5 * (q[..., :-1] + q[..., 1:])
    mid_area = (mid_vals * dτ).sum(dim=-1)

    n = min(tail_outer_n, K // 4)

    # Left tail: estimate λ_L from slope of first n quantiles
    q_left = q[..., :n]
    tau_left = tau_t[:n]
    log_tau_left = torch.log(tau_left.clamp(min=1e-12))
    dq_left = q_left[..., -1] - q_left[..., 0]
    dlog_left = log_tau_left[-1] - log_tau_left[0]
    lambda_L = (dq_left / dlog_left.clamp(min=1e-12)).clamp(min=0)
    # q(tau) = q_0 + lambda_L * log(tau / tau_0). Its anchored
    # primitive over [0, tau_0] is tau_0 * (q_0 - lambda_L); no
    # log(tau_0) term remains after evaluating the boundary.
    left_area = tau_t[0] * (q[..., 0] - lambda_L)

    # Right tail: estimate λ_R from slope of last n quantiles
    q_right = q[..., -n:]
    tau_right = tau_t[-n:]
    log_1mtau_right = torch.log((1.0 - tau_right).clamp(min=1e-12))
    dq_right = q_right[..., -1] - q_right[..., 0]
    dlog_right = log_1mtau_right[0] - log_1mtau_right[-1]
    lambda_R = (dq_right / dlog_right.clamp(min=1e-12)).clamp(min=0)
    rem = 1.0 - tau_t[-1]
    # With u = 1 - tau, the anchored right-tail integral is
    # integral_0^rem [q_last + lambda_R * log(rem / u)] du
    # = rem * (q_last + lambda_R).
    right_area = rem * (q[..., -1] + lambda_R)

    # Fallback: if tail estimate is non-finite, use simple boundary
    left_simple = tau_t[0] * q[..., 0]
    right_simple = (1.0 - tau_t[-1]) * q[..., -1]
    left_area = torch.where(torch.isfinite(left_area), left_area, left_simple)
    right_area = torch.where(torch.isfinite(right_area), right_area, right_simple)

    return left_area + mid_area + right_area
