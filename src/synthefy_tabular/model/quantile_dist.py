"""Quantile-distribution point estimate decoders (V13).

Convert K predicted quantiles (at τ levels) into a single point estimate
by treating them as a piecewise-linear CDF and computing the analytical
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


def quantile_dist_mean_simple(
    q: torch.Tensor,
    tau: torch.Tensor,
    *,
    enforce_monotone_first: bool = True,
) -> torch.Tensor:
    """Analytical mean of the piecewise-linear quantile distribution.

    Given K quantile predictions q[..., k] at τ levels tau[k], the implied
    CDF is piecewise linear between consecutive (τ, q) points. The mean of
    this distribution is the area under the inverse-CDF (quantile function),
    computed as a trapezoidal integral: Σ 0.5*(q[k]+q[k+1])*(τ[k+1]-τ[k]).

    Boundaries: [0, τ_0] uses q[0]; [τ_{K-1}, 1] uses q[K-1].
    """
    if enforce_monotone_first:
        q = _enforce_monotone(q)
    K = q.shape[-1]
    if K <= 1:
        return q.squeeze(-1)

    tau = tau.to(device=q.device, dtype=q.dtype)

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
    if enforce_monotone_first:
        q = _enforce_monotone(q)
    K = q.shape[-1]
    if K < 8:
        tau_t = torch.as_tensor(tau_np, device=q.device, dtype=q.dtype)
        return quantile_dist_mean_simple(q, tau_t, enforce_monotone_first=False)

    tau_t = torch.as_tensor(tau_np, device=q.device, dtype=q.dtype)

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
    left_area = tau_t[0] * q[..., 0] - lambda_L * tau_t[0] * (torch.log(tau_t[0].clamp(min=1e-12)) - 1.0 + 1.0)

    # Right tail: estimate λ_R from slope of last n quantiles
    q_right = q[..., -n:]
    tau_right = tau_t[-n:]
    log_1mtau_right = torch.log((1.0 - tau_right).clamp(min=1e-12))
    dq_right = q_right[..., -1] - q_right[..., 0]
    dlog_right = log_1mtau_right[0] - log_1mtau_right[-1]
    lambda_R = (dq_right / dlog_right.clamp(min=1e-12)).clamp(min=0)
    rem = 1.0 - tau_t[-1]
    right_area = rem * q[..., -1] + lambda_R * rem * (torch.log(rem.clamp(min=1e-12)) - 1.0 + 1.0)

    # Fallback: if tail estimate is non-finite, use simple boundary
    left_simple = tau_t[0] * q[..., 0]
    right_simple = (1.0 - tau_t[-1]) * q[..., -1]
    left_area = torch.where(torch.isfinite(left_area), left_area, left_simple)
    right_area = torch.where(torch.isfinite(right_area), right_area, right_simple)

    return left_area + mid_area + right_area
