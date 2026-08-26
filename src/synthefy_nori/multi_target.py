"""Policies and numerical helpers for compositional multi-target regression."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


MultiTargetPredictionStrategy = Literal["independent", "copula", "autoregressive"]
MULTI_TARGET_PREDICTION_STRATEGIES = ("independent", "copula", "autoregressive")
DEFAULT_MULTI_TARGET_PREDICTION_STRATEGY: MultiTargetPredictionStrategy = "copula"
MAX_MULTI_TARGET_RANDOM_STATE = 2**32 - 1


class MultiTargetPredictionPolicy(BaseModel):
    """Advanced controls for multi-target joint prediction.

    The model is frozen and rejects unknown fields so a misspelled control never
    becomes a silently ignored experiment setting. At prediction time an object
    with only some explicitly set fields acts as a partial override.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_draws: PositiveInt = Field(
        300,
        description="Joint Monte Carlo draws returned or averaged per query row.",
    )
    random_state: int | None = Field(
        0,
        ge=0,
        le=MAX_MULTI_TARGET_RANDOM_STATE,
        description="Seed for reproducible copula and autoregressive sampling.",
    )
    copula_cv: int = Field(
        default=5,
        ge=2,
        description="Cross-fitting folds used to estimate copula residual ranks.",
    )
    copula_pit_jitter: float = Field(
        default=1e-4,
        ge=0.0,
        description="Uniform jitter applied to tied copula probability transforms.",
    )
    autoregressive_n_orders: PositiveInt = Field(
        3,
        description=(
            "Number of automatically generated target permutations. Capped at the "
            "number of unique permutations; do not set with autoregressive_orders."
        ),
    )
    autoregressive_orders: list[list[int]] | None = Field(
        default=None,
        description=(
            "Explicit target-index permutations used exactly as supplied. Every order "
            "must contain each target once. Orders control predictive factorization, "
            "not causality, and are fit-dependent."
        ),
    )

    @model_validator(mode="after")
    def _validate_explicit_orders(self):
        orders = self.autoregressive_orders
        if orders is None:
            return self
        if "autoregressive_n_orders" in self.model_fields_set:
            raise ValueError(
                "autoregressive_orders cannot be combined with explicitly supplied autoregressive_n_orders"
            )
        if not orders or any(not order for order in orders):
            raise ValueError("autoregressive_orders must contain non-empty target orders")
        keys = [tuple(order) for order in orders]
        if len(keys) != len(set(keys)):
            raise ValueError("autoregressive_orders must not contain duplicate orders")
        return self

    @classmethod
    def coerce(cls, value: "MultiTargetPredictionPolicy | dict | None") -> "MultiTargetPredictionPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("multi_target_prediction_policy must be a MultiTargetPredictionPolicy, dict, or None")

    def merge_prediction_override(
        self, override: "MultiTargetPredictionPolicy | dict | None"
    ) -> "MultiTargetPredictionPolicy":
        if override is None:
            return self
        candidate = self.coerce(override)
        updates = candidate.model_dump(exclude_unset=True)
        merged = self.model_copy(update=updates)
        for field in (
            "copula_cv",
            "copula_pit_jitter",
            "autoregressive_n_orders",
            "autoregressive_orders",
        ):
            if getattr(merged, field) != getattr(self, field):
                raise ValueError(
                    f"multi_target_prediction_policy.{field} is fit-dependent and "
                    "cannot change at predict time; refit the estimator instead"
                )
        return merged


def cdf_from_quantile_bank(quantiles: np.ndarray, taus: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Evaluate each row's monotone piecewise-linear CDF with clamped tails."""
    Q = np.asarray(quantiles, dtype=np.float64)
    levels = np.asarray(taus, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    if Q.ndim != 2 or Q.shape[0] != y.shape[0] or Q.shape[1] != levels.shape[0]:
        raise ValueError("quantile bank, tau levels, and values have incompatible shapes")
    if levels.shape[0] < 2:
        raise ValueError("quantile bank must contain at least two probability levels")
    out = np.empty(y.shape[0], dtype=np.float64)
    for row in range(Q.shape[0]):
        # A flat stretch of the quantile function is an atom. Interpolation from
        # below must approach the tied value's first probability; at the value
        # itself (and when departing above it), the CDF uses its last probability.
        xp, first, counts = np.unique(Q[row], return_index=True, return_counts=True)
        first_probabilities = levels[first]
        last_probabilities = levels[first + counts - 1]
        if y[row] < xp[0]:
            out[row] = 0.0
        elif y[row] >= xp[-1]:
            # The inverse CDF clamps levels above the last native tau to the
            # maximum quantile. That creates an endpoint atom, so the CDF is
            # one at the maximum itself as well as above it. This branch also
            # gives a constant quantile bank the correct point-mass CDF.
            out[row] = 1.0
        else:
            upper = int(np.searchsorted(xp, y[row], side="left"))
            if xp[upper] == y[row]:
                out[row] = last_probabilities[upper]
            else:
                lower = upper - 1
                weight = (y[row] - xp[lower]) / (xp[upper] - xp[lower])
                out[row] = (1.0 - weight) * last_probabilities[lower] + weight * first_probabilities[upper]
    return out


def quantiles_from_levels(quantiles: np.ndarray, taus: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Invert row-specific quantile banks at a matrix of probability levels."""
    Q = np.asarray(quantiles, dtype=np.float64)
    tau = np.asarray(taus, dtype=np.float64)
    U = np.asarray(levels, dtype=np.float64)
    if Q.ndim != 2 or U.ndim != 2 or Q.shape[0] != U.shape[0]:
        raise ValueError("quantile bank and probability levels have incompatible shapes")
    if Q.shape[1] != tau.shape[0]:
        raise ValueError("quantile bank width does not match tau levels")
    if tau.shape[0] < 2:
        raise ValueError("quantile bank must contain at least two probability levels")
    clipped = np.clip(U, tau[0], tau[-1])
    hi = np.searchsorted(tau, clipped, side="right")
    hi = np.clip(hi, 1, tau.shape[0] - 1)
    lo = hi - 1
    tau_lo = tau[lo]
    tau_hi = tau[hi]
    weight = np.divide(
        clipped - tau_lo,
        tau_hi - tau_lo,
        out=np.zeros_like(clipped),
        where=tau_hi > tau_lo,
    )
    rows = np.arange(Q.shape[0])[:, None]
    result = (1.0 - weight) * Q[rows, lo] + weight * Q[rows, hi]
    result = np.where(clipped <= tau[0], Q[:, :1], result)
    result = np.where(clipped >= tau[-1], Q[:, -1:], result)
    return result


def build_target_orders(n_outputs: int, n_orders: int, random_state: int | None) -> list[np.ndarray]:
    """Build deterministic distinct target orders, anchored by identity.

    Requests above ``n_outputs!`` are capped rather than repeating a permutation
    and unintentionally giving one factorization extra ensemble weight.
    """
    n_orders = min(n_orders, math.factorial(n_outputs))
    orders = [np.arange(n_outputs, dtype=int)]
    if n_orders == 1:
        return orders
    rng = np.random.default_rng(random_state)
    seen = {tuple(orders[0])}
    max_unique = math.factorial(n_outputs)
    attempts = 0
    while len(orders) < n_orders and attempts < 50 * n_orders:
        candidate = rng.permutation(n_outputs)
        attempts += 1
        key = tuple(candidate)
        if key in seen and len(seen) < max_unique:
            continue
        seen.add(key)
        orders.append(candidate)
    return orders
