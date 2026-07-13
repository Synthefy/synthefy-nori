"""Discretize predictions onto a categorical/ordinal target's level lattice.

When a regression target only takes a small set of discrete values (ratings,
counts, wine-quality scores, ...), the caller may want a prediction *on* that
lattice, and which lattice point is optimal depends on the loss:

- ``"map-cell"`` — the mode of the discrete posterior: carve the real line into
  cells around each level (split at midpoints), integrate the predictive
  distribution's mass per cell, predict the level with the most mass. Optimal
  for exact-hit accuracy; the default.
- ``"median-cell"`` — the median of that discrete posterior (first level where
  cumulative cell mass reaches 0.5). Optimal for absolute error (MAE) on
  ordinal targets.
- ``"snap-mean"`` / ``"snap-median"`` — nearest level to the model's point
  prediction (mean / distribution median). No distribution needed, so they
  also work for bar-distribution checkpoints; ``snap-mean`` is preferable when
  the error metric penalizes large misses quadratically (QWK-like losses).

Benchmarked head-to-head on the K<=10 discrete-target datasets of
TALENT/OpenML-CTR23/TabArena (nori-monorepo experiment
``experiments/categorical-discretization``): map-cell wins accuracy/macro-F1,
median-cell wins MAE, and for R²-scored tasks the *undiscretized* mean remains
the right output — discretization always costs squared-error performance.

Discretization is strictly opt-in: nothing here runs unless the caller passes
``categorical_target=True`` to ``NoriRegressor.predict`` / ``infer`` (or uses
these helpers directly).
"""
from __future__ import annotations

import warnings

import numpy as np

DISCRETIZE_METHODS = ("map-cell", "median-cell", "snap-mean", "snap-median")
SNAP_METHODS = ("snap-mean", "snap-median")  # need only a point prediction

# Above this many distinct levels the target is unlikely to be categorical and
# cell probabilities degenerate toward one bank-quantile per cell; warn.
_MANY_LEVELS_WARN = 256


def target_levels(y) -> np.ndarray:
    """The sorted, finite, distinct values of a target column."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    levels = np.unique(arr[np.isfinite(arr)])
    if levels.size == 0:
        raise ValueError("target has no finite values to derive levels from.")
    if levels.size > _MANY_LEVELS_WARN:
        warnings.warn(
            f"categorical_target=True but the target has {levels.size} "
            "distinct values - is it really categorical? Predictions will be "
            "snapped to this large lattice.",
            stacklevel=3,
        )
    return levels


def snap_to_levels(values, levels: np.ndarray) -> np.ndarray:
    """Nearest level to each value.

    Ties go to the *upper* level (same convention as
    ``NoriPredictor._maybe_snap_discrete_y``). ``NaN`` propagates — a failed
    prediction must stay visible, not become a confident label. ``±inf`` snap
    to the extreme levels.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    out = np.full(v.shape, np.nan)
    finite = np.isfinite(v)
    if levels.size == 1:
        out[finite] = levels[0]
    else:
        vf = v[finite]
        idx = np.searchsorted(levels, vf).clip(1, levels.size - 1)
        lo, hi = levels[idx - 1], levels[idx]
        out[finite] = np.where(np.abs(vf - lo) < np.abs(vf - hi), lo, hi)
    out[v == np.inf] = levels[-1]
    out[v == -np.inf] = levels[0]
    return out


def cell_masses(Q: np.ndarray, taus: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Predictive probability mass per level cell, ``(n_rows, n_levels)``.

    Cells split the real line at midpoints between adjacent levels; row ``i``'s
    CDF is the linear interpolation of ``taus`` over its (ascending) quantile
    bank ``Q[i]``, clamped to 0 below the bank and 1 at/above its top value.
    Rows sum to 1. ``levels`` must be sorted ascending and unique
    (``target_levels`` / ``discretize_predictions`` guarantee this).
    """
    Q = np.asarray(Q, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64).ravel()
    if Q.ndim != 2 or Q.shape[1] != taus.size:
        raise ValueError(f"Q {Q.shape} does not match taus ({taus.size},).")
    levels = np.asarray(levels, dtype=np.float64)
    n = Q.shape[0]
    Qs = np.sort(Q, axis=1)
    mids = (levels[:-1] + levels[1:]) / 2.0
    if mids.size:
        F = np.stack([np.interp(mids, q, taus, left=0.0, right=1.0) for q in Qs])
        # np.interp returns taus[-1] when a midpoint EQUALS the bank top; the
        # CDF convention here is "mass at or below" -> clamp those to 1.
        F = np.where(mids[None, :] >= Qs[:, -1:], 1.0, F)
    else:  # single level -> one cell holding all the mass
        F = np.empty((n, 0), dtype=np.float64)
    F = np.hstack([np.zeros((n, 1)), F, np.ones((n, 1))])
    return np.clip(np.diff(F, axis=1), 0.0, None)  # clip float round-off


def discretize_predictions(
    method: str,
    levels,
    *,
    point: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    taus: np.ndarray | None = None,
) -> np.ndarray:
    """Map predictions onto ``levels`` with one of ``DISCRETIZE_METHODS``.

    ``levels`` may be unsorted / contain duplicates — it is normalized here.
    ``snap-*`` methods need ``point`` (the already-collapsed point prediction);
    ``*-cell`` methods need the quantile bank ``Q`` and its ``taus``.
    """
    if method not in DISCRETIZE_METHODS:
        raise ValueError(
            f"Unknown discretize method {method!r}; expected one of {DISCRETIZE_METHODS}."
        )
    levels = np.asarray(levels, dtype=np.float64).ravel()
    levels = np.unique(levels[np.isfinite(levels)])
    if levels.size == 0:
        raise ValueError("levels has no finite values.")
    if method in SNAP_METHODS:
        if point is None:
            raise ValueError(f"discretize method {method!r} requires point predictions.")
        return snap_to_levels(point, levels)
    if Q is None or taus is None:
        raise ValueError(f"discretize method {method!r} requires the quantile bank (Q, taus).")
    P = cell_masses(Q, taus, levels)
    if method == "map-cell":
        return levels[P.argmax(axis=1)]
    # median-cell: first level where cumulative mass reaches 0.5
    return levels[np.argmax(np.cumsum(P, axis=1) >= 0.5, axis=1)]
