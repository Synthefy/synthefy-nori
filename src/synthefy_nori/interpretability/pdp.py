"""Partial dependence / ICE plots for Nori (thin wrapper over sklearn).

Because ``NoriRegressor`` is a scikit-learn regressor, this is just a convenience
wrapper around ``PartialDependenceDisplay.from_estimator``. Each grid point is a
``predict`` call, so cost scales with ``grid_resolution`` × samples × features —
limit to a few features at a time.
"""

from __future__ import annotations

from typing import Any


def partial_dependence_plots(
    estimator,
    X,
    features,
    *,
    kind: str = "average",
    grid_resolution: int = 20,
    ax=None,
    **kwargs: Any,
):
    """Plot partial dependence (and/or ICE) for a fitted ``NoriRegressor``.

    Args:
        estimator: a fitted ``NoriRegressor``.
        X: input features used to build the grid / average over, ``(n, d)``.
        features: feature indices for 1-D plots, or ``(i, j)`` tuples for 2-D
            interaction plots.
        kind: ``"average"`` (PDP), ``"individual"`` (ICE), or ``"both"``.
        grid_resolution: grid points per feature axis.
        ax: optional matplotlib axes.
        **kwargs: forwarded to ``PartialDependenceDisplay.from_estimator``.

    Returns:
        ``sklearn.inspection.PartialDependenceDisplay``.
    """
    import numpy as np
    from sklearn.inspection import PartialDependenceDisplay

    return PartialDependenceDisplay.from_estimator(
        estimator,
        np.asarray(X),
        features,
        kind=kind,
        grid_resolution=grid_resolution,
        ax=ax,
        **kwargs,
    )
