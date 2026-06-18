"""Sequential feature selection for Nori (thin wrapper over sklearn).

Uses ``sklearn.feature_selection.SequentialFeatureSelector`` with cross-validation
and reports baseline-vs-selected CV scores. ``NoriRegressor`` is cloned per fold
(it is a ``BaseEstimator``), and scored with R² by default (``RegressorMixin``).

In-context inference runs on every CV fit, so cost multiplies quickly — best under
a few thousand samples and a modest feature count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FeatureSelectionResult:
    selector: Any
    support_mask: Any
    selected_indices: list
    selected_names: Optional[list]
    baseline_score_mean: float
    baseline_score_std: float
    selected_score_mean: float
    selected_score_std: float


def feature_selection(
    estimator,
    X,
    y,
    n_features_to_select,
    *,
    feature_names: Optional[list] = None,
    cv=5,
    scoring=None,
    direction: str = "forward",
    n_jobs: Optional[int] = None,
    tol: Optional[float] = None,
    **kwargs: Any,
) -> FeatureSelectionResult:
    """Select a minimal feature subset that preserves CV performance.

    Args mirror ``SequentialFeatureSelector``; ``n_features_to_select`` may be an
    int, a fraction, or ``"auto"`` (with ``tol``). Returns a
    :class:`FeatureSelectionResult` with the fitted selector, selected indices /
    names, and baseline-vs-selected CV scores.
    """
    import numpy as np
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.model_selection import cross_val_score

    X = np.asarray(X)
    y = np.asarray(y)

    sfs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_to_select,
        cv=cv,
        scoring=scoring,
        direction=direction,
        n_jobs=n_jobs,
        tol=tol,
        **kwargs,
    )
    sfs.fit(X, y)
    mask = sfs.get_support()
    idx = [int(i) for i in np.where(mask)[0]]

    base = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    sel = cross_val_score(estimator, X[:, mask], y, cv=cv, scoring=scoring)

    return FeatureSelectionResult(
        selector=sfs,
        support_mask=mask,
        selected_indices=idx,
        selected_names=([feature_names[i] for i in idx] if feature_names is not None else None),
        baseline_score_mean=float(base.mean()),
        baseline_score_std=float(base.std()),
        selected_score_mean=float(sel.mean()),
        selected_score_std=float(sel.std()),
    )
