"""shapiq adapters for Nori — Shapley values and Shapley interactions.

Nori is regression-only and follows the sklearn estimator API, so the recommended
explainer is the model-agnostic, imputation-based ``shapiq.TabularExplainer``: the
training context is fixed and query features are removed by imputation, so each
coalition is a single ``predict`` call (cost controlled by ``budget``).
"""

from __future__ import annotations

from typing import Any


def _require_shapiq():
    try:
        import shapiq  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "shapiq is required for interpretability. Install with: "
            'pip install "synthefy-nori[interpretability]"'
        ) from exc
    return shapiq


def _as_predict_fn(model):
    """Wrap a fitted NoriRegressor as a 1-D-returning predict callable.

    Passing an explicit prediction function makes the explainer independent of
    shapiq's per-version model auto-detection, and pins the regression point
    estimate to ``predict``'s default (the distribution mean).
    """
    import numpy as np

    if not hasattr(model, "predict"):
        raise TypeError("model must be a fitted NoriRegressor (or expose .predict)")

    def predict(x):
        return np.asarray(model.predict(np.asarray(x)), dtype=np.float64).reshape(-1)

    return predict


def get_nori_imputation_explainer(
    model,
    data,
    *,
    index: str = "k-SII",
    max_order: int = 2,
    imputer: str = "baseline",
    random_state: int | None = 0,
    **kwargs: Any,
):
    """Recommended explainer: imputation-based feature removal (model-agnostic).

    Args:
        model: a fitted ``NoriRegressor``.
        data: background data for the imputer, shape ``(n, n_features)``.
        index: Shapley index — ``"SV"`` (values), ``"k-SII"`` (k-Shapley
            interactions; with ``max_order=1`` it reduces to standard Shapley
            values), or any other shapiq tabular index.
        max_order: maximum interaction order (``1`` = single-feature attributions).
        imputer: ``"baseline"`` (one forward per coalition; recommended),
            ``"marginal"`` or ``"conditional"`` (multi-sample, much slower).
        random_state: seed for reproducible coalition sampling.
        **kwargs: forwarded to ``shapiq.TabularExplainer``.

    Returns:
        ``shapiq.TabularExplainer``. Call ``.explain(x, budget=N)`` with ``x`` of
        shape ``(1, n_features)``; the result exposes ``.plot_waterfall()`` etc.
    """
    import numpy as np

    shapiq = _require_shapiq()
    return shapiq.TabularExplainer(
        model=_as_predict_fn(model),
        data=np.asarray(data, dtype=np.float64),
        index=index,
        max_order=max_order,
        imputer=imputer,
        random_state=random_state,
        **kwargs,
    )