"""Model-faithful feature importance for a fitted ``NoriRegressor``.

Both methods attribute importance to the *raw input columns*, so they honour
Nori's full internal preprocessing (Yeo-Johnson → polynomial interactions →
TruncatedSVD → transformer) — the attribution is of the deployed model, not of
some surrogate:

  * :func:`nori_permutation_importance` — shuffle each raw column, run the fixed
    fitted pipeline, measure the skill drop. Higher drop ⇒ more important. Cheap,
    accuracy-relevant, and the recommended ranking for pruning.
  * :func:`nori_shap_importance` — imputation-based Shapley values via shapiq
    (reuses :mod:`synthefy_nori.interpretability.shapiq`). Needs the
    ``interpretability`` extra (``pip install "synthefy-nori[interpretability]"``).
"""
import numpy as np


def nori_permutation_importance(model, X, y, metric, *, n_repeats=3, random_state=0):
    """Permutation importance of a *fitted* model on (X, y).

    The model is used exactly as at inference — ``model.predict`` is called
    unchanged. The only additions are (1) shuffling one raw column at a time and
    (2) storing the resulting skill drop. (Subsample X, y in the caller if the
    eval set is large.)

    Args:
        model: fitted estimator with ``.predict`` (a ``NoriRegressor``).
        X, y: evaluation split (held-out is best-practice).
        metric: ``fn(y_true, y_pred) -> float``; higher = better (e.g. R² / ROC-AUC).
        n_repeats: shuffles averaged per feature.
        random_state: RNG seed.

    Returns:
        ``(importance, base_skill)`` — importance is length-d (skill drop when each
        raw column is shuffled); base_skill is the unshuffled metric value.
    """
    rng = np.random.RandomState(random_state)
    base = metric(y, model.predict(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drop = 0.0
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(len(X)), j]        # shuffle column j
            drop += base - metric(y, model.predict(Xp))     # store the skill drop
        imp[j] = drop / n_repeats
    return imp, base


def nori_shap_importance(model, X_query, background, *, budget=256, max_order=1, random_state=0):
    """Mean |Shapley value| per feature over ``X_query`` rows (imputation-based).

    Args:
        model: fitted ``NoriRegressor``.
        X_query: rows to explain, shape (q, d).
        background: reference rows for the imputer, shape (b, d).
        budget: coalition budget per explained row (higher = more exact, slower).
        max_order: 1 for single-feature attributions.
        random_state: seed for coalition sampling.

    Returns:
        length-d array of mean absolute Shapley values.
    """
    try:                                             # optional dep: the interpretability extra
        from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer
    except ImportError as exc:
        raise ImportError("shapiq is required for SHAP importance; install it with "
                          'pip install "synthefy-nori[interpretability]"') from exc

    X_query = np.asarray(X_query, np.float64)
    background = np.asarray(background, np.float64)
    d = X_query.shape[1]
    expl = get_nori_imputation_explainer(model, background, index="SV", max_order=max_order,
                                         imputer="baseline", random_state=random_state)
    acc = np.zeros(d)
    for row in X_query:
        iv = expl.explain(row.reshape(1, -1), budget=budget)
        acc += np.abs(first_order_vector(iv, d))
    return acc / max(len(X_query), 1)


def first_order_vector(iv, d):
    """Extract single-feature (order-1) values from a shapiq ``InteractionValues`` into a length-d vector."""
    vec = np.zeros(d)
    lookup = getattr(iv, "interaction_lookup", None)
    if lookup is not None:
        vals = np.asarray(iv.values)
        for coalition, pos in lookup.items():
            if len(coalition) == 1:
                vec[coalition[0]] = vals[pos]
        return vec
    for coalition, val in dict(getattr(iv, "dictionary", {})).items():  # fallback for older shapiq
        if len(coalition) == 1:
            vec[coalition[0]] = val
    return vec
