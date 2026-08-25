"""Fit and introspect a glass-box EBM (Explainable Boosting Machine, a GA²M).

An EBM is fully specified by ``ŷ = intercept + Σ_j f_j(x_j) + Σ_jk f_jk(x_j, x_k)``
where each ``f`` is a binned lookup table — the per-bin scores ARE the weights.
:func:`ebm_structure` serialises that to plain Python so a fitted model can be
read (or shipped as JSON) without unpickling.
"""
import numpy as np

from synthefy_nori.explainability._common import clip_inf_edges, shape_direction


def fit_ebm(X, y, feature_names, task, *, interactions=None, outer_bags=4, random_state=0):
    """Fit an EBM on X, y: a classifier for ``classification``/``multiclass``, else a regressor.

    Pairwise interactions default to 10 when d<=32, else 0 (they get costly and hard to read in
    high dimensions). Multiclass forces 0: interpret's EBM does not support interaction terms
    for K>2, so anything else would either error or be silently dropped.
    """
    # deferred on purpose: keeps `import synthefy_nori.explainability` free of interpret
    try:                                             # optional dep: the explainability extra
        if task in ("classification", "multiclass"):
            from interpret.glassbox import ExplainableBoostingClassifier as EBM
        else:
            from interpret.glassbox import ExplainableBoostingRegressor as EBM
    except ImportError as exc:
        raise ImportError("interpret-core is required to fit a glass-box EBM; install it with "
                          'pip install "synthefy-nori[explainability]"') from exc
    if task == "multiclass":
        interactions = 0                             # unsupported for K>2 by interpret
    elif interactions is None:
        interactions = 0 if X.shape[1] > 32 else 10
    return EBM(interactions=interactions, outer_bags=outer_bags, random_state=random_state,
               feature_names=list(feature_names)).fit(X, y)


def ebm_score(model, X, task):
    """Prediction fed to the skill metric.

    Binary gives the class-1 probability (one column, for ROC-AUC), multiclass the full
    ``(n, K)`` probability matrix (for the macro one-vs-rest AUC), regression the value.
    """
    if task == "multiclass":
        return model.predict_proba(X)
    if task == "classification":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def ebm_structure(model, *, include_interactions=True):
    """Serialise a fitted EBM to plain Python: intercept, per-term importances, each
    main-effect shape function (bin edges + per-bin scores), and the pairwise interaction
    tables (2-D scores).

    Interactions are included by default because an EBM's prediction is
    ``intercept + sum_j f_j(x_j) + sum_jk f_jk(x_j, x_k)``: drop the pairwise terms and the
    serialised model no longer reproduces the model it claims to describe, and each main
    effect read on its own silently absorbs whatever the pairwise terms carry. Pass
    ``include_interactions=False`` only when you explicitly want main effects alone.
    """
    g = model.explain_global()
    overall = g.data()
    terms = [{"term": str(t), "importance": float(s)}
             for t, s in zip(overall["names"], overall["scores"])]
    shapes, interactions = [], []
    # term arity, NOT a "&" substring test: a feature legitimately named e.g. "R&D spend"
    # would otherwise be misread as an interaction and vanish from the serialised model.
    for i, (tname, feats) in enumerate(zip(model.term_names_, model.term_features_)):
        d_i = g.data(i)
        if len(feats) > 1:
            if include_interactions:
                interactions.append({
                    "term": str(tname),
                    "feature_indices": [int(j) for j in feats],
                    "left_edges": _flt(d_i.get("left_names", [])),
                    "right_edges": _flt(d_i.get("right_names", [])),
                    "scores": np.asarray(d_i.get("scores", []), float).round(6).tolist(),
                })
            continue
        raw_scores = np.asarray(d_i.get("scores", []), float)
        if raw_scores.ndim > 1:
            # multiclass: one score column per class, so there is no single trend to name
            shapes.append({
                "feature": str(tname),
                "bin_edges": _flt(d_i.get("names", [])),
                "scores_per_class": raw_scores.round(6).tolist(),
                "direction": None,
            })
            continue
        shapes.append({
            "feature": str(tname),
            "bin_edges": _flt(d_i.get("names", [])),
            "scores": _flt(raw_scores),
            "direction": shape_direction(raw_scores),
        })
    out = {"intercept": float(np.asarray(model.intercept_).ravel()[0]),
           "term_importances": terms, "shape_functions": shapes}
    if include_interactions:
        out["interactions"] = interactions
    return out


def _flt(a):
    """Clip ±inf bin edges, then round to a plain float list for JSON."""
    return [round(float(x), 6) for x in clip_inf_edges(a)]
