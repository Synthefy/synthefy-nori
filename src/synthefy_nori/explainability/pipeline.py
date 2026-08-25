"""End-to-end: Nori feature importance → keep the fewest features for ≥95% of Nori's skill (n95)
→ fit a glass-box EBM on exactly those features. Emits a single self-describing JSON (importance
scores + the inspectable EBM model) plus a joblib of the fitted EBM for exact reuse.

Regression (metric = R²) and binary classification (metric = ROC-AUC) are auto-detected from y.

Library use:
    from synthefy_nori.explainability.pipeline import run
    res = run(Xtr, ytr, Xte, yte, feature_names)          # dict; also writes JSON + joblib if out_dir set

Command line (bundled sklearn demo — runs with zero setup):
    python -m synthefy_nori.explainability.pipeline --demo diabetes
    python -m synthefy_nori.explainability.pipeline --demo breast_cancer
    python -m synthefy_nori.explainability.pipeline --npz mydata.npz            # Xtr,ytr,Xte,yte (+feature_names)
    python -m synthefy_nori.explainability.pipeline --csv data.csv --target y
    python -m synthefy_nori.explainability.pipeline --parquet data.parquet --target y
"""
import argparse
import json
import math
import os

import joblib
import numpy as np

from synthefy_nori.explainability import data as _data
from synthefy_nori.explainability._common import (detect_task, encode_labels, make_metric,
                                                  target_classes)
from synthefy_nori.explainability._core import (SELECTION_FRACTION, SWEEP_FRACTIONS,
                                                distill_glassbox, prepare_rows, score_on_test,
                                                select_features)
from synthefy_nori.explainability.ebm import ebm_structure
from synthefy_nori.explainability.importance import nori_permutation_importance, nori_shap_importance

FRACS = SWEEP_FRACTIONS   # re-exported: the sweep grid lives in _core now


def run(Xtr, ytr, Xte, yte, feature_names, *, task="auto", method="permutation",
        nori_model="nori-6m", nori_cap=8000, perm_repeats=3, perm_eval=2000,
        shap_query=64, shap_background=200, shap_budget=256, reduce_threshold=16,
        out_dir=None, tag="model", random_state=0, verbose=True, use_test=True):
    """Run the full pipeline. Returns a dict; if ``out_dir`` is set, also writes
    ``<tag>.json`` and ``<tag>.ebm.joblib`` there.

    Feature selection only kicks in when ``d > reduce_threshold`` (default 16):
    below that, low-dimensional data tends to lose a little accuracy from trimming,
    so all features are kept and the EBM is fit on everything."""
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    d = Xtr.shape[1]
    names = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(d)]
    task = detect_task(ytr, task)
    metric, metric_name = make_metric(task)
    if task in ("classification", "multiclass"):     # work in 0..K-1 so the metric stays valid
        classes = target_classes(ytr, yte, expect=2 if task == "classification" else None)
        ytr, yte = encode_labels(ytr, classes), encode_labels(yte, classes)
    else:
        classes = None
        ytr, yte = ytr.astype(np.float32), yte.astype(np.float32)

    rng = np.random.RandomState(random_state)
    log = (lambda *m: print(*m, flush=True)) if verbose else (lambda *m: None)
    log(f"[{tag}] d={d} ntr={len(Xtr)} nte={len(Xte)} task={task} metric={metric_name} method={method}")

    rows, _mu, _features_train, features_test = prepare_rows(
        Xtr, ytr, Xte, yte, use_test=use_test,
        stratify=(ytr if task in ("classification", "multiclass") else None), rng=rng,
        nori_cap=nori_cap,
        selection_fraction=SELECTION_FRACTION, random_state=random_state)
    log(f"[{tag}] selection stage: nfit={rows.n_select_fit} neval={rows.n_select_eval} "
        + ("on the TEST split (use_test=True; selection_at_n95 is a criterion value)"
           if use_test else "carved out of train (test split untouched until final scoring)"))

    def importance_fn(fitted, eval_features, eval_target):
        """SHAP or permutation — the only part of selection that differs between methods."""
        if method == "shap":
            q = rng.choice(len(eval_features), min(shap_query, len(eval_features)), replace=False)
            bg_rows = rows.select_fit_features
            bg = bg_rows[rng.choice(len(bg_rows), min(shap_background, len(bg_rows)), replace=False)]
            budget = shap_budget if d <= 32 else max(shap_budget, 512)
            imp = nori_shap_importance(fitted, eval_features[q], bg, budget=budget,
                                       random_state=random_state)
            return imp, metric(eval_target, fitted.predict(eval_features))
        if len(eval_features) > perm_eval:           # keep the many predicts cheap
            take = rng.choice(len(eval_features), perm_eval, replace=False)
            eval_features, eval_target = eval_features[take], eval_target[take]
        return nori_permutation_importance(fitted, eval_features, eval_target, metric,
                                           n_repeats=perm_repeats, random_state=random_state)

    selection, select_model = select_features(
        rows, model=nori_model, metric=metric, metric_name=metric_name, task=task,
        importance_fn=importance_fn, retain=0.95, reduce_threshold=reduce_threshold,
        n_features=d, log=(lambda *m: log(f"[{tag}]", *m)))

    ranked = [{"feature": names[j], "index": int(j),
               "importance": round(float(selection.importance[j]), 6)} for j in selection.order]
    cols95, n95 = selection.columns, len(selection.columns)

    _full_model, nori_full, nori_n95 = score_on_test(
        rows, selection, model=nori_model, metric=metric, features_test=features_test,
        target_test=yte, task=task, full_model=select_model if rows.use_test else None)
    log(f"[{tag}] held-out test: Nori full={nori_full:+.4f}  @n95={nori_n95:+.4f}")

    ebm95, ebm95_skill, ebm_full, ebm_full_skill, fn95 = distill_glassbox(
        rows, selection, names=names, task=task, metric=metric,
        features_test=features_test, target_test=yte)

    res = {
        "tag": tag, "d": d, "task": task, "metric": metric_name,
        "classes": None if classes is None else np.asarray(classes).tolist(),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "use_test": bool(use_test),
        "importance_method": f"nori_{method}",
        "importance_base_skill": round(selection.base_score, 4),
        "importance": ranked,
        "nori_full": round(nori_full, 4), "target_95pct": round(selection.target, 4),
        # selection-stage values: the criterion pruning thresholded on, NOT held-out
        "selection_full": round(selection.select_full_score, 4),
        "selection_at_n95": round(selection.select_score, 4),
        "reduce_threshold": reduce_threshold, "reduced": selection.reduced,
        "n95": n95, "pct95": round(100 * n95 / d, 1), "nori_at_n95": round(nori_n95, 4),
        "selected_features": fn95, "selected_indices": [int(c) for c in cols95],
        "ebm_at_n95": round(ebm95_skill, 4), "ebm_full": round(ebm_full_skill, 4),
        "sweep": selection.curve, "ebm_model": ebm_structure(ebm95),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        jp = os.path.join(out_dir, f"{tag}.ebm.joblib")
        joblib.dump({"model": ebm95, "feature_indices": [int(c) for c in cols95],
                     "feature_names": fn95, "task": task}, jp)
        res["ebm_joblib"] = os.path.abspath(jp)
        with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        log(f"[{tag}] wrote {os.path.join(out_dir, tag + '.json')}  +  {jp}")

    log(f"[{tag}] n95={n95} ({res['pct95']}% of {d})  Nori full={nori_full:+.3f} @n95={nori_n95:+.3f}"
        f" | EBM @n95={ebm95_skill:+.3f} full={ebm_full_skill:+.3f}")
    return res


def main():
    ap = argparse.ArgumentParser(description="Nori importance -> 95% feature selection -> glass-box EBM")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--demo", choices=sorted(_data._DEMOS), help="bundled sklearn dataset (default: diabetes)")
    src.add_argument("--npz", help="npz with arrays Xtr,ytr,Xte,yte (+ optional feature_names)")
    src.add_argument("--csv", help="single CSV; requires --target")
    src.add_argument("--parquet", help="single Parquet file; requires --target")
    ap.add_argument("--target", help="target column name (with --csv)")
    ap.add_argument("--task", default="auto", choices=["auto", "regression", "classification"])
    ap.add_argument("--method", default="permutation", choices=["permutation", "shap"])
    ap.add_argument("--nori-model", default="nori-6m")
    ap.add_argument("--reduce-threshold", type=int, default=16,
                    help="only select fewer features when d > this (else keep all; default 16)")
    ap.add_argument("--tag", default=None, help="output basename (default derived from the source)")
    ap.add_argument("--out-dir", default="explainability_out", help="where to write <tag>.json + <tag>.ebm.joblib")
    a = ap.parse_args()
    if (a.csv or a.parquet) and not a.target:
        ap.error("--csv/--parquet requires --target")

    Xtr, ytr, Xte, yte, names = _data.load_from_args(a)
    src = a.npz or a.csv or a.parquet                # None when running a --demo (or the default demo)
    tag = a.tag or a.demo or (os.path.splitext(os.path.basename(src))[0] if src else "diabetes")
    run(Xtr, ytr, Xte, yte, names, task=a.task, method=a.method, nori_model=a.nori_model,
        reduce_threshold=a.reduce_threshold, out_dir=a.out_dir, tag=tag)


if __name__ == "__main__":
    main()
