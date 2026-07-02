"""Global feature importance for a fitted Nori model: mean(|Shapley value|)
over a sample of query rows, plus optional PDP and feature selection.

    python explain.py                                # sklearn diabetes demo
    python explain.py --data my.csv --target price --k 10 --budget 256
    python explain.py --pdp 0 2                      # also draw PDPs for features 0 and 2
    python explain.py --feature-selection 5          # also run (slow) selection

Needs the interpretability extra:  pip install "synthefy-nori[interpretability]"
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from synthefy_nori import NoriRegressor

try:
    from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer
except ImportError as e:  # pragma: no cover - env-dependent
    raise SystemExit(
        'interpretability extra missing — pip install "synthefy-nori[interpretability]"'
    ) from e


def first_order_vector(iv, n_features: int) -> np.ndarray:
    """Length-n first-order attribution vector from a shapiq InteractionValues.

    Recent shapiq versions expose get_n_order_values(1); older ones only the
    coalition dict — support both.
    """
    getter = getattr(iv, "get_n_order_values", None)
    if callable(getter):
        try:
            return np.asarray(getter(1), dtype=float).reshape(-1)[:n_features]
        except Exception:
            pass
    vec = np.zeros(n_features, dtype=float)
    d = getattr(iv, "dict_values", None) or dict(iv)
    for coalition, val in d.items():
        if isinstance(coalition, tuple) and len(coalition) == 1 and 0 <= coalition[0] < n_features:
            vec[coalition[0]] = float(val)
    return vec


def shap_importance(reg, X_context: np.ndarray, X_query: np.ndarray,
                    feature_names: list[str], *, budget: int = 128) -> pd.DataFrame:
    """Per-feature mean(|Shapley|) over the query rows, sorted descending."""
    explainer = get_nori_imputation_explainer(reg, X_context, index="SV", max_order=1)
    acc = np.zeros(len(feature_names), dtype=float)
    for row in X_query:
        iv = explainer.explain(row.reshape(1, -1), budget=budget)
        acc += np.abs(first_order_vector(iv, len(feature_names)))
    out = pd.DataFrame({"feature": feature_names, "mean_abs_shap": acc / max(len(X_query), 1)})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="CSV path (default: sklearn diabetes demo)")
    ap.add_argument("--target", default=None, help="target column name (required with --data)")
    ap.add_argument("--device", default=None, help="torch device (default: auto)")
    ap.add_argument("--k", type=int, default=8, help="query rows to explain (Shapley is per-row)")
    ap.add_argument("--budget", type=int, default=128, help="coalitions per row; see interpretability.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pdp", nargs="+", type=int, default=None, metavar="FEAT",
                    help="feature indices to draw partial-dependence plots for")
    ap.add_argument("--feature-selection", type=int, default=None, metavar="N",
                    help="run sequential selection down to N features (slow)")
    args = ap.parse_args()

    if args.data is None:
        X, y = load_diabetes(return_X_y=True, as_frame=True)
    else:
        if args.target is None:
            raise SystemExit("--target is required with --data")
        df = pd.read_csv(args.data)
        y = df[args.target]
        X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
        X, y = X.loc[y.notna()], y.loc[y.notna()]

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    reg = NoriRegressor(device=args.device)
    reg.fit(X_train.values, y_train.values)

    imp = shap_importance(reg, X_train.values, X_test.values[: args.k],
                          list(X.columns), budget=args.budget)
    print(f"mean(|SHAP|) over {min(args.k, len(X_test))} query rows (budget={args.budget}):")
    print(imp.to_string(index=False))

    if args.pdp:
        from synthefy_nori.interpretability.pdp import partial_dependence_plots
        partial_dependence_plots(reg, X_test.values, features=args.pdp, kind="average")

    if args.feature_selection:
        from synthefy_nori.interpretability.feature_selection import feature_selection
        res = feature_selection(reg, X_train.values, y_train.values,
                                n_features_to_select=args.feature_selection, cv=3,
                                feature_names=list(X.columns))
        print("selected:", res.selected_names)
        print(f"CV R²: all-features {res.baseline_score_mean:.3f} -> "
              f"selected {res.selected_score_mean:.3f}")


if __name__ == "__main__":
    main()
