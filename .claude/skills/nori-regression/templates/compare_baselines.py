"""Compare Nori against Ridge and RandomForest under ONE fixed CV protocol.

    python compare_baselines.py                              # sklearn diabetes demo
    python compare_baselines.py --data my.csv --target price --folds 5

The folds are built once and shared by every model — that (plus deciding the
metric up front) is what makes the comparison honest. Nori is a scikit-learn
estimator, so it drops straight into cross_validate. Results within one
fold-std of the best are reported as ties.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate

from synthefy_nori import NoriRegressor


def load_data(data: str | None, target: str | None) -> tuple[pd.DataFrame, pd.Series]:
    if data is None:
        return load_diabetes(return_X_y=True, as_frame=True)
    if target is None:
        raise SystemExit("--target is required with --data")
    df = pd.read_csv(data)
    y = df[target]
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    keep = y.notna()
    return X.loc[keep], y.loc[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="CSV path (default: sklearn diabetes demo)")
    ap.add_argument("--target", default=None, help="target column name (required with --data)")
    ap.add_argument("--device", default=None, help="torch device for Nori (default: auto)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y = load_data(args.data, args.target)
    # NaN features are fine for Nori but not for the sklearn baselines — fill a
    # copy for them so every model sees the same rows.
    X_sk = X.fillna(X.median(numeric_only=True))

    cv = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)  # built once, shared
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}

    models = {
        "nori": (NoriRegressor(device=args.device), X),
        "ridge": (Ridge(alpha=1.0), X_sk),
        "random_forest": (RandomForestRegressor(n_estimators=300, random_state=args.seed), X_sk),
    }

    rows = []
    for name, (model, X_m) in models.items():
        res = cross_validate(model, X_m, y, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "r2_mean": res["test_r2"].mean(), "r2_std": res["test_r2"].std(),
            "mae_mean": -res["test_mae"].mean(), "mae_std": res["test_mae"].std(),
        })
        print(f"{name:14s} R² {rows[-1]['r2_mean']:.3f} ± {rows[-1]['r2_std']:.3f}   "
              f"MAE {rows[-1]['mae_mean']:.3f} ± {rows[-1]['mae_std']:.3f}")

    table = pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    tied = table[table.r2_mean >= best.r2_mean - best.r2_std]
    if len(tied) > 1:
        print(f"\nwithin one fold-std of the best ({best.model}): "
              f"{', '.join(tied.model)} — treat these as tied.")
    else:
        print(f"\nbest: {best.model} (clear of the fold-noise band)")


if __name__ == "__main__":
    main()
