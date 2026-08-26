"""End-to-end tabular regression with Nori: point predictions, prediction
intervals with an empirical coverage check, and a results.json summary.

    python regress.py                                  # sklearn diabetes demo
    python regress.py --data my.csv --target price     # your CSV
    python regress.py --device cuda:0 --out results.json

The script holds out a test split, fits Nori on the rest (fit = storing
context; all compute happens in predict), reports R²/MAE for both the mean and
median point estimates, and checks that the nominal 80% interval [q10, q90]
actually covers ~80% of the held-out targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from synthefy_nori import NoriRegressor


def load_data(data: str | None, target: str | None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from a CSV, or the sklearn diabetes demo when no CSV given."""
    if data is None:
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        return X, y
    if target is None:
        raise SystemExit("--target is required with --data")
    df = pd.read_csv(data)
    if target not in df.columns:
        raise SystemExit(f"target column {target!r} not in {list(df.columns)}")
    y = df[target]
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    keep = y.notna()  # the target must be finite; NaN features are fine as-is
    return X.loc[keep], y.loc[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="CSV path (default: sklearn diabetes demo)")
    ap.add_argument("--target", default=None, help="target column name (required with --data)")
    ap.add_argument("--device", default=None, help="torch device, e.g. cuda:0 (default: auto)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.json", help="where to write the JSON summary")
    args = ap.parse_args()

    X, y = load_data(args.data, args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    print(f"rows: {len(X_train)} context / {len(X_test)} query, {X.shape[1]} features")

    reg = NoriRegressor(device=args.device, model="nori-30m")
    reg.fit(X_train, y_train)

    mean_pred = reg.predict(X_test)  # (n,) distribution mean
    median_pred = reg.predict(X_test, output_type="median")  # robust for skewed targets
    q10, q50, q90 = reg.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])

    y_true = np.asarray(y_test, dtype=float)
    coverage = float(((y_true >= q10) & (y_true <= q90)).mean())
    results = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "r2_mean": float(r2_score(y_true, mean_pred)),
        "r2_median": float(r2_score(y_true, median_pred)),
        "mae_mean": float(mean_absolute_error(y_true, mean_pred)),
        "mae_median": float(mean_absolute_error(y_true, median_pred)),
        "coverage_10_90": coverage,  # nominal 0.80
        "interval_width_mean": float(np.mean(q90 - q10)),
    }

    print(f"R²   mean={results['r2_mean']:.3f}  median={results['r2_median']:.3f}")
    print(f"MAE  mean={results['mae_mean']:.3f}  median={results['mae_median']:.3f}")
    print(
        f"[q10, q90] empirical coverage: {coverage:.2f} (nominal 0.80), mean width {results['interval_width_mean']:.3f}"
    )

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
