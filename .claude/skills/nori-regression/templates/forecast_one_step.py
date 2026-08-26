"""One-step-ahead time-series forecasting with Nori: leak-safe lag features,
rolling-origin backtest with expanding context, quantile bands, and naive
baselines (last-value, seasonal-naive) for an honest comparison.

    python forecast_one_step.py                                   # synthetic monthly demo
    python forecast_one_step.py --data my.csv --target sales --date-col date
    python forecast_one_step.py --season 7 --n-test 28            # daily data

Writes forecasts.csv (one row per origin) and results.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthefy_nori import NoriRegressor


def synthetic_series(n: int = 120, season: int = 12, seed: int = 42) -> pd.DataFrame:
    """Trend + seasonality + noise; the demo stand-in for your CSV."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 50 + 0.3 * t + 12 * np.sin(2 * np.pi * t / season) + rng.normal(0, 3, n)
    dates = pd.date_range("2015-01-31", periods=n, freq="ME")
    return pd.DataFrame({"date": dates, "y": y})


def build_features(y: pd.Series, *, season: int, lags: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Leak-safe feature frame: every column uses only data strictly before its row."""
    df = pd.DataFrame(index=y.index)
    for k in (*lags, season):
        df[f"lag{k}"] = y.shift(k)
    trailing = y.shift(1).rolling(season)
    df["roll_mean"] = trailing.mean()
    df["roll_std"] = trailing.std()
    t = np.arange(len(y))
    df["trend"] = t
    df["phase_sin"] = np.sin(2 * np.pi * (t % season) / season)
    df["phase_cos"] = np.cos(2 * np.pi * (t % season) / season)
    return df  # early rows hold NaN lags — fine as Nori context; don't zero-fill


def rolling_one_step(y: np.ndarray, X: np.ndarray, origins: range, *, device: str | None) -> pd.DataFrame:
    """Expanding-context backtest: refit (free) at each origin, predict one row."""
    reg = NoriRegressor(device=device, model="nori-30m")  # construct once, refit per origin
    rows = []
    for t in origins:
        reg.fit(X[:t], y[:t])
        # single-row predict can come back 0-d — normalize the shapes
        point = float(np.atleast_1d(reg.predict(X[t : t + 1], output_type="median"))[0])
        q = np.asarray(reg.predict(X[t : t + 1], output_type="quantiles", quantiles=[0.1, 0.9])).reshape(2, -1)
        rows.append({"t": t, "y_true": float(y[t]), "y_pred": point, "q10": float(q[0, 0]), "q90": float(q[1, 0])})
    return pd.DataFrame(rows)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="CSV path (default: synthetic monthly demo)")
    ap.add_argument("--target", default="y", help="value column in the CSV")
    ap.add_argument("--date-col", default="date", help="date column in the CSV (sorted on)")
    ap.add_argument("--season", type=int, default=12, help="seasonal period (12 monthly, 7 daily)")
    ap.add_argument("--lags", default="1,2,3", help="comma-separated short lags")
    ap.add_argument("--n-test", type=int, default=24, help="rolling origins held out at the end")
    ap.add_argument("--device", default=None, help="torch device (default: auto)")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if args.data is None:
        df = synthetic_series(season=args.season)
        target, date_col = "y", "date"
    else:
        df = pd.read_csv(args.data)
        target, date_col = args.target, args.date_col
        df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    y = df[target].astype(float)
    if y.isna().any():
        raise SystemExit("target has NaN — fill or drop those periods first")

    lags = tuple(int(k) for k in args.lags.split(","))
    X = build_features(y, season=args.season, lags=lags).to_numpy(np.float32)
    yv = y.to_numpy(np.float64)
    origins = range(len(yv) - args.n_test, len(yv))
    print(f"{len(yv)} periods, forecasting the last {args.n_test} one step ahead (features: {X.shape[1]})")

    fc = rolling_one_step(yv, X, origins, device=args.device)
    fc["date"] = df[date_col].iloc[fc.t].dt.date.values

    truth, pred = fc.y_true.to_numpy(), fc.y_pred.to_numpy()
    naive_last = yv[np.asarray(origins) - 1]
    naive_seasonal = yv[np.asarray(origins) - args.season]
    results = {
        "n_origins": int(args.n_test),
        "mae": float(np.abs(truth - pred).mean()),
        "wape": wape(truth, pred),
        "mae_naive_last": float(np.abs(truth - naive_last).mean()),
        "mae_naive_seasonal": float(np.abs(truth - naive_seasonal).mean()),
        "coverage_10_90": float(((truth >= fc.q10) & (truth <= fc.q90)).mean()),
        "interval_width_mean": float((fc.q90 - fc.q10).mean()),
    }

    print(
        f"MAE  nori={results['mae']:.3f}  last-value={results['mae_naive_last']:.3f}  "
        f"seasonal-naive={results['mae_naive_seasonal']:.3f}"
    )
    print(f"WAPE nori={results['wape']:.3%}   [q10,q90] coverage {results['coverage_10_90']:.2f} (nominal 0.80)")

    fc[["date", "y_true", "y_pred", "q10", "q90"]].to_csv("forecasts.csv", index=False)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote forecasts.csv and {args.out}")


if __name__ == "__main__":
    main()
