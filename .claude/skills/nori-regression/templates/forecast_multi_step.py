"""Direct multi-step time-series forecasting with Nori: predict h periods
ahead by shifting every feature back h steps — the leak-safe way to do
"2 months ahead" — with a rolling backtest, quantile bands, and naive
baselines.

    python forecast_multi_step.py --horizon 2                     # synthetic demo
    python forecast_multi_step.py --data my.csv --target sales --date-col date --horizon 3

Direct vs recursive: recursive feeds predictions back as lags (errors
compound, bands stop being honest). Direct reframes the table per horizon —
features for the row at period t may use only values from periods <= t-h,
because at forecast time the h-1 most recent periods are not observed yet.
That is the whole trick: lag_h is the newest usable lag; lag_1..lag_{h-1}
would be leaks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthefy_nori import NoriRegressor


def synthetic_series(n: int = 120, season: int = 12, seed: int = 7,
                     ar: float = 0.7, sigma: float = 3.0) -> pd.DataFrame:
    """Trend + seasonality + AR(1) noise. The autocorrelation matters: it makes
    horizon-2 genuinely harder than horizon-1, like real demand/telemetry."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = ar * eps[i - 1] + rng.normal(0, sigma)
    y = 50 + 0.3 * t + 12 * np.sin(2 * np.pi * t / season) + eps
    dates = pd.date_range("2015-01-31", periods=n, freq="ME")
    return pd.DataFrame({"date": dates, "y": y})


def build_features(y: pd.Series, *, horizon: int, season: int,
                   n_lags: int = 3) -> pd.DataFrame:
    """Leak-safe for horizon h: every column uses only values <= t-h."""
    df = pd.DataFrame(index=y.index)
    for k in range(horizon, horizon + n_lags):          # lag_h is the newest legal lag
        df[f"lag{k}"] = y.shift(k)
    df[f"lag{max(season, horizon)}"] = y.shift(max(season, horizon))
    trailing = y.shift(horizon).rolling(season)
    df["roll_mean"] = trailing.mean()
    df["roll_std"] = trailing.std()
    t = np.arange(len(y))
    df["trend"] = t                                      # calendar of the TARGET period is known
    df["phase_sin"] = np.sin(2 * np.pi * (t % season) / season)
    df["phase_cos"] = np.cos(2 * np.pi * (t % season) / season)
    return df


def rolling_direct(y: np.ndarray, X: np.ndarray, origins: range, *, horizon: int,
                   device: str | None) -> pd.DataFrame:
    """At each origin t: context = rows whose features AND target are fully in
    the past (target rows <= t-h), then predict row t."""
    reg = NoriRegressor(device=device, model="nori-30m")
    rows = []
    for t in origins:
        cut = t - horizon + 1  # rows [0, cut) have targets <= t-h+... strictly before t's info
        reg.fit(X[:cut], y[:cut])
        point = float(np.atleast_1d(reg.predict(X[t:t + 1], output_type="median"))[0])
        q = np.asarray(reg.predict(X[t:t + 1], output_type="quantiles",
                                   quantiles=[0.1, 0.9])).reshape(2, -1)
        rows.append({"t": t, "y_true": float(y[t]), "y_pred": point,
                     "q10": float(q[0, 0]), "q90": float(q[1, 0])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="CSV path (default: synthetic AR demo)")
    ap.add_argument("--target", default="y")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--horizon", type=int, default=2, help="steps ahead (h=1 -> one-step)")
    ap.add_argument("--season", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--device", default=None)
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

    X = build_features(y, horizon=args.horizon, season=args.season).to_numpy(np.float32)
    yv = y.to_numpy(np.float64)
    origins = range(len(yv) - args.n_test, len(yv))
    print(f"{len(yv)} periods, horizon={args.horizon}, last {args.n_test} origins "
          f"({X.shape[1]} features; newest legal lag = lag{args.horizon})")

    fc = rolling_direct(yv, X, origins, horizon=args.horizon, device=args.device)
    fc["date"] = df[date_col].iloc[fc.t].dt.date.values

    truth, pred = fc.y_true.to_numpy(), fc.y_pred.to_numpy()
    idx = np.asarray(origins)
    results = {
        "horizon": int(args.horizon),
        "n_origins": int(args.n_test),
        "mae": float(np.abs(truth - pred).mean()),
        "mae_naive_seasonal": float(np.abs(yv[idx] - yv[idx - args.season]).mean()),
        "mae_naive_lastknown": float(np.abs(yv[idx] - yv[idx - args.horizon]).mean()),
        "coverage_10_90": float(((truth >= fc.q10) & (truth <= fc.q90)).mean()),
        "interval_width_mean": float((fc.q90 - fc.q10).mean()),
    }
    print(f"MAE  nori={results['mae']:.3f}  last-known={results['mae_naive_lastknown']:.3f}  "
          f"seasonal-naive={results['mae_naive_seasonal']:.3f}")
    print(f"coverage [q10,q90] {results['coverage_10_90']:.2f} (nominal 0.80)")

    fc[["date", "y_true", "y_pred", "q10", "q90"]].to_csv("forecasts.csv", index=False)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote forecasts.csv and {args.out}")


if __name__ == "__main__":
    main()
