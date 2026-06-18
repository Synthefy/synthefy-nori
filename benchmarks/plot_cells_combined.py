#!/usr/bin/env python
"""Latency vs number of cells (n_rows x n_cols) across BOTH sweeps on one plot.

Combines the small grid (latency_sweep_h100.csv: 100-10k rows x 5-100 cols) and the
large grid (latency_sweep_large.csv: 10k-100k rows x 50-1000 cols). Together they
span ~5e2 .. ~1e8 cells, so latency is strongly sublinear -> log-log axes, where a
power law latency = A * cells^p is a straight line. Writes latency_vs_cells_combined.png.

Usage: uv run --no-sync --with matplotlib python benchmarks/plot_cells_combined.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
# Axis scaling: "loglog" (default), "linear" (both linear), or "logx" (log x, linear y).
SCALE = sys.argv[1] if len(sys.argv) > 1 else "loglog"


def load(name, label):
    df = pd.read_csv(HERE / name)
    df = df[df["error"].fillna("") == ""].copy()
    df["cells"] = df.n_rows * df.n_cols
    df["lat_s"] = df.mean_ms / 1000.0
    df["sweep"] = label
    return df


def main():
    small = load("latency_sweep_h100.csv", "small grid  (100–10k rows × 5–100 cols)")
    large = load("latency_sweep_large.csv", "large grid  (10k–100k rows × 50–1000 cols)")
    both = pd.concat([small, large], ignore_index=True)

    # Power-law fit across ALL points: lat = A * cells^p  (linear in log-log space).
    lc, ll = np.log(both.cells.values), np.log(both.lat_s.values)
    p, logA = np.polyfit(lc, ll, 1)
    A = np.exp(logA)
    pred = A * both.cells.values ** p
    r2 = 1 - ((both.lat_s.values - pred) ** 2).sum() / ((both.lat_s.values - both.lat_s.mean()) ** 2).sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(small.cells, small.lat_s, s=12, alpha=0.45, color="steelblue", label=small.sweep.iloc[0])
    ax.scatter(large.cells, large.lat_s, s=16, alpha=0.7, color="darkorange", label=large.sweep.iloc[0])
    xx = np.geomspace(both.cells.min(), both.cells.max(), 400)
    ax.plot(xx, A * xx ** p, "k--", lw=2,
            label=f"power-law fit: {A:.3g}·cells$^{{{p:.2f}}}$   ($R^2$={r2:.3f})")

    if SCALE in ("loglog", "logx"):
        ax.set_xscale("log")
    if SCALE == "loglog":
        ax.set_yscale("log")
    ax.set_xlabel("number of cells in X_train  (n_rows × n_cols)")
    ax.set_ylabel("mean latency (s)")
    ax.set_title(f"Nori inference latency vs dataset size — both sweeps ({SCALE} axes)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    suffix = "" if SCALE == "loglog" else f"_{SCALE}"
    fig.savefig(HERE / f"latency_vs_cells_combined{suffix}.png", dpi=140)
    plt.close(fig)

    print(f"combined points: {len(both)}  (small={len(small)}, large={len(large)})")
    print(f"cells span: {both.cells.min():,} .. {both.cells.max():,}")
    print(f"latency span: {both.lat_s.min():.2f}s .. {both.lat_s.max():.2f}s")
    print(f"power-law fit: latency = {A:.4g} * cells^{p:.3f}   R2={r2:.3f}")
    print(f"wrote: latency_vs_cells_combined{suffix}.png  ({SCALE} axes)")


if __name__ == "__main__":
    main()
