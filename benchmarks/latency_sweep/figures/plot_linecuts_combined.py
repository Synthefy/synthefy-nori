#!/usr/bin/env python
"""Two combined line plots across BOTH sweeps: latency vs n_rows and vs n_cols.

Overlays the small grid (latency_sweep_h100.csv) and large grid
(latency_sweep_large.csv) on shared, log-scaled axes. Lines are keyed by the held
value; where a value exists in both grids (cols=50, rows=10000) the line spans the
full range continuously, so the two sweeps visibly join.

Usage: uv run --no-sync --with matplotlib python benchmarks/plot_linecuts_combined.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def load(name):
    df = pd.read_csv(HERE.parent / "tables" / name)
    return df[df["error"].fillna("") == ""].copy()


def main():
    both = pd.concat([load("latency_sweep_h100.csv"), load("latency_sweep_large.csv")], ignore_index=True)
    both["lat_s"] = both.mean_ms / 1000.0
    gpu = both.gpu.mode().iloc[0] if "gpu" in both else "H100"

    fig, (axr, axc) = plt.subplots(1, 2, figsize=(16, 6.5))

    # --- latency vs n_rows (one line per selected col; cols=50 bridges both grids) ---
    for c in [5, 50, 100, 250, 550, 1000]:
        sub = both[both.n_cols == c].drop_duplicates("n_rows").sort_values("n_rows")
        if len(sub):
            axr.plot(sub.n_rows, sub.lat_s, marker=".", ms=4, lw=1, label=f"{c} cols")
    axr.axvline(10_000, color="gray", ls=":", lw=1, alpha=0.7)
    axr.text(10_000, axr.get_ylim()[1], " grid boundary", color="gray", fontsize=8, va="top")
    axr.set_xscale("log")
    axr.set(xlabel="n_rows (context rows)", ylabel="mean latency (s)",
            title="Latency vs context rows — both sweeps")
    axr.grid(alpha=0.3, which="both"); axr.legend(title="features", fontsize=8)

    # --- latency vs n_cols (one line per selected row; rows=10000 bridges both grids) ---
    for r in [1_000, 10_000, 50_000, 100_000]:
        sub = both[both.n_rows == r].drop_duplicates("n_cols").sort_values("n_cols")
        if len(sub):
            axc.plot(sub.n_cols, sub.lat_s, marker=".", ms=4, lw=1, label=f"{r} rows")
    axc.set_xscale("log")
    axc.set(xlabel="n_cols (features)", ylabel="mean latency (s)",
            title="Latency vs features — both sweeps")
    axc.grid(alpha=0.3, which="both"); axc.legend(title="context rows", fontsize=8)

    fig.suptitle(f"Nori inference latency line cuts across both sweeps ({gpu})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(HERE / "latency_linecuts_combined.png", dpi=140)
    plt.close(fig)
    print("wrote latency_linecuts_combined.png")


if __name__ == "__main__":
    main()
