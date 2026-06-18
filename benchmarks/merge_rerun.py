#!/usr/bin/env python
"""Merge the pinned re-run into the main sweep CSV, replacing the throttled combos.

Reads the original sweep (``latency_sweep_h100.csv``) and the clean pinned re-run
(``latency_rerun_pinned.csv``), overwrites the measurement columns for every
(n_rows, n_cols) present in the re-run, backs up the original, writes the
corrected CSV in place, and reports how many local-median spikes remain.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
# argv[1] = re-run CSV (defaults to the first pass); argv[2] = main CSV to update.
RERUN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "latency_rerun_pinned.csv"
MAIN = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "latency_sweep_h100.csv"
MEASURE_COLS = ["mean_ms", "p90_ms", "p99_ms", "std_ms", "min_ms", "max_ms", "gpu", "error"]


def spike_count(df, thr=12.0):
    rows = sorted(df.n_rows.unique()); cols = sorted(df.n_cols.unique())
    ri = {r: i for i, r in enumerate(rows)}; ci = {c: i for i, c in enumerate(cols)}
    M = np.full((len(rows), len(cols)), np.nan)
    for _, x in df.iterrows():
        M[ri[x.n_rows], ci[x.n_cols]] = x.mean_ms
    base = median_filter(M, size=3, mode="nearest")
    infl = np.array([100 * (M[ri[r], ci[c]] / base[ri[r], ci[c]] - 1)
                     for r, c in zip(df.n_rows, df.n_cols)])
    return int((infl > thr).sum()), infl


def main():
    if not RERUN.exists():
        sys.exit(f"re-run CSV not found: {RERUN}")
    main_df = pd.read_csv(MAIN).set_index(["n_rows", "n_cols"])
    rerun = pd.read_csv(RERUN).set_index(["n_rows", "n_cols"])

    before_spikes, _ = spike_count(main_df.reset_index())
    replaced = 0
    for key, row in rerun.iterrows():
        if key in main_df.index and not pd.isna(row.get("mean_ms")):
            for col in MEASURE_COLS:
                if col in row:
                    main_df.loc[key, col] = row[col]
            replaced += 1

    out = main_df.reset_index().sort_values(["n_rows", "n_cols"]).reset_index(drop=True)
    after_spikes, infl = spike_count(out)

    backup = MAIN.with_name(MAIN.stem + ".orig.csv")  # preserve the very first original
    if not backup.exists():
        shutil.copy(MAIN, backup)
    out.to_csv(MAIN, index=False)

    print(f"replaced {replaced} combos from the pinned re-run")
    print(f"local-median spikes (>12%):  before={before_spikes}  ->  after={after_spikes}")
    rem = out.assign(infl=infl)
    rem = rem[rem.infl > 12].sort_values("infl", ascending=False)
    if len(rem):
        print(f"remaining {len(rem)} spike(s) (top 8):")
        print(rem.head(8)[["n_rows", "n_cols", "mean_ms", "infl"]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    else:
        print("no spikes remain -- surface is smooth")
    print(f"original backed up -> {backup.name}; corrected CSV written -> {MAIN.name}")


if __name__ == "__main__":
    main()
