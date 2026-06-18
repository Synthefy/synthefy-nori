#!/usr/bin/env python
"""Concat the 4 pinned re-run CSVs, merge into latency_sweep_large.csv, re-verify.

Verification uses a DOMINANCE check (a shape slower than a strictly-larger shape is
an artifact) -- the right test for this coarse grid, where a 3x3 spatial median is
noisy. Reports max dominance excess before vs after.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE / "latency_sweep_large.csv"
MEASURE = ["mean_ms", "p90_ms", "p99_ms", "std_ms", "min_ms", "max_ms", "gpu", "error"]


def max_dom_excess(df):
    r, c, m = df.n_rows.values, df.n_cols.values, df.mean_ms.values
    out = np.zeros(len(df))
    for i in range(len(df)):
        dom = (r >= r[i]) & (c >= c[i]) & ~((r == r[i]) & (c == c[i]))
        if dom.any():
            out[i] = max(0.0, (m[i] - m[dom].min()) / m[dom].min() * 100)
    return out


def main():
    parts = []
    for gi in range(4):
        p = HERE / f"latency_large_rerun_{gi}.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    rerun = pd.concat(parts, ignore_index=True)
    rerun.to_csv(HERE / "latency_large_rerun_all.csv", index=False)

    main_df = pd.read_csv(MAIN)
    before = max_dom_excess(main_df)
    backup = MAIN.with_name(MAIN.stem + ".orig.csv")
    if not backup.exists():
        shutil.copy(MAIN, backup)

    mi = main_df.set_index(["n_rows", "n_cols"])
    rr = rerun.set_index(["n_rows", "n_cols"])
    replaced = 0
    for key, row in rr.iterrows():
        if key in mi.index and not pd.isna(row.get("mean_ms")):
            for col in MEASURE:
                if col in row:
                    mi.loc[key, col] = row[col]
            replaced += 1
    out = mi.reset_index().sort_values(["n_rows", "n_cols"]).reset_index(drop=True)
    after = max_dom_excess(out)
    out.to_csv(MAIN, index=False)

    print(f"replaced {replaced} combos from {len(parts)} pinned re-run files")
    print(f"GPUs now: {out.gpu.value_counts().to_dict()}")
    print(f"dominance violations >8%:  before={int((before>8).sum())}  ->  after={int((after>8).sum())}")
    print(f"max dominance excess:      before={before.max():.1f}%  ->  after={after.max():.1f}%")
    rem = out.assign(dom=after)
    rem = rem[rem.dom > 8].sort_values("dom", ascending=False)
    if len(rem):
        print(f"remaining >8% ({len(rem)}):")
        print(rem.head(10)[["n_rows", "n_cols", "mean_ms", "dom"]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"latency range now: {out.mean_ms.min()/1000:.2f}s .. {out.mean_ms.max()/1000:.2f}s")


if __name__ == "__main__":
    main()
