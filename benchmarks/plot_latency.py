#!/usr/bin/env python
"""Plot how Nori inference latency varies across the (n_rows x n_cols) grid.

Reads the sweep CSV and writes PNGs to benchmarks/:
  * latency_heatmap_mean.png  -- mean latency over all 2000 combos (the headline view)
  * latency_overview.png      -- 2x2: mean heatmap, p99 heatmap, vs-rows, vs-cols

Usage:
  uv run --no-sync --with matplotlib python benchmarks/plot_latency.py [csv_path]
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
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "latency_sweep_h100.csv"


def _grid(df, value):
    """Pivot a column into a (rows x cols) matrix plus the axis tick values."""
    p = df.pivot(index="n_rows", columns="n_cols", values=value).sort_index()
    return p.values, p.index.values, p.columns.values


def heatmap(ax, Z, rows, cols, title, label):
    im = ax.imshow(Z / 1000.0, origin="lower", aspect="auto", cmap="viridis",
                   extent=[cols.min(), cols.max(), rows.min(), rows.max()])
    ax.set_xlabel("n_cols (features)")
    ax.set_ylabel("n_rows (context rows)")
    ax.set_title(title)
    cb = ax.figure.colorbar(im, ax=ax)
    cb.set_label(label)


def cells_plot(df, gpu):
    """Scatter of latency vs number of cells in X_train (n_rows*n_cols).

    Colored by n_cols: if latency were a pure function of cell count the points
    would lie on one curve; the color-separated spread shows features cost more
    per cell than rows do.
    """
    cells = (df.n_rows * df.n_cols).values.astype(float)
    y = df.mean_ms.values / 1000.0

    # Line of best fit: latency(s) = a + b*cells (linear in cells fits best here,
    # R2~0.96 vs ~0.86 for a power law -- there's a fixed overhead + per-cell cost).
    # It is a STRAIGHT line in cells-vs-latency space; on a log x-axis it looks
    # curved purely because the axis is stretched. Show both to make that clear.
    b, a = np.polyfit(cells, y, 1)
    r2 = 1 - ((y - (a + b * cells)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    xx = np.linspace(cells.min(), cells.max(), 400)
    label = f"best fit: {a:.2f} + {b*1e6:.2f}·(cells/10⁶) s   ($R^2$={r2:.3f})"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, logx in zip(axes, (True, False)):
        sc = ax.scatter(cells, y, c=df.n_cols, cmap="plasma", s=10, alpha=0.7)
        ax.plot(xx, a + b * xx, "k--", lw=2, label=label)
        if logx:
            ax.set_xscale("log")
            ax.set_title("log x-axis (fit looks curved — axis is stretched)")
        else:
            ax.set_title("linear x-axis (same fit — now visibly a straight line)")
        ax.set_xlabel("number of cells in X_train  (n_rows × n_cols)")
        ax.set_ylabel("mean latency (s)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=9)
    fig.colorbar(sc, ax=axes, label="n_cols (features)")
    fig.suptitle(f"Nori latency vs dataset size (cells) on {gpu}", fontsize=13)
    fig.savefig(HERE / "latency_vs_cells.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(CSV)
    df = df[df["error"].fillna("") == ""].copy()
    Zmean, rows, cols = _grid(df, "mean_ms")
    Zp99, _, _ = _grid(df, "p99_ms")
    gpu = df["gpu"].mode().iloc[0] if "gpu" in df else "GPU"

    # 1) Headline heatmap -- every combination at a glance
    fig, ax = plt.subplots(figsize=(9, 7))
    heatmap(ax, Zmean, rows, cols, f"Nori mean inference latency on {gpu}\n"
            f"(fit+predict, 10 measured runs/combo; X_test = ceil(0.25*n_rows))",
            "mean latency (s)")
    fig.tight_layout()
    fig.savefig(HERE / "latency_heatmap_mean.png", dpi=140)
    plt.close(fig)

    # 2) Overview: mean + p99 heatmaps, and line cuts along each axis
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    heatmap(axes[0, 0], Zmean, rows, cols, "Mean latency", "mean latency (s)")
    heatmap(axes[0, 1], Zp99, rows, cols, "p99 latency", "p99 latency (s)")

    # latency vs rows, one line per selected col
    axr = axes[1, 0]
    for c in [5, 25, 50, 75, 100]:
        sub = df[df.n_cols == c].sort_values("n_rows")
        axr.plot(sub.n_rows, sub.mean_ms / 1000.0, marker=".", ms=3, lw=1, label=f"{c} cols")
    axr.set(xlabel="n_rows (context rows)", ylabel="mean latency (s)",
            title="Latency vs context rows")
    axr.grid(alpha=0.3); axr.legend(title="features", fontsize=8)

    # latency vs cols, one line per selected row
    axc = axes[1, 1]
    for r in [500, 2000, 5000, 8000, 10000]:
        sub = df[df.n_rows == r].sort_values("n_cols")
        axc.plot(sub.n_cols, sub.mean_ms / 1000.0, marker=".", ms=4, lw=1, label=f"{r} rows")
    axc.set(xlabel="n_cols (features)", ylabel="mean latency (s)",
            title="Latency vs features")
    axc.grid(alpha=0.3); axc.legend(title="context rows", fontsize=8)

    fig.suptitle(f"Nori latency sweep -- {len(df)} combos on {gpu}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(HERE / "latency_overview.png", dpi=140)
    plt.close(fig)

    # 3) latency vs number of cells in X_train
    cells_plot(df, gpu)

    print(f"latency range: {df.mean_ms.min()/1000:.2f}s .. {df.mean_ms.max()/1000:.2f}s "
          f"(median {df.mean_ms.median()/1000:.2f}s) over {len(df)} combos")
    print("wrote: latency_heatmap_mean.png, latency_overview.png, latency_vs_cells.png")


if __name__ == "__main__":
    main()
