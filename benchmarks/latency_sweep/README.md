# Nori single-H100 latency sweep

End-to-end inference latency (`fit` + `predict` for one request) of the **local**
`synthefy-nori` build, measured on a single NVIDIA **H100 80GB HBM3** via Modal,
across two grids of `X_train` shapes. `X_test` always has `ceil(0.25 · n_rows)`
rows. Each combination is timed 13× (3 warm-up discarded + 10 measured).

## Layout

```
latency_sweep/
  README.md
  tables/                       CSVs + the code that PRODUCES them
    latency_sweep_h100.csv      final small-grid results
    latency_sweep_large.csv     final large-grid results
    modal_latency_sweep.py      the Modal sweep (run_shard / run_pinned / drive / merge)
    merge_rerun.py              merge a pinned re-run into a sweep CSV (de-throttle)
    _finish_large_rerun.py      concat + merge the large-grid pinned re-runs
    provenance/                 de-throttle audit trail (optional to keep)
  figures/                    PNGs + the code that PRODUCES them
    *.png
    plot_latency.py             per-grid heatmaps + line cuts (reads ../tables)
    plot_cells_combined.py      latency vs cells, both sweeps (loglog|linear|logx)
    plot_linecuts_combined.py   vs-rows & vs-cols line cuts, both sweeps
    combine_overviews.py        stacks the two overview PNGs
```

## The two grids

| Grid | `n_rows` | `n_cols` | combos | latency range |
|---|---|---|---|---|
| small | 100 … 10,000 (step 100) | 5 … 100 (step 5) | 2,000 | 0.41 – 9.06 s |
| large | 10,000 … 100,000 (step 5,000) | 50,150,…,950,1000 | 209 | 4.7 – 46.7 s |

## Data dictionary — what each CSV is

All measurement CSVs share the same columns (below). One row = one (n_rows, n_cols) combo.

| CSV | Rows | What it is |
|---|---|---|
| `tables/latency_sweep_h100.csv` | 2000 | **Final small-grid results** (de-throttled). The deliverable. |
| `tables/latency_sweep_large.csv` | 209 | **Final large-grid results** (de-throttled). The deliverable. |
| `tables/provenance/latency_sweep_h100.orig.csv` | 2000 | Small grid **before** de-throttling (raw 50-way fan-out). |
| `tables/provenance/latency_sweep_large.orig.csv` | 209 | Large grid **before** de-throttling. |
| `tables/provenance/latency_rerun_pinned.csv` | 136 | Small-grid de-throttle **pass 1**: flagged combos re-measured on one pinned GPU. |
| `tables/provenance/latency_rerun2_pinned.csv` | 33 | Small-grid de-throttle **pass 2** (combos exposed after pass 1). |
| `tables/provenance/latency_large_rerun_all.csv` | 32 | Large-grid de-throttle: the 32 re-measured combos (concat of 4 pinned workers). |

(The 4 per-shard `latency_large_rerun_{0..3}.csv` and the transient `_flagged_combos.csv`
were pruned — fully redundant with the concat / one-shot intermediates.)

### CSV columns

| Column | Meaning |
|---|---|
| `n_rows`, `n_cols` | `X_train` shape. |
| `n_test` | `X_test` rows = `ceil(0.25 · n_rows)`. |
| `n_warmup`, `n_measured` | Timed calls discarded / kept (3 / 10). |
| `mean_ms`, `p90_ms`, `p99_ms`, `std_ms`, `min_ms`, `max_ms` | Latency stats over the measured calls (ms). |
| `gpu` | Actual GPU name (audit: `gpu="H100"` is occasionally satisfied with H200/NVL). |
| `error` | Non-empty if the combo failed (e.g. CUDA OOM); stats are NaN. |
| `gpu_temp_c`, `cooldown_s` | *(pinned re-runs only)* GPU temp before the combo, and cooldown idle time. |

## Method

Combos are load-balanced and **sharded across many H100s in parallel** (each combo's
13 runs stay on one H100); a driver merges per-shard CSVs. The job is `deploy`ed and
`spawn`ed, so it runs server-side and survives client disconnect.

**De-throttling.** The parallel fan-out leaves scattered contention artifacts (some
shards land on hot/contended GPUs). Flagged combos are re-measured on pinned, cooled,
single-tenant H100s. This **separates genuine contention** (faster on re-run → fixed)
**from real, reproducible behavior** (re-measures within ±5%). The `.orig` + `rerun`
CSVs in `provenance/` are that audit trail.

## Findings

- **Cross-sweep scaling:** across both grids (cells ~5e2 → ~1e8), latency follows a
  clean power law **latency(s) ≈ 0.0051 · cells^0.50** (R²=0.88) — i.e. ~√(n_rows·n_cols).
- **Real non-monotonicity (large grid):** a reproducible latency regime around
  **20k–40k context rows** at low feature counts. e.g. at 50 cols latency spikes to
  **46.7 s at 35k rows**, then drops to ~14 s at 40k. `35000×50` reproducibly exceeds
  `100000×50` (2×) on a clean 31 °C GPU (CV 0.2%) — a smaller input slower than a
  strictly larger one. Points to a regime transition in the inference/preprocessing
  path (adaptive-SVD / Yeo-Johnson / KDI). Left in the data deliberately; may warrant a look.

## Figures (`figures/`)

| PNG | What |
|---|---|
| `latency_overview.png`, `latency_large_overview.png` | 2×2 per grid: mean + p99 heatmaps, vs-rows, vs-cols. |
| `latency_heatmap_mean.png`, `latency_large_heatmap_mean.png` | Standalone mean-latency heatmap per grid. |
| `latency_vs_cells.png`, `latency_large_vs_cells.png` | Latency vs cell count per grid, with a linear fit. |
| `latency_vs_cells_combined.png` / `_linear.png` | Latency vs cells across BOTH sweeps (log-log / linear), power-law fit. |
| `latency_linecuts_combined.png` | vs-rows & vs-cols line cuts merged across both sweeps. |
| `latency_overview_combined.png` | The two per-grid overviews stacked. |

## Reproduce

```bash
uv build                                                                # refresh the local wheel

# run a sweep (server-side, durable)
uv run --no-sync --with modal modal deploy benchmarks/latency_sweep/tables/modal_latency_sweep.py
uv run --no-sync --with modal python -c "import modal; \
    print(modal.Function.from_name('nori-latency-sweep','drive').spawn('large',50).object_id)"
uv run --no-sync --with modal modal volume get nori-latency-results \
    latency_sweep_large.csv benchmarks/latency_sweep/tables/latency_sweep_large.csv --force

# regenerate plots (reads ../tables, writes here)
uv run --no-sync --with matplotlib python benchmarks/latency_sweep/figures/plot_latency.py
uv run --no-sync --with matplotlib python benchmarks/latency_sweep/figures/plot_latency.py \
    benchmarks/latency_sweep/tables/latency_sweep_large.csv latency_large
uv run --no-sync --with matplotlib python benchmarks/latency_sweep/figures/plot_cells_combined.py
uv run --no-sync --with matplotlib python benchmarks/latency_sweep/figures/plot_linecuts_combined.py
```
