# Nori single-H100 latency sweep

End-to-end inference latency (`fit` + `predict` for one request) of the **local**
`synthefy-nori` build across a grid of context/query shapes, measured on a single
NVIDIA **H100 80GB HBM3** via Modal.

## Experiment

- `X_train` rows: **100 … 10000**, step 100 (100 values)
- `X_train` cols: **5 … 100**, step 5 (20 values)
- `X_test` rows: **ceil(n_rows × 0.25)**
- Per combination (2000 total): **3 warm-up runs (discarded) + 10 measured** = 13 calls.
  Reported per combo: `mean / p90 / p99 / std` (+ `min`/`max`, `n_test`, `gpu`).
- One warmed predictor per worker (checkpoint loaded once, **not** timed).

## How it was run

`modal_latency_sweep.py` ships the locally built wheel (`dist/synthefy_nori-*.whl`)
into the image, so the cloud runs exactly the local code. The 2000 combos are
load-balanced and **sharded across 50 H100s in parallel**; each combo's 13 runs stay
on one H100. The job is `deploy`ed and `spawn`ed so it runs fully server-side
(survives client disconnect); each shard commits its own CSV and a driver merges them.

```bash
uv build                                                                 # refresh wheel
uv run --no-sync --with modal modal deploy benchmarks/modal_latency_sweep.py
uv run --no-sync --with modal python -c "import modal; \
    print(modal.Function.from_name('nori-latency-sweep','drive').spawn('full',50).object_id)"
uv run --no-sync --with modal modal volume get nori-latency-results \
    latency_sweep_h100.csv benchmarks/latency_sweep_h100.csv --force     # fetch results
```

## De-throttling (data cleaning)

The initial fan-out left **throttle/contention artifacts scattered across the grid**
(some shards landed on hot/contended GPUs → steady-elevated latencies). These were
detected as local-median spikes (>12% above a 3×3 neighbourhood) and **re-measured on
a single pinned H100 with active thermal cooldowns** (`run_pinned`), then merged back
(`merge_rerun.py`). Two passes converged the spike count **136 → 33 → 9**; the 9
residuals are sub-second low-row combos within run-to-run noise. The big motivating
outliers are fixed (e.g. `10000×90`: 12.8 s → 9.06 s).

## Results

- Latency range **0.41 s → 9.06 s** (median 2.15 s), all 2000 combos on H100, 0 errors.
- Roughly **linear in context rows**; slope steepens with feature count.
- Best fit vs dataset size: **latency(s) ≈ 0.59 + 7.68 × (cells / 10⁶)**, R²=0.964
  (cells = n_rows × n_cols) — a ~0.59 s fixed overhead plus ~7.68 µs/cell. For a fixed
  cell count, row-heavy/feature-light shapes are slightly slower (per-row overhead).

## Files

| File | What |
|---|---|
| `modal_latency_sweep.py` | The sweep: `run_shard` / `run_pinned` / `drive` / `merge` (Modal). |
| `merge_rerun.py` | Merge a pinned re-run into the main CSV + re-check spikes. |
| `plot_latency.py` | Generate the plots below. |
| `latency_sweep_h100.csv` | **Final, de-throttled** results (2000 combos). |
| `latency_sweep_h100.orig.csv` | Original pre-correction results (audit). |
| `latency_rerun_pinned.csv`, `latency_rerun2_pinned.csv` | Pinned re-run data (per-row GPU temp / cooldown). |
| `_flagged_combos.csv` | Combos flagged as throttled in pass 1. |
| `latency_heatmap_mean.png` | Mean latency over the full grid. |
| `latency_overview.png` | Mean + p99 heatmaps and vs-rows / vs-cols cuts. |
| `latency_vs_cells.png` | Latency vs cell count with best-fit line (log + linear axes). |

Regenerate plots: `uv run --no-sync --with matplotlib python benchmarks/plot_latency.py`
