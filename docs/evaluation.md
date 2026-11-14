# Evaluation

## Reproduce the published benchmark

```bash
pip install "synthefy-nori[eval]"

synthefy-nori-eval --download-benchmarks --openml-reg
```

- `--download-benchmarks` fetches the TabArena and TALENT regression datasets
  from OpenML into `cache/tabarena_reg/` and `cache/talent_reg/` as CSVs
  (skipped on later runs once the files exist). TabArena uses the official
  TabArena curated uploads, pinned by OpenML dataset ID; TALENT is fetched by
  dataset name. Membership is pinned by the lists shipped in
  `synthefy_nori/evaluation/benchmark_lists/`, and train/test splits use a
  fixed seed (70/30, seed 42) — identical everywhere, so the CSVs are
  bit-reproducible.
- `--openml-reg` adds the curated 11-dataset OpenML regression suite (loaded
  directly through the `openml` package).
- With no `--checkpoint`, the published checkpoint is downloaded from the
  Hugging Face Hub and evaluated with the bundled default regression config
  (`default_inference.json`).
- The default protocol targets large GPUs: up to `--max-train-samples` (50000)
  context rows with no memory-based cap, and an inference element budget of 8M
  (`--max-elements-budget`, exported as `SYNTHEFY_MAX_ELEMENTS_BUDGET`). On
  smaller GPUs pass `--gpu-mem-gb <GiB>` to cap context rows by a memory model
  and/or lower `--max-elements-budget`; large-table results will be lower.

The command prints a per-source mean R² summary and writes per-dataset metrics
to `results/eval/all_results.csv`.

## Evaluate your own checkpoint

```bash
synthefy-nori-eval --checkpoint "MyRun:checkpoints/best_reg_r2.pt"
```

`--checkpoint` is repeatable (`label:path`), so several checkpoints can be
compared in one run.

## Predictive-distribution metrics

By default only point predictions are scored (R², RMSE, MAE). Add
`--distribution-metrics` to also score the predicted quantiles of a
quantile-head checkpoint:

```bash
synthefy-nori-eval --checkpoint "MyRun:checkpoints/best_reg_r2.pt" --distribution-metrics
```

| Column | Meaning | Ideal |
| --- | --- | --- |
| `crps` | Continuous ranked probability score from the quantile bank | lower |
| `crps_rel` | `crps` divided by the test target's standard deviation, comparable across datasets | lower |
| `cov50` … `cov95` | Fraction of test targets inside the central 50/80/90/95% interval | the level |
| `width90_rel` | Mean 90% interval width divided by the target's standard deviation | narrow at the right coverage |
| `pit_ks` | KS distance of the PIT values from uniform (overall calibration) | 0 |

One distribution call per dataset supplies both the quantiles and, under the
default `quantile_collapse="mean"`, the point prediction. Checkpoints without a
quantile head are scored on point metrics only. Results are cached separately
from point-only runs, and the summary adds mean `cov90`, `crps_rel` and `pit_ks`
per source.

## Dataset locations

The CLI loads local CSV caches by default:

```text
cache/tabarena_reg/    # --tabarena-reg-dir
cache/talent_reg/      # --talent-reg-dir
```

Each dataset is a folder `<name>/` containing `<name>_train.csv` and
`<name>_test.csv` with the target in a `target` column (TALENT-style layout).
Use `--custom-reg-dir` for local custom datasets.
