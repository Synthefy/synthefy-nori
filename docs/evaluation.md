# Evaluation

`synthefy-nori-eval` reproduces Nori on three regression benchmarks using each
benchmark's native official train/test protocol:

- **TALENT-100**: the 100 regression datasets from the January 2025 native
  archive. Context is the supplied train+validation arrays and scoring uses the
  supplied test array, capped at 10,000/20,000 rows.
- **OpenML-CTR23**: the 35 suite-353 regression tasks, pinned by OpenML task ID.
  Every split registered on each task is evaluated.
- **TabArena v0.1**: the 13 regression tasks, pinned to TabArena's own OpenML
  task registrations. The official outer repeat policy is 10x3 folds below
  2,500 rows, 3x3 through 250,000 rows, and 1x3 above that threshold.

There are no generated CSV holdouts or legacy 72/11-dataset subsets.

## Install and acquire the data

```bash
pip install "synthefy-nori[eval]"

mkdir -p cache/talent
curl -L --fail \
  -o cache/talent/dataset-latest.zip \
  https://huggingface.co/datasets/LAMDA-Tabular/TALENT/resolve/7bd276bcb7f6b4c0998025855528bd76bd88f13d/dataset-latest.zip
echo "4c8481107153593eb98ef5bb677b00dedfa04cd4c31441babf696a8d100f207c  cache/talent/dataset-latest.zip" \
  | sha256sum --check
unzip cache/talent/dataset-latest.zip -d cache/talent
```

The archive should produce `cache/talent/data/<dataset>/...`. OpenML-CTR23 and
TabArena download through the OpenML client on first use and then reuse
`cache/openml/`. The task IDs are package data under
`synthefy_nori/evaluation/benchmark_lists/`; the evaluator never queries a
mutable live suite membership.

Check the discovered protocol before using GPU time:

```bash
synthefy-nori-eval --dry-run --talent-root cache/talent/data
```

Counts are derived from the loaders and official task metadata. The command
refuses to run if they differ from:

| Suite | Datasets | Evaluation units |
|---|---:|---:|
| TALENT | 100 | 100 |
| OpenML-CTR23 | 35 | 800 |
| TabArena | 13 | 222 |

An evaluation unit is one dataset/fold. OpenML owns the CTR23 fold/repeat
dimensions. TabArena's 222 units follow its size-dependent repeat policy; they
are not the stale `13 * 30 = 390` approximation.

## Reproduce both public models

```bash
synthefy-nori-eval \
  --model nori-6m \
  --model nori-30m \
  --talent-root cache/talent/data \
  --output results/eval/official_results.jsonl
```

The canonical run is intentionally strict:

| Setting | Canonical value |
|---|---|
| Regression config | `reg_allordinal_poly10_adaptive_svd256.json` |
| Config SHA-256 | `134fe355510887086a0d55a419400916c82d713e8b780037d9779c280f0c25f6` |
| Nori-6M HF revision | `7d55529ac3cbf5e3ba1b8129605f391c874a70da` |
| Nori-6M checkpoint SHA-256 | `a13b2bc31d8db24d17bae6d04844e0adf669e446087b0b7a34c7b05045d61323` |
| Nori-30M HF revision | `b9b2a734315225a8188afcfd997ad9d5a4fd9466` |
| Nori-30M checkpoint SHA-256 | `818433f8af12c1137b96d9ff47e109b4eef5818d4e52a9656b2e573dbf13b74d` |
| Imputation | train-column median; all-missing columns become zero |
| Subsampling RNG | per-unit SHA-256-derived seed, base seed `0` |
| Inference element budget | `8,000,000` |
| Silent context subsampling | forbidden (`allow_subsample=False`) |

The CLI verifies the bundled config and public checkpoint hashes before
inference. The config, imputation, seed, element budget, and subsampling policy
cannot be changed through the official command. `--limit` and `--fold-stride`
are available for smoke tests, but their partial outputs are not complete
protocol results.

SVD failure is a failed row, never a degraded score. If the full official
context cannot fit under the 8M policy, the row fails instead of silently
dropping examples. Use GPUs with enough memory for the canonical command.

## Results and aggregation

The JSONL contains one row per `(suite, dataset, fold, model)` and flushes after
every unit so an interrupted run can resume. Each row records the exact model
and config hashes, model selector, package version, shipped source-tree and
benchmark-input fingerprints, fold identity, OpenML task ID, caps, seed,
imputation policy, element budget, and `allow_subsample` value. Publishable
metadata contains hashes and selectors, not local checkpoint or config paths.
Changing model, data, or code identity does not reuse stale rows.

The suite statistic is computed in two steps:

1. mean R² across a dataset's official folds;
2. mean and median of those dataset-level R² values, weighting every dataset
   equally.

The CLI prints that aggregation only after every selected unit finishes without
an execution error. The JSONL remains the source artifact for auditing or
alternative aggregation.

## Evaluate a local checkpoint

```bash
synthefy-nori-eval \
  --checkpoint "my-run:checkpoints/best_reg_r2.pt" \
  --suite openml-ctr23 \
  --output results/eval/my-run.jsonl
```

`--checkpoint` and `--model` are repeatable, so public and local checkpoints can
be compared on exactly the same units. Use `--fresh` to overwrite an existing
JSONL instead of resuming it.
