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
  (`reg_allordinal_poly10_adaptive_svd256.json`).
- The default protocol targets large GPUs: up to `--max-train-samples` (50000)
  context rows with no memory-based cap, and an inference element budget of 8M
  (`--max-elements-budget`, exported as `SYNTHEFY_MAX_ELEMENTS_BUDGET`). On
  smaller GPUs pass `--gpu-mem-gb <GiB>` to cap context rows by a memory model
  and/or lower `--max-elements-budget`; large-table results will be lower.

The command prints a per-source mean R² summary and writes per-dataset metrics
to `results/eval/all_results.csv`.

## RelBench (relational tasks)

Nori can be evaluated on the [RelBench](https://relbench.stanford.edu)
leaderboard tasks via the **entity-table tabular protocol**: each relational
predictive task is flattened into one feature table (the task rows merged with
their entity table) that Nori fits in-context — the same regime tabular
foundation models (e.g. TabPFN) are listed under on the RelBench leaderboard.

```bash
pip install "synthefy-nori[relbench]"

synthefy-nori-eval --relbench                       # full pinned suite
synthefy-nori-eval --relbench --relbench-tasks rel-f1/driver-position rel-f1/driver-dnf
```

- Covers the **entity tasks** (binary classification → AUROC, regression → MAE)
  across the seven canonical RelBench datasets; recommendation / link-prediction
  tasks are skipped (not a tabular-model fit). The task list is pinned in
  `synthefy_nori/evaluation/benchmark_lists/relbench_entity.csv`.
- `--relbench-mode entity` (default) merges with the entity table only;
  `--relbench-mode temporal` additionally adds per-entity temporal aggregations.
- Scoring uses RelBench's own `task.evaluate`, so metrics match the leaderboard:
  validation is scored locally; the held-out test split is scored against
  RelBench's hidden labels.
- Results are written to `--relbench-out` (default `results/relbench/`):
  `classification.csv`, `regression.csv`, and `SUBMISSION.md` (results split by
  task type, plus the current submission status).
- Classification reuses the trained classification head already shipped in the
  checkpoint (re-exposed via `NoriClassifier`).

**Submission status:** RelBench currently has **no self-service submission
endpoint** — the leaderboard is a maintainer-generated static page under
redesign. `SUBMISSION.md` documents the contribution path (open an issue/PR on
`snap-stanford/relbench`) and is formatted to match the leaderboard tables.

## Evaluate your own checkpoint

```bash
synthefy-nori-eval --checkpoint "MyRun:checkpoints/best_reg_r2.pt"
```

`--checkpoint` is repeatable (`label:path`), so several checkpoints can be
compared in one run.

## Dataset locations

The CLI loads local CSV caches by default:

```text
cache/tabarena_reg/    # --tabarena-reg-dir
cache/talent_reg/      # --talent-reg-dir
```

Each dataset is a folder `<name>/` containing `<name>_train.csv` and
`<name>_test.csv` with the target in a `target` column (TALENT-style layout).
Use `--custom-reg-dir` for local custom datasets.
