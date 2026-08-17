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

## Dataset locations

The CLI loads local CSV caches by default:

```text
cache/tabarena_reg/    # --tabarena-reg-dir
cache/talent_reg/      # --talent-reg-dir
```

Each dataset is a folder `<name>/` containing `<name>_train.csv` and
`<name>_test.csv` with the target in a `target` column (TALENT-style layout).
Use `--custom-reg-dir` for local custom datasets.
