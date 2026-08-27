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

## Named suites for training checkpoint validation

The public `synthefy-nori-eval` command above remains a supported interface.
Training recipes additionally use an eval-owned named-suite layer built on the
newer `BenchmarkLoader`/`run_benchmark` framework:

```bash
scripts/train suites
scripts/train validate checkpoints/production/<run> \
  --checkpoint checkpoints/production/<run>/checkpoint_step_1000.pt
```

A production recipe declares only `validation: <suite_name>` plus
`validation_interval_steps`; it may use `null` with no interval to disable
validation. At each interval the trainer saves a checkpoint and invokes this
same command synchronously. The versioned YAML under
`src/synthefy_nori/evaluation/suite_defs/` exclusively owns task IDs, folds,
caps, preprocessing, scoring, and aggregation. There are no training-side
dataset/fold/metric overrides: a different subset is a different reviewed suite
name. This keeps two researchers' validation numbers comparable and makes the
suite definition/digest part of every training manifest.

For an online production run, `validate` joins the original W&B run as a
non-primary shared writer. It logs `validation/primary_score`, the metric alias
(for this recipe, `validation/mean_r2`), and every
`validation/dataset/<source>__<dataset>/r2` value against
`validation/checkpoint_step`. The step is read from `optimizer_step` inside the
checkpoint rather than inferred from its filename. Incomplete suites are never
published as scores. Direction-aware W&B summaries retain the best complete
macro and per-dataset scores even when asynchronous checkpoints finish out of
order. The validator requires the original W&B run to exist; it will not
silently create an eval-only replacement. A successful upload writes
`wandb-log.json` beside the eval results, so rerunning the same validation does
not duplicate the W&B point. If upload fails, the eval files remain resumable
and the command exits nonzero so the upload cannot be silently missed.

The two initial declarations are `tabarena_fold0_v1` and
`openml_ctr23_fold0_v1`. This is not a list of all benchmarks the eval framework
supports. Named-suite selections can use the existing OpenML, TabArena, TALENT,
BeyondArena, ScoringBench, or manifest-backed directory loaders. The latter is
the route for customer/POC data. New subsets are added only when their exact
dataset/fold membership and primary metric have been reviewed; a training recipe
never improvises them.

## Dataset locations

The CLI loads local CSV caches by default:

```text
cache/tabarena_reg/    # --tabarena-reg-dir
cache/talent_reg/      # --talent-reg-dir
```

Each dataset is a folder `<name>/` containing `<name>_train.csv` and
`<name>_test.csv` with the target in a `target` column (TALENT-style layout).
Use `--custom-reg-dir` for local custom datasets.

## Adding a benchmark

The detailed intake checklist and candidate ledger live in
[`docs/evaluation/real-world-intake.md`](evaluation/real-world-intake.md) and
[`docs/evaluation/real-world-candidates.md`](evaluation/real-world-candidates.md). Dataset builders
belong under `scripts/evaluation/real_world/`; data artifacts never belong in git.

Most new datasets should use `DirectoryBenchmarkLoader`: stage a `manifest.json` plus explicit
train/test parquet tables under
`s3://synthefy-nori-eval-datasets/benchmarks/<source>/`. Add a custom loader only when the source
has a protocol that cannot be represented by table mode or index mode.

Before trusting a baseline, verify the dataset itself: use the source's published split where one
exists; otherwise split by time or entity whenever rows are related. Remove identifiers and
features unavailable at decision time, preserve missing values for the harness's declared
imputation policy, and confirm train/test schemas, target units, label support, duplicate rows,
and provenance. A plausible score is not evidence that these are wired correctly.

Each accepted dataset gets one end-to-end release gate, not a test per helper. The gate must cross
the production boundaries that can silently corrupt a number:

1. Load the staged manifest and real representative data through the actual loader.
2. Run a deterministic reference model through `run_benchmark` with `predictions_dir` enabled.
3. Replay the saved predictions with `check_predictions`; this verifies source-row identity and
   recomputes the declared metric family, including postprocessing.
4. Compare at least one headline result with an independent baseline or a defensible expected
   band. This catches a consistently wrong loader that round-trips its own mistake.

A failure of that gate means the benchmark cannot currently produce a trustworthy score and must
be fixed before merging. Avoid broad helper-level test matrices that can fail without invalidating
the benchmark; keep validation at the workflow boundary.

Production baselines should use the full published context. Do not pass a global `max_ctx` or
`max_test`; any protocol cap belongs to the benchmark declaration. Nori runs should also reject
silent memory-driven context subsampling. Persist every real sweep in one `predictions_dir` so it
streams `results.jsonl`, survives interruption, and retains the raw outputs needed to recompute a
metric without rerunning inference.

Finally, wire the benchmark's primary metric through its named-suite aggregation and the Supabase
leaderboard schema before promotion. Uploading raw run artifacts is reversible; publishing a
leaderboard baseline is a deliberate final step after the dataset, inference policy, replay, and
reference result have all been reviewed.

### Run a candidate benchmark

Build locally first, then run the same shared command against either that directory or its staged
S3 prefix:

```bash
uv run python scripts/evaluation/run_directory.py \
  --root /tmp/rw-example \
  --checkpoint /path/to/checkpoint.pt \
  --model-name nori-candidate \
  --predictions-dir /tmp/nori-evals/rw-example
```

For staged data, replace `--root` with
`s3://synthefy-nori-eval-datasets/benchmarks/rw-example`. Add `--dataset NAME` to smoke one
manifest entry. The command intentionally exposes no context or test-row cap: it uses the full
benchmark protocol and configures Nori with `allow_subsample=False`. It streams resumable results
to `PREDICTIONS_DIR/results.jsonl`, persists every prediction, replays the declared metric from
those artifacts, and exits nonzero on an inference or replay failure.

### Inspect a persisted run

The read-only explorer shows the selected dataset's numeric context and query tables, persisted
model output, and score metrics. The context table includes the labels supplied to the model; the
query table excludes its held-out target, while the output table joins that truth back for scoring.
The page also runs the same source-row and metric replay check used by the acceptance gate and only
surfaces it when verification fails.

`--runs-root` may be one predictions directory or a directory containing several of them:

```bash
uv run --extra dev --extra s3 --with uvicorn python scripts/evaluation/serve_explorer.py \
  --benchmark-root s3://synthefy-nori-eval-datasets/benchmarks/rw-example \
  --runs-root s3://synthefy-nori-eval-datasets/runs/rw-example/nori-baseline \
  --port 8765
```

The process binds only to remote loopback. From a workstation, forward the port and open
`http://localhost:8765`:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@eval-host
```

Both roots accept local paths or `s3://` prefixes. S3 objects are downloaded incrementally into
the eval cache, so a previously viewed run reopens without another manual sync. The UI has no
inference, upload, or Supabase-promotion actions.
