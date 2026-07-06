# AGENTS.md

Guidance for coding agents (Claude Code, Cursor, Codex, etc.) working in this
repo. Keep it accurate — update it when commands, layout, or conventions change.

## What this is

`synthefy-nori` is a small (~5.5M-parameter) tabular foundation model
(`FeaturesTransformer`) for **regression** via in-context
learning. Given a few labeled context rows, it predicts on query rows in a
single forward pass — no task-specific training. It is trained entirely on
synthetic data. The public API wraps an internal `NoriPredictor`.

## Setup

- Python **≥ 3.9**. The interpreter and dependencies are managed by **uv**
  (`uv.lock` is committed). There may be no bare `python` on PATH — use `uv run`.
- Install everything (incl. dev tools): `uv sync --extra dev`
- Optional extras: `--extra train` (wandb, xgboost), `--extra eval`
  (matplotlib, openml).

## Core commands (these mirror CI — run them before any PR)

```bash
uv sync --extra dev
uv run pytest                       # fast suite; slow/network tests deselected by default
uv run ruff check src scripts tests
uv build
```

- Import smoke (what CI gates on first): `uv run python -c "import synthefy_nori"`
- Full inference check (downloads the public ~47MB checkpoint, ~15s on CPU):
  `uv run pytest -m slow`
- **Before finalizing a PR, run `/review` and/or `/code-review`** to catch bugs
  and reuse/simplification issues in the diff. (`/code-review` reviews the
  current diff for correctness + cleanups; `/review` reviews a pull request.)

## How inference works (and how to test it)

- The default checkpoint lives at the **public** HF repo
  `Synthefy/Nori` (file `nori.pt`). First use downloads
  and caches it — **no token or access request needed**. A token is only used
  for higher rate limits or for pointing at a private/custom repo.
- Public API (`src/synthefy_nori/api.py`): `NoriRegressor`
  (sklearn-style `fit` / `predict`; `predict` takes
  `output_type="mean"/"median"/"mode"` per the `TabPFNRegressor` contract),
  plus the one-shot `infer` / `predict` helpers and `config_path`. The public
  API is **regression-only** as of 0.2.0 (classification was removed in #10).
- `fit()` only stores the context rows; all compute happens in `predict()`.
  Uses GPU when available, else CPU.
- Pass `model_path="…/checkpoint.pt"` to run a local checkpoint and skip the
  download entirely.

## Layout

```
src/synthefy_nori/
  api.py          Public API (sklearn-style NoriRegressor + one-shot helpers).
  hf.py           HF download/upload + console-script entry points
  model/          FeaturesTransformer architecture
  inference/      NoriPredictor + preprocessing
  embedding/      NoriEmbedding (sklearn transformer for row embeddings)
  training/       data generation, trainer, loss, config, CLI (GPU / DDP)
  evaluation/     benchmark runner + CLI
  configs/*.json  bundled inference configs (shipped via package-data)
scripts/          train.sh, continue_training.sh, evaluate.sh
docs/             training / inference / evaluation / huggingface guides
examples/         runnable inference + upload scripts
tests/            fast unit/smoke tests + slow e2e tests (marked `slow`)
```

## Conventions & gotchas

- **Imports go at the top of the file, never inside functions.** `torch`,
  `numpy`, etc. are imported at module top (`torch` is a hard dependency, so
  `import synthefy_nori` loads it regardless). The one deliberately deferred
  import in `api.py` is the internal `NoriPredictor` (loaded in
  `_get_predictor`), kept lazy only to avoid pulling the model/inference stack
  in at package-import time — not a license to move package imports into
  functions.
- **Ruff is intentionally narrow**: only `E9,F821,F822,F823` (syntax +
  undefined names), line length 120, target `py310`. It is a correctness gate,
  not a full style/format gate. A pre-commit hook (`ruff`) is configured.
- **Network tests are marked `slow`** and deselected via
  `addopts = -m 'not slow'`. Plain `pytest` stays offline and fast (~0.1s).
- **Never commit checkpoints or data.** `.gitignore` covers
  `*.pt`/`*.ckpt`/`*.safetensors`, `checkpoints/`, `data/`, `results/`,
  `wandb/`, `cache/`.
- **Versioning & distribution**: the package is **public** and published to PyPI
  (`pip install synthefy-nori`). Keep `pyproject.toml` `version` and
  `__init__.py` `__version__` in sync — `publish.yml` enforces a match with the
  release tag and uploads to PyPI over OIDC trusted publishing. Release process is
  in `RELEASING.md`.
- **Training is GPU + DDP.** Real runs go through `scripts/train.sh` (torchrun)
  on one or more CUDA GPUs; the distributed path places each rank on
  `cuda:<rank>`. Heads-up: the non-distributed `--device` default is `cuda:2`,
  so `NPROC_PER_NODE=1 bash scripts/train.sh` only works on a box with ≥3 GPUs.
- **CPU smoke (no GPU needed)** — exercises the full loop (data-gen →
  CCMM/pinball loss → Muon step → checkpoint write) in 2 steps:

  ```bash
  WANDB_MODE=disabled uv run synthefy-nori-train \
    --device cpu --no-mixed-precision --no-prefetch \
    --task-type reg --total-steps 2 --run-steps 2 --save-interval 2 \
    --embed-dim 32 --hid-dim 64 --nlayers 2 --nhead 2 \
    --batch-size 2 --max-features 16 --max-budget 4000 \
    --checkpoint-dir /tmp/nori_smoke
  ```
- Shell scripts run with the project venv (`.venv/bin/python`); they do not rely
  on a bare `python`.

## Provenance / attribution (do not strip)

The codebase originated as a fork of **LimiX** (Apache-2.0,
<https://github.com/limix-ldm-ai/LimiX>) and has since diverged; the vestigial
`LimiX*` class names were renamed to `Nori*`. The **canonical LimiX-2M
model is used in exactly one place** — an optional training-time ICL learnability
filter, downloaded via `download_limix()` / `--icl-filter-model limix`. Those
references name the real upstream model — leave them intact, and don't
re-introduce `LimiX*` naming for Synthefy's own classes. Parts of the
synthetic-data prior generator draw on **TabICL** (BSD-3-Clause,
<https://github.com/soda-inria/tabicl>).

Keep the `LICENSE`, `NOTICE`, and `licenses/` attributions intact; check those
files before removing upstream names. The full upstream license texts live in
`licenses/` and are shipped via `license-files` in `pyproject.toml`.
