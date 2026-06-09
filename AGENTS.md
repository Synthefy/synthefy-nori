# AGENTS.md

Guidance for coding agents (Claude Code, Cursor, Codex, etc.) working in this
repo. Keep it accurate — update it when commands, layout, or conventions change.

## What this is

`synthefy-tabular` is a small (~5.5M-parameter) tabular foundation model
(`FeaturesTransformer`) for **regression and classification** via in-context
learning. Given a few labeled context rows, it predicts on query rows in a
single forward pass — no task-specific training. It is trained entirely on
synthetic data. The public API wraps an internal `SynthefyTabularPredictor`.

## Setup

- Python **≥ 3.10**. The interpreter and dependencies are managed by **uv**
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

- Import smoke (what CI gates on first): `uv run python -c "import synthefy_tabular"`
- Full inference check (downloads the public ~47MB checkpoint, ~15s on CPU):
  `uv run pytest -m slow`

## How inference works (and how to test it)

- The default checkpoint lives at the **public** HF repo
  `Synthefy/synthefy-tabular` (file `synthefy-tabular.pt`). First use downloads
  and caches it — **no token or access request needed**. A token is only used
  for higher rate limits or for pointing at a private/custom repo.
- Public API (`src/synthefy_tabular/api.py`): `SynthefyTabularRegressor` and
  `SynthefyTabularClassifier` (sklearn-style `fit` / `predict` /
  `predict_proba`), plus the one-shot `infer` / `predict` helpers and
  `config_path`.
- `fit()` only stores the context rows; all compute happens in `predict()`.
  Uses GPU when available, else CPU.
- Pass `model_path="…/checkpoint.pt"` to run a local checkpoint and skip the
  download entirely.

## Layout

```
src/synthefy_tabular/
  api.py          Public API. Imports the heavy stack LAZILY — keep it import-light.
  hf.py           HF download/upload + console-script entry points
  model/          FeaturesTransformer architecture
  inference/      SynthefyTabularPredictor + preprocessing
  training/       data generation, trainer, loss, config, CLI (GPU / DDP)
  evaluation/     benchmark runner + CLI
  configs/*.json  bundled inference configs (shipped via package-data)
scripts/          train.sh, continue_training.sh, evaluate.sh
docs/             training / inference / evaluation / huggingface guides
examples/         runnable inference + upload scripts
tests/            fast unit/smoke tests + slow e2e tests (marked `slow`)
```

## Conventions & gotchas

- **Keep `api.py` and the top-level import cheap.** `torch`/`numpy`/etc. are
  imported lazily inside functions so `import synthefy_tabular` works before the
  heavy accelerator stack loads. Do not hoist heavy imports to module top level.
- **Ruff is intentionally narrow**: only `E9,F821,F822,F823` (syntax +
  undefined names), line length 120, target `py310`. It is a correctness gate,
  not a full style/format gate. A pre-commit hook (`ruff`) is configured.
- **Network tests are marked `slow`** and deselected via
  `addopts = -m 'not slow'`. Plain `pytest` stays offline and fast (~0.1s).
- **Never commit checkpoints or data.** `.gitignore` covers
  `*.pt`/`*.ckpt`/`*.safetensors`, `checkpoints/`, `data/`, `results/`,
  `wandb/`, `cache/`.
- **Versioning**: keep `pyproject.toml` `version` and `__init__.py`
  `__version__` in sync — the publish workflow enforces a match with the git
  tag. Release process is in `RELEASING.md`.
- **Training is GPU + DDP** via `scripts/train.sh` (torchrun); it is not meant
  to run meaningfully on CPU. Smoke-test the wiring with:
  `TOTAL_STEPS=2 NPROC_PER_NODE=1 WANDB_MODE=disabled bash scripts/train.sh`.
- Shell scripts run with the project venv (`.venv/bin/python`); they do not rely
  on a bare `python`.

## Provenance / attribution (do not strip)

The codebase originated as a fork of **LimiX** (Apache-2.0,
<https://github.com/limix-ldm-ai/LimiX>) and has since diverged; the vestigial
`LimiX*` class names were renamed to `SynthefyTabular*`. The **canonical LimiX-2M
model is used in exactly one place** — an optional training-time ICL learnability
filter, downloaded via `download_limix()` / `--icl-filter-model limix`. Those
references name the real upstream model — leave them intact, and don't
re-introduce `LimiX*` naming for Synthefy's own classes. Parts of the
synthetic-data prior generator draw on **TabICL** (BSD-3-Clause,
<https://github.com/soda-inria/tabicl>).

Keep the `LICENSE`/`NOTICE` attributions intact; check those files before
removing upstream names.
