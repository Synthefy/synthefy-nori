# Nori

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Synthefy%2FNori-blue?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/Synthefy/Nori)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20710462.svg)](https://doi.org/10.5281/zenodo.20710462)

Nori is a tabular foundation model for **regression**
via in-context learning (ICL). Given a few labeled rows as context, it predicts on
new query rows in a single forward pass, with no task-specific training or fine-tuning.
The model is trained entirely on synthetic data.

This repository contains the public training, inference, evaluation, and Hugging
Face checkpoint tooling.

## Results

Mean and median R² of the base model across 96 regression tasks from three
public benchmark suites (~5.9M-parameter model):

| Suite | Datasets | Mean R² | Median R² |
|-------|---------:|--------:|----------:|
| TabArena | 13 | 0.8117 | 0.8757 |
| TALENT | 72 | 0.7569 | 0.8802 |
| OpenML | 11 | 0.6373 | 0.5856 |
| **Overall** | **96** | **0.7506** | **0.8702** |

Per-dataset numbers behind this table are in
[`benchmarks/benchmark_results.csv`](benchmarks/benchmark_results.csv); the
table is produced by the one-command run in
[Reproducing these numbers](#reproducing-these-numbers).

Large-N / long-context tables (common in TabArena) are the current focus of the
large-table training stages.

> **Thinking** is an inference-time reasoning extension that improves these
> numbers further. Details are forthcoming.

### Reproducing these numbers

```bash
pip install "synthefy-nori[eval]"

synthefy-nori-eval --download-benchmarks --openml-reg
```

The first run downloads the pretrained checkpoint from the Hugging Face Hub and
fetches the benchmark datasets into `cache/` as CSVs: TabArena from the
official TabArena curated uploads on OpenML (pinned by OpenML dataset ID, so
the data is immutable), TALENT from OpenML by name, and the OpenML regression
suite on the fly. Dataset membership is pinned by lists shipped with the
package (`synthefy_nori/evaluation/benchmark_lists/`), and train/test
splits use a fixed seed, so the evaluation data is fully deterministic.
Evaluation uses the bundled default inference config
(`reg_allordinal_poly10_adaptive_svd256.json`).

The benchmark uses the **large-GPU protocol**: up to 50,000 context rows per
dataset (no memory-based row cap) and an inference element budget of 8M
(`SYNTHEFY_MAX_ELEMENTS_BUDGET`, settable via `--max-elements-budget`). The
table was produced on a single H200. On smaller GPUs, pass `--gpu-mem-gb
<GiB>` to enable a memory-based cap on context rows and/or lower
`--max-elements-budget` — the run then fits in memory, but results on the
largest tables drop below the table above (more context is genuinely better).

The command prints a per-source mean R² summary matching the table above and
writes per-dataset metrics to `results/eval/all_results.csv`. Expect roughly
30–40 minutes on a single large GPU (`--device cuda:0` by default).

Exact per-dataset R² can move by ±0.001–0.002 across GPU models and
PyTorch/NumPy versions; per-source means should match the table to within
about ±0.003. The TALENT dataset `stock_fardamento02` has a heavy-tailed
target and is the least stable single dataset across environments.

## How it works

### Architecture

Nori is a **FeaturesTransformer (~5.9M parameters)** that alternates
two kinds of attention:

- **Feature attention** learns relationships between columns.
- **Sample attention** learns relationships between rows (context and query).
- **In-context learning**: predictions condition on labeled context rows, with no
  gradient updates at inference.

Key config: 16 transformer layers, embed_dim 128, hidden 384, 2 heads, the
**v2-lite** block (SwiGLU + RMSNorm + pre-norm), features grouped in pairs
(`features_per_group=2`), with **column-specific y-aware** feature attention.
Features are encoded with RBF embeddings; missing values are handled natively
via learned mask embeddings.

### Synthetic data

The model never sees real data during training. Its capability comes from a diverse
synthetic data generator covering real-world tabular regimes:

- **Structural Causal Models (SCM)**: hierarchical DAGs with 8 edge-function types
  (MLP, decision tree, piecewise-linear, polynomial, periodic, RBF, log/exp, conv1d).
- **Regression priors**: 9 target families (dense/sparse linear, GAM, interactions,
  random MLP, random tree, radial/RBF, Fourier features, chained trigonometric).
- **Realism augmentations**: discretized features, noise features, correlated blocks,
  structural missingness, label noise.
- **Learnability filter**: an ExtraTrees signal-quality filter rejects unlearnable
  datasets so training compute is spent on learnable tasks.

See [docs/training.md](docs/training.md) for the full recipe.

## Install

```bash
pip install synthefy-nori
```

Optional extras:

```bash
pip install "synthefy-nori[train]"   # training-only deps (wandb, xgboost)
pip install "synthefy-nori[eval]"    # evaluation-only deps (matplotlib, openml)
```

### Develop from source

```bash
git clone https://github.com/Synthefy/synthefy-nori
cd synthefy-nori
uv sync --extra dev
```

`uv sync` installs a **CUDA 12.8** PyTorch 2.8 build from PyTorch's wheel index.
The lock targets CUDA-capable platforms (Linux/Windows) only. If cu128 does not
match your driver, override the index in `[tool.uv.sources]` (e.g. swap
`pytorch-cu128` for `pytorch-cu126`) or install a matching PyTorch wheel yourself.
The Muon optimizer used in training prefers `torch.optim.Muon`; if your PyTorch
lacks it, the package automatically falls back to a built-in implementation.

## Authentication (optional)

The default checkpoint at
[`Synthefy/Nori`](https://huggingface.co/Synthefy/Nori)
is **public**: the first inference call downloads and caches it automatically,
with no token and no access request.

A Hugging Face token is only worth setting if you hit anonymous download rate
limits, or if you point the package at a private/gated checkpoint of your own.
Provide one in any of these ways:

```bash
# Option A: env var (one-shot)
export HF_TOKEN=hf_xxxxxxxx

# Option B: persist via the HF CLI (huggingface-hub >= 1.0)
hf auth login
```

```python
# Option C: pass explicitly in code
from synthefy_nori import NoriRegressor
model = NoriRegressor(token="hf_xxxxxxxx")
```

Get a token at <https://huggingface.co/settings/tokens> (read scope is
sufficient). If you supply a local `model_path=` instead, no network access is
needed at all.

## Inference

Pretrained weights are hosted on the Hugging Face Hub at
[`Synthefy/Nori`](https://huggingface.co/Synthefy/Nori).
The first call downloads and caches the checkpoint automatically, so a complete
working example is just:

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from synthefy_nori import NoriRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = NoriRegressor()    # downloads weights from the HF Hub on first use
model.fit(X_train, y_train)           # "fit" just stores the labeled rows as context
pred = model.predict(X_test)          # predictions in a single forward pass, no training
```

It uses a GPU when one is available and falls back to CPU. A one-shot helper
skips the object entirely:

```python
from synthefy_nori import predict
pred = predict(X_train, y_train, X_test, task="regression")
```

To run from your own checkpoint instead of the Hub default, pass a path:

```python
model = NoriRegressor(model_path="path/to/checkpoint.pt")
```

`predict` follows the `TabPFNRegressor.predict` contract: pass
`output_type="mean"` (default), `"median"`, or `"mode"` to choose the point
estimate drawn from the model's predictive distribution.

Runnable example: [`examples/inference_regression.py`](examples/inference_regression.py).
More detail in [docs/inference.md](docs/inference.md).

## Training

Smoke test (2 steps, single GPU, no logging):

```bash
TOTAL_STEPS=2 NPROC_PER_NODE=1 WANDB_MODE=disabled bash scripts/train.sh
```

Training runs entirely on synthetic data and **trains to completion**: there is
no real-data validation in the loop, so no benchmark data needs to
be downloaded to train, and no eval signal influences checkpoint selection. Each
run writes periodic and final checkpoints, and each curriculum tier seeds from
the previous tier's final checkpoint.

### Tier 1: from scratch

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh
```

Configurable via environment variables (`TOTAL_STEPS`, `LR`, `BATCH_SIZE`,
`CUDA_VISIBLE_DEVICES`, ...; see the script header). Checkpoints land in
`checkpoints/<run>/tier1/`.

### Tiers 2 to 5: curriculum continuation

One script runs the rest of the curriculum, each tier seeding from the previous
tier's final checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/continue_training.sh
```

| Tier | Table shapes (N x F) | Focus |
|---|---|---|
| 2 | N ≤ 4K, F ≤ 384 | larger tables |
| 3 | N ≤ 8K, F ≤ 768 | largest tables |
| 4 | N ≤ 56K, F ≤ 96 | large-N / long-context specialist |
| 5 | N ≤ 33K, F ≤ 1280 | both-large corner (N and F coupled by a cell budget) |

It auto-detects the most recent tier-1 run, or point it at one with
`RUN_ROOT=checkpoints/<run>`. Run a subset with `START_TIER` / `END_TIER`
(e.g. `END_TIER=3` for tiers 2 to 3 only).

> **Tiers 4 and 5 push N up to 56K rows.** Dense O(N²) sample attention at that
> scale forces `batch=1` with large gradient accumulation, and can OOM or hang
> depending on GPU memory. Smoke-probe them first; see the script header.

Training uses the **Muon** optimizer (EMA 0.999), a **pinball** loss with 999
quantiles + a monotonicity penalty, and bf16 mixed precision with DDP. Pass
`--seed` for reproducible runs. Full options: [docs/training.md](docs/training.md).

## Evaluation

```bash
synthefy-nori-eval --checkpoint "Synthefy:path/to/checkpoint.pt"
```

or `bash scripts/evaluate.sh`. See [docs/evaluation.md](docs/evaluation.md) for
benchmark sources and how to evaluate a Nori checkpoint, and
[Reproducing these numbers](#reproducing-these-numbers) for the published
benchmark run.

## Interpretability

`NoriRegressor` is a scikit-learn estimator, so it works directly with shapiq and
the sklearn interpretability ecosystem — Shapley values & interactions, partial
dependence / ICE, and sequential feature selection:

```bash
pip install "synthefy-nori[interpretability]"
```

```python
from synthefy_nori import NoriRegressor
from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

model = NoriRegressor().fit(X_train, y_train)
explainer = get_nori_imputation_explainer(model, X_train)   # imputation-based, model-agnostic
sv = explainer.explain(X_test[:1], budget=128)              # Shapley values for one prediction
sv.plot_waterfall()
```

Runnable: [`examples/interpretability_regression.py`](examples/interpretability_regression.py).
More detail in [docs/interpretability.md](docs/interpretability.md).

## Benchmarks

The published [Results](#results) table is produced by the packaged CLI — see
[Reproducing these numbers](#reproducing-these-numbers). Per-dataset metrics
are committed at
[`benchmarks/benchmark_results.csv`](benchmarks/benchmark_results.csv).

An alternative script-style harness that drives the public
`NoriRegressor` API directly lives at
[`tests/test_benchmark_performance.py`](tests/test_benchmark_performance.py).
It reads the same CSV caches under `./cache/`; populate them once with
`synthefy-nori-eval --download-benchmarks` (TabArena from the official
TabArena uploads on OpenML pinned by dataset ID, TALENT by name), then run
from the repo root (`uv sync` installs a CUDA 12.8 torch build on Linux, so
`uv run` works as-is):

```bash
# OpenML only — works out of the box, no cached CSVs needed
uv run python tests/test_benchmark_performance.py --suites openml

# full sweep over the downloaded caches
uv run python tests/test_benchmark_performance.py --device cuda:0
```

Note the script's OpenML suite uses its own 70/30 split (the packaged CLI uses
80/20), so its OpenML numbers differ slightly from the Results table.

## Hugging Face

```bash
synthefy-nori-download                                            # fetch default checkpoint
synthefy-nori-upload path/to/checkpoint.pt --repo-id Synthefy/Nori
```

See [docs/huggingface.md](docs/huggingface.md).

## Repository layout

```
src/synthefy_nori/
  api.py            Public API (NoriRegressor, infer, predict)
  model/            FeaturesTransformer architecture
  training/         Data generation, trainer, loss, config, CLI
  inference/        Sklearn-compatible predictor + preprocessing
  evaluation/       Benchmark runner over public benchmark suites
  hf.py             Hugging Face download / upload
scripts/            train.sh, continue_training.sh, evaluate.sh
docs/               training, inference, evaluation, huggingface guides
examples/           Runnable inference / upload scripts
```

## Citation

If you use this project, please cite it as:

```bibtex
@software{synthefy_2026_20710462,
  author       = {Synthefy and
                  Li, Po-han and
                  Narayanan, Aditya and
                  Narasimhan, Sai Shankar and
                  Mallampalli, Raghav and
                  Agrawal, Aahan and
                  Ajan, Bekzat and
                  Shah, Raimi and
                  Agarwal, Shubhankar},
  title        = {Synthefy Nori: Tabular Foundation Model for Regression},
  month        = jun,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {0.3.0},
  doi          = {10.5281/zenodo.20710462},
  url          = {https://doi.org/10.5281/zenodo.20710462},
}
```

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
