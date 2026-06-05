# Synthefy Tabular

Synthefy Tabular is a tabular foundation model for **regression and classification**
via in-context learning (ICL). Given a few labeled rows as context, it predicts on
new query rows in a single forward pass, with no task-specific training or fine-tuning.
The model is trained entirely on synthetic data.

This repository contains the public training, inference, evaluation, and Hugging
Face checkpoint tooling.

## Results

Mean R² across 96 regression tasks from three public benchmark suites:

| Source | Tasks | Mean R² |
|--------|------:|--------:|
| OpenML Regression | 11 | 0.6104 |
| TabArena | 13 | 0.8089 |
| TALENT | 72 | 0.7591 |

Large-N / long-context tables (common in TabArena) are the current focus of the
large-table training stages.

> **Thinking** is an inference-time reasoning extension. Details are forthcoming.

## How it works

### Architecture

Synthefy Tabular is a **FeaturesTransformer (~5.5M parameters)** that alternates
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
  structural missingness, label noise, class imbalance.
- **Learnability filter**: an ExtraTrees signal-quality filter rejects unlearnable
  datasets so training compute is spent on learnable tasks.

See [docs/training.md](docs/training.md) for the full recipe.

## Install

```bash
pip install synthefy-tabular
```

Optional extras:

```bash
pip install "synthefy-tabular[train]"   # training-only deps (wandb, xgboost)
pip install "synthefy-tabular[eval]"    # evaluation-only deps (matplotlib, openml)
```

### Develop from source

```bash
git clone https://github.com/Synthefy/synthefy-tabular
cd synthefy-tabular
uv sync --extra dev
```

`uv sync` installs a pinned PyTorch build. If that CUDA build does not match your
driver, install a PyTorch wheel matching your CUDA version instead. The Muon
optimizer used in training prefers `torch.optim.Muon`; if your PyTorch lacks it,
the package automatically falls back to a built-in implementation.

## Authentication

The default model checkpoint lives at
[`Synthefy/synthefy-tabular`](https://huggingface.co/Synthefy/synthefy-tabular)
on the Hugging Face Hub and currently requires access. To use the package:

1. Request access at the model page (or contact Synthefy if the page isn't
   visible to you yet).
2. Provide your Hugging Face token in any one of these ways:

   ```bash
   # Option A: env var (one-shot)
   export HF_TOKEN=hf_xxxxxxxx

   # Option B: persist via the HF CLI
   huggingface-cli login
   ```

   ```python
   # Option C: pass explicitly in code
   from synthefy_tabular import SynthefyTabularRegressor
   model = SynthefyTabularRegressor(token="hf_xxxxxxxx")
   ```

Get a token at <https://huggingface.co/settings/tokens> (read scope is
sufficient). If you supply a local `model_path=` instead, no token is needed.

## Inference

Pretrained weights are hosted on the Hugging Face Hub at
[`Synthefy/synthefy-tabular`](https://huggingface.co/Synthefy/synthefy-tabular).
The first call downloads and caches the checkpoint automatically, so a complete
working example is just:

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from synthefy_tabular import SynthefyTabularRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = SynthefyTabularRegressor()    # downloads weights from the HF Hub on first use
model.fit(X_train, y_train)           # "fit" just stores the labeled rows as context
pred = model.predict(X_test)          # predictions in a single forward pass, no training
```

It uses a GPU when one is available and falls back to CPU. Classification uses
`SynthefyTabularClassifier` with the same `fit` / `predict` API. A one-shot
helper skips the object entirely:

```python
from synthefy_tabular import predict
pred = predict(X_train, y_train, X_test, task="regression")
```

To run from your own checkpoint instead of the Hub default, pass a path:

```python
model = SynthefyTabularRegressor(model_path="path/to/checkpoint.pt")
```

Runnable examples: [`examples/inference_regression.py`](examples/inference_regression.py),
[`examples/inference_classification.py`](examples/inference_classification.py).
More detail in [docs/inference.md](docs/inference.md).

## Training

Smoke test (2 steps, single GPU, no logging):

```bash
TOTAL_STEPS=2 NPROC_PER_NODE=1 WANDB_MODE=disabled bash scripts/train.sh
```

Training runs entirely on synthetic data and **trains to completion**: there is
no real-data validation in the loop (`--no-eval`), so no benchmark data needs to
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
synthefy-tabular-eval --checkpoint "Synthefy:path/to/checkpoint.pt"
```

or `bash scripts/evaluate.sh`. See [docs/evaluation.md](docs/evaluation.md) for
benchmark sources and how to evaluate a Synthefy Tabular checkpoint.

## Hugging Face

```bash
synthefy-tabular-download                                            # fetch default checkpoint
synthefy-tabular-upload path/to/checkpoint.pt --repo-id Synthefy/synthefy-tabular
```

See [docs/huggingface.md](docs/huggingface.md).

## Repository layout

```
src/synthefy_tabular/
  api.py            Public API (SynthefyTabularRegressor / Classifier, infer, predict)
  model/            FeaturesTransformer architecture
  training/         Data generation, trainer, loss, config, CLI
  inference/        Sklearn-compatible predictor + preprocessing
  evaluation/       Benchmark runner over public suites (Synthefy / LimiX)
  hf.py             Hugging Face download / upload
scripts/            train.sh, continue_training.sh, evaluate.sh
docs/               training, inference, evaluation, huggingface guides
examples/           Runnable inference / upload scripts
```

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
