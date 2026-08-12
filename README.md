<p align="center">
  <img src="synthefy_nori_banner.png" alt="Nori" width="100%">
</p>

# Nori

[![Docs](https://img.shields.io/badge/Docs-docs.synthefy.com-2ea44f?logo=readthedocs&logoColor=white)](https://docs.synthefy.com/nori/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Synthefy%2FNori-blue?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/Synthefy/Nori)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20710462.svg)](https://doi.org/10.5281/zenodo.20710462)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/jpsXMXGza)
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Synthefy/synthefy-nori/blob/main/examples/notebooks/Nori_Demo_Local.ipynb)

Nori is a tabular foundation model for **regression**
via in-context learning (ICL). Given a few labeled rows as context, it predicts on
new query rows in a single forward pass, with no task-specific training or fine-tuning.
The model is trained entirely on synthetic data.

This repository contains the public training, inference, evaluation, and Hugging
Face checkpoint tooling.

Across 96 public regression tasks the base (~6M) averages **0.75 mean / 0.87 median R²**, and the
larger **Nori-30M** variant (`model="nori-30m"`) is stronger on every suite — see
[Benchmarks](#benchmarks) for the full breakdown and how to reproduce it.

## Table of contents

- [Use it from your AI coding assistant](#use-it-from-your-ai-coding-assistant)
- [Install](#install)
- [Quickstart](#quickstart)
- [Authentication](#authentication-optional)
- [How it works](#how-it-works)
- [Interpretability](#interpretability)
- [Benchmarks](#benchmarks)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hugging Face](#hugging-face)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

## Use it from your AI coding assistant

Paste this into Claude Code, Cursor, or any AI coding assistant and it will wire
Nori into your own project:

````text
Look at my code/task/report here and figure out where Nori would best fit — it's
Synthefy's tabular foundation model, a drop-in scikit-learn estimator that predicts
a continuous target by in-context learning: no training loop, no hyperparameters,
and it uses the GPU automatically when one's available (CPU otherwise).

1. Install it with this project's package manager
   (e.g. `uv add synthefy-nori`, or `pip install -U synthefy-nori`).

2. Use it wherever a tabular regression / prediction step fits:

   ```python
   from synthefy_nori import NoriRegressor

   reg = NoriRegressor(model="nori")      # "nori-30m" for the larger, stronger variant
   reg.fit(X_train, y_train)              # stores your rows as context — no training happens
   y_pred = reg.predict(X_test)           # point predictions (predictive-distribution mean)

   # Prediction intervals come free — no conformal/quantile add-ons:
   lo, mid, hi = reg.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
   ```

X is a numeric feature matrix (encode categoricals as ordinals/one-hot, leave
missing values as NaN, no scaling needed); y is a finite continuous target. If I
already have a model, wire Nori up alongside it on the same train/test split and
metric so I can compare them. If the best place to plug Nori in isn't obvious,
show me where you'd put it and confirm with me before making changes.

Going deeper: synthefy-nori ships a ready-made nori-regression skill for AI coding
assistants with vetted recipes — calibrated prediction intervals, honest baseline
comparison under fixed CV, SHAP/PDP interpretability, and leak-safe one-step
time-series forecasting. Read and follow it if relevant:
https://github.com/Synthefy/synthefy-nori/tree/main/.claude/skills/nori-regression
````

## Install

This source tree builds two independently versioned distributions with disjoint
namespaces:

| Use case | Install | Public entry point |
|---|---|---|
| Hosted regression | `pip install synthefy` | `SynthefyNoriClient(mode="remote", model=...)` |
| SageMaker regression | `pip install "synthefy[aws]"` | `SynthefyNoriClient(mode="sagemaker", model=..., endpoint_name=...)` |
| Local regression | `pip install synthefy-nori` | `NoriRegressor(model=...)` or `SynthefyNoriClient(mode="local", model=...)` |
| Hosted forecasting | `pip install "synthefy[forecasting]"` | `synthefy.nori_ts.NoriTSForecaster(mode="remote", model=...)` |
| Local forecasting | `pip install "synthefy-nori[forecasting]"` | `synthefy.nori_ts.NoriTSForecaster(mode="local", model=...)` |

Hosted users receive the lightweight client and workflow code without Torch or
model weights. Local users install `synthefy-nori`, which supplies the model
runtime and depends on the lightweight `synthefy` package. The dependency never
points in the other direction.

For local regression:

```bash
pip install synthefy-nori
```

For hosted regression, the client reads `SYNTHEFY_NORI_API_KEY` unless
`api_key=` is supplied:

```bash
pip install synthefy
export SYNTHEFY_NORI_API_KEY="<your key>"
```

```python
from synthefy import SynthefyNoriClient

client = SynthefyNoriClient(mode="remote", model="nori-30m")
predictions = client.predict([[0.0], [1.0]], [0.0, 1.0], [[0.5]])
client.close()
```

Execution modes are limited to `remote`, `sagemaker`, and `local`; `auto` is
not supported. The Nori client retains `remote` when `mode=` is omitted for
backwards compatibility, while the forecaster requires an explicit mode. Both
require an explicit `model=`; there is no default model.

Optional heavyweight extras:

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

`uv sync` installs `torch` from PyPI, whose default wheel is a **CUDA 12.8**
build on Linux (CPU on Windows) — enough to reproduce the benchmarks without a
manual step. The package pins **no** torch index and supports the tested
`torch>=2.8,<2.14` range, so it composes cleanly as a git/path dependency: a
consumer picks its own torch build with no index conflict. To use a different
CUDA build, add your own `[[tool.uv.index]]` + `[tool.uv.sources]` in *your*
project — e.g. `pytorch-cu130` (needs torch >= 2.9) or `pytorch-cu132` (needs
torch >= 2.12), both requiring Python >= 3.10 since torch dropped 3.9 in 2.9.

For a one-off CUDA 13.0 environment in this checkout:

```bash
nori_cuda_venv="$(mktemp -d)"
uv venv --python 3.11 "$nori_cuda_venv"
uv pip install --python "$nori_cuda_venv/bin/python" --no-config \
  "torch>=2.9,<2.14" --torch-backend=cu130
uv pip install --python "$nori_cuda_venv/bin/python" --no-config -e ".[dev]"
"$nori_cuda_venv/bin/python" -c "import torch; print(torch.__version__, torch.version.cuda)"
```

The separate environment avoids mixing CUDA 12 and CUDA 13 NVIDIA packages in
the benchmark `.venv`; `--no-config` bypasses this checkout's deliberate
`torch<2.9` development constraint for those commands. The repo-local
`constraint-dependencies` that holds the normal lock is not read by consumers.

Nori excludes cuDNN from PyTorch SDPA dispatch by default because cuDNN attention
has been unreliable on the model's dynamic tabular shapes and small attention
heads. The restriction is scoped to Nori's attention call and leaves global
PyTorch backend settings untouched. Set `SYNTHEFY_NORI_ALLOW_CUDNN_SDP=1` before
importing Nori to opt back into PyTorch's default backend selection.
The Muon optimizer used in training prefers `torch.optim.Muon`; if your PyTorch
lacks it, the package automatically falls back to a built-in implementation.

## Quickstart

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

### Probabilistic output (quantiles)

The default checkpoint has a 999-quantile pinball head, so the full predictive
distribution is available — not just a point estimate. Use
`output_type="quantiles"` for specific levels, or `output_type="full"` for the
whole quantile bank (handy for CRPS / interval scoring, calibration, and
prediction intervals):

```python
model = NoriRegressor().fit(X_train, y_train)

# Quantiles at chosen levels -> shape (n_levels, n_samples)
q10, q50, q90 = model.predict(X_test, output_type="quantiles",
                              quantiles=[0.1, 0.5, 0.9])

# Full distribution as a per-row quantile function
dist = model.predict(X_test, output_type="full")
dist["quantiles"]  # (n_samples, K) ascending quantile values, K = 999
dist["taus"]       # (K,) quantile levels, evenly spaced in (0, 1)
dist["mean"]       # (n_samples,) distribution mean (== output_type="mean")
```

Quantiles are returned in original-`y` units and sorted to a valid (monotone)
quantile function per row. `quantiles`/`full` require the default pinball
checkpoint; a `bar_distribution` checkpoint raises `NotImplementedError`.

### Categorical / ordinal targets

If the target only takes a few discrete values (ratings, counts, quality
scores), pass `discretize=` and predictions are mapped onto the levels seen in
`fit`'s `y` — `"map-cell"` (the mode of the induced discrete posterior) is
best for accuracy, `"median-cell"` is the MAE-optimal alternative,
`"snap-mean"`/`"snap-median"` snap the point estimate. Snapping is strictly
opt-in — the default `predict()` always returns the continuous point estimate —
and for R²-scored tasks that default is what you want. See
[docs/inference.md](docs/inference.md#categorical--ordinal-targets-discretize--categorical_levels).

```python
labels = model.predict(X_test, discretize="map-cell")  # values from y_train's lattice
```

Runnable example: [`examples/inference_regression.py`](examples/inference_regression.py).
More detail in [docs/inference.md](docs/inference.md).

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

## How it works

### Architecture

Nori is a **FeaturesTransformer (~6M parameters)** that alternates
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

## Interpretability

Explain Nori's predictions with **SHAP / Shapley values**, feature interactions,
partial dependence / ICE, and sequential feature selection — see which features
drive a prediction, detect interactions, and debug unexpected outputs. Because
`NoriRegressor` is a scikit-learn estimator, it works directly with
[shapiq](https://github.com/mmschlk/shapiq) (a fast SHAP implementation with
native Shapley-interaction support) and the sklearn interpretability ecosystem —
no adapters needed beyond the thin convenience wrappers in
`synthefy_nori.interpretability`.

```bash
pip install "synthefy-nori[interpretability]"
```

```python
from synthefy_nori import NoriRegressor
from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

model = NoriRegressor().fit(X_train, y_train)
explainer = get_nori_imputation_explainer(model, X_train)   # imputation-based, model-agnostic
sv = explainer.explain(X_test[:1], budget=128)              # SHAP/Shapley values for one prediction
sv.plot_waterfall()                                         # additive contribution waterfall
```

Also available: `interpretability.pdp.partial_dependence_plots` (global feature
effects) and `interpretability.feature_selection.feature_selection`. Regression
only. Runnable example:
[`examples/interpretability_regression.py`](examples/interpretability_regression.py);
full guide in [docs/interpretability.md](docs/interpretability.md).

### SHAPIQ explanation speed

SHAPIQ's baseline imputer stacks every coalition of an explanation into a single
batched `model.predict` call, so on Nori the per-explanation wall-clock is
dominated by one forward pass — encoding the fixed training context — and is
essentially **flat in the coalition budget**. On a single H200 (superconductivity,
top-12 features, 1500 context rows, mean over 5 test rows), order-1 Shapley values
(`index="SV"`) cost ~2.2–2.3 s/explanation and order-2 pairwise interactions
(`max_order=2`, k-SII) ~1.3–1.5 s/explanation across budgets of 32–512, both
hovering near the ~1.9 s single-batched-predict floor. The takeaway: you can raise
`budget` for far more accurate attributions at near-zero extra cost, and
interaction order barely changes runtime.

Reproduce (prints the results table and writes `benchmarks/plots/shapiq_speed.png`):
`uv run python benchmarks/bench_shapiq_speed.py`

### SHAP explanation speed

The classic `shap` library also works on Nori via a `model.predict` callable. The
benchmark below measures per-row explanation time as the evaluation budget grows on
a 12-feature subset of `superconductivity` (context = 1500 rows, 5 test rows, H200).
Because every coalition is one Nori forward pass, `shap.KernelExplainer` stays
roughly flat (~3.0–3.7 s/row from nsamples 32 to 256, ~6.2 s/row at 512) while
`shap.PermutationExplainer` scales near-linearly with `max_evals`, climbing from
~5 s/row to ~28 s/row at 512. For comparison, shapiq's imputation explainer
(`index="SV"`) computes the same single-feature Shapley values in ~0.9–1.6 s/row
regardless of budget — roughly 2–4× faster than KernelExplainer and up to ~20×
faster than PermutationExplainer at high budgets.

Reproduce (prints the results table and writes `benchmarks/plots/shap_speed.png`):
`uv run python benchmarks/bench_shap_speed.py`

## Benchmarks

Mean and median R² across 96 regression tasks from three public benchmark suites, for both
Nori sizes — select with `model="nori"` (~6M, default) or `model="nori-30m"` (~29M):

| Suite | Datasets | Nori · mean / median | Nori-30M · mean / median |
|-------|---------:|:--------------------:|:------------------------:|
| TabArena | 13 | 0.8117 / 0.8757 | 0.8148 / 0.8834 |
| TALENT | 72 | 0.7569 / 0.8802 | 0.7575 / 0.8844 |
| OpenML | 11 | 0.6373 / 0.5856 | 0.6459 / 0.6212 |
| **Overall** | **96** | **0.7506 / 0.8702** | **0.7525 / 0.8745** |

Nori-30M is stronger on every suite. Both models are evaluated under the identical protocol
below. Per-dataset numbers behind the base-model column are in
[`benchmarks/benchmark_results.csv`](benchmarks/benchmark_results.csv).

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

### Script-style harness

An alternative harness drives the public `NoriRegressor` API directly at
[`tests/test_benchmark_performance.py`](tests/test_benchmark_performance.py).
It reads the same CSV caches under `./cache/`; populate them once with
`synthefy-nori-eval --download-benchmarks` (TabArena from the official
TabArena uploads on OpenML pinned by dataset ID, TALENT by name), then run
from the repo root (`uv sync` installs a CUDA 12.8 torch build on Linux by
default, so `uv run` works as-is):

```bash
# OpenML only — works out of the box, no cached CSVs needed
uv run python tests/test_benchmark_performance.py --suites openml

# full sweep over the downloaded caches
uv run python tests/test_benchmark_performance.py --device cuda:0
```

Note the script's OpenML suite uses its own 70/30 split (the packaged CLI uses
80/20), so its OpenML numbers differ slightly from the table above.

### RelBench (relational tasks)

Nori also runs on the [RelBench](https://relbench.stanford.edu) leaderboard
tasks via the entity-table tabular protocol (the regime tabular foundation
models like TabPFN are listed under), covering the classification (AUROC) and
regression (MAE) entity tasks across the seven canonical RelBench datasets:

```bash
pip install "synthefy-nori[relbench]"
synthefy-nori-eval --relbench
```

Results (split by task type) and a submission package land in
`results/relbench/`. See [`docs/evaluation.md`](docs/evaluation.md#relbench-relational-tasks)
for details, including the current RelBench submission status.

## Performance (inference speedups)

The speedups below are **on by default** and **deterministic** — identical results
run-to-run with the same settings — and the published [Results](#results) were
produced with them on. The **KV cache** is **R²-equivalent** to the un-cached path,
not bit-identical: the two paths reduce in a different order, so predictions differ by
mixed-precision noise (max abs diff ~3e-3, measured below) at identical R². The
**preprocessing speedups** are **R²-neutral**:
toggling them shifts individual predictions by a tiny, R²-equivalent amount (below
cross-environment noise), not bit-for-bit. For the exact un-accelerated path, set
each to its off value (see below).

| Env var | Default | What it does |
|---|---|---|
| `SYNTHEFY_GPU_SVD` | `1` (on) | Run the high-dimensional feature SVD on the GPU (exact, not randomized). Acts when features ≥256; set `0` for the CPU/randomized path. |
| `SYNTHEFY_CAP_QUANTILES` | `1` (on) | Cap quantile-transform resolution + subsample its fit. Acts on large context (>2000 rows); set `0` to disable. |
| `SYNTHEFY_QUANTILE_MAX` / `SYNTHEFY_QUANTILE_SUBSAMPLE` | — | Tune the cap above (max quantiles / fit-subsample size). |
| `SYNTHEFY_ADAPTIVE_FIT_SUBSAMPLE` | `2000` | Fit preprocessing on at most this many rows, apply to all rows. Acts on large context; set `0` to fit on all rows. |
| `SYNTHEFY_ENABLE_CACHED_INFERENCE` | `1` (on) | Reuse the train-side attention K/V across test chunks (KV cache); ~2-3x faster on large test sets that chunk. Set `0` to disable (or `SYNTHEFY_DISABLE_CACHED_INFERENCE=1`). |
| `SYNTHEFY_MAX_ELEMENTS_BUDGET` | VRAM-aware | Inference element budget; raise on large GPUs for full-context inference. Prefer `memory_policy={"elements_budget": N}`. |

> ⚠️ **`SYNTHEFY_CACHE_MAX_GB` has been removed** and now raises if set. It used to
> *skip* the KV cache above a fixed 6 GB; the cache is now **offloaded to host RAM**
> instead of skipped, so the old value does not translate. Use
> `memory_policy={"gpu_budget_frac": 0.4}` for a share of VRAM (portable across GPUs) or
> `memory_policy={"gpu_budget_absolute_gb": N}` for a hard cap on a shared GPU — see
> [Serving memory on large tables](#serving-memory-on-large-tables). Memory is
> configured through `memory_policy=` now, not environment variables; the only env vars left
> on this path are the kill switches above.

### Serving memory on large tables

Nori does in-context regression: your table is *input*, not weights. So one `predict`
call keeps a per-layer key/value cache over every context row, and that cache — not the
model — is what runs you out of GPU memory on a big table.

`memory_policy=` decides what to do about it. Omit it and you get the defaults, which handle
the common cases:

```python
from synthefy_nori import NoriRegressor, MemoryPolicy

NoriRegressor(model="nori-6m")                                  # default memory policy
NoriRegressor(model="nori-6m", memory_policy="exact")                   # never quantize
NoriRegressor(model="nori-6m", memory_policy="max_context")             # fit the biggest table
NoriRegressor(model="nori-6m", memory_policy="off")                     # no cache at all
NoriRegressor(model="nori-6m", memory_policy={"gpu_budget_frac": 0.25}) # e.g. from a config file
```

Under the hood it walks a ladder, using the cheapest rung that can serve the request:

| rung | what it does | exact? |
|---|---|---|
| `resident_bf16` | cache fits VRAM at full precision | **yes** |
| `resident_int8` | quantize to stay on the GPU instead of streaming | ~6e-6 R² |
| `offload_int8` | cache lives in host RAM, streamed per layer | quantized |
| `context_row_chunk` | after an OOM: also cap rows per build step | **yes** |
| `plain_loop` | no cache; several times slower, may drop context rows | **yes** |

**Only the int8 rungs cost accuracy, and `resident_int8` is only reached when full
precision would not fit.** A table that serves correctly today keeps bit-exact
predictions — accuracy is spent only to avoid a fallback that is slower or fatal. Use
`memory_policy="exact"` to forbid quantizing outright (it offloads instead).

Budgets are **fractions of your hardware**, so one setting travels from a laptop GPU to
an H200:

| field | default | meaning |
|---|---|---|
| `gpu_budget_frac` | `0.4` | share of total VRAM the resident cache may use |
| `host_budget_frac` | `0.25` | share of total RAM an offloaded cache may use |
| `gpu_budget_absolute_gb` / `host_budget_absolute_gb` | — | hard caps, for a shared GPU |
| `reuse_context_cache` | `True` | retain an unchanged encoded context across separate local `predict()` calls; set `False` to rebuild each call while keeping the normal within-request K/V cache |
| `cache_dtype` | `"bf16"` | precision the cache **starts** at |
| `allow_quantization` | `True` | may bf16 drop to int8 to stay resident? |
| `offload_to_host` | `True` | may the cache move to host RAM? |
| `context_row_chunk` | `None` | cap context rows per build step (auto after an OOM) |
| `elements_budget` | auto | per-forward element cap; drives chunking + subsampling |
| `allow_subsample` | `True` | may context rows be dropped to fit? `False` = raise |

> **Host offload needs RAM > 1.6 × VRAM at these defaults.** Offload only engages once
> the cache exceeds the GPU budget, and it can only succeed within the host budget — so
> `0.25 × RAM` has to exceed `0.4 × VRAM`. Below that ratio the offload rung is
> unreachable and a spilling request goes straight to `plain_loop`. Concretely, a 143 GB
> H200 needs more than ~229 GB of RAM; an 80 GB card needs more than ~128 GB. If your
> box is under the ratio, raise `host_budget_frac` (Nori warns, naming this, the first
> time a request actually needs the fallback and cannot get it).
>
> The default is deliberately conservative rather than higher: the fraction is of
> *total* RAM, your own input table is already resident in it, and overshooting host RAM
> gets the process OOM-killed by the kernel — unlike overshooting VRAM, which raises a
> catchable error and degrades to the next rung.

**Which rung ran** is on the estimator after `predict`:

```python
model.predict(X_test)
model.memory_report_
# {'rung': 'resident_bf16', 'cache_dtype': 'bf16', 'est_cache_gb': 30.5,
#  'gpu_budget_absolute_gb': 57.2, 'dropped_context_rows': 0, ...}
MemoryPolicy(**model.memory_report_).is_bit_exact      # -> True
```

Fallbacks are logged (`logging.getLogger("synthefy_nori.inference.predictor")`), and
dropping to `plain_loop` also emits a `RuntimeWarning` — it is the one rung that may
subsample your context, which otherwise looks like an unexplained accuracy loss.

**Incoherent settings fail instead of being ignored.** Asking for something that cannot
take effect raises rather than silently doing nothing:

```python
model = NoriRegressor(model="nori-6m", memory_policy={"cache": False, "context_row_chunk": 2048})
model.fit(X, y)
# ValidationError: ... 'row chunking without KV caching' is not a reachable
# configuration.  (the chunk caps the K/V *build*; with no cache there is no build)
```

The check runs in `fit`, not `__init__` — scikit-learn requires `__init__` to store
parameters verbatim so `clone` works — but it does run before any inference, so a bad
config fails in seconds rather than minutes into a job.

Settings that are merely redundant warn instead — e.g. setting both a fraction and an
absolute budget tells you the fraction is ignored, rather than refusing a layered
config. Unknown keys always raise.

Local fit-once/predict-many estimators reuse an unchanged encoded context by default. To
make a local process discard context-derived state after every prediction, use the typed
configuration `memory_policy={"reuse_context_cache": False}`. All shared serving targets
(Baseten, SageMaker/AWS Marketplace, Snowflake, and custom adapters using the shared
engine) enforce that value automatically, because one replica may handle multiple people
or workloads. This does **not** disable the K/V cache within a request; it only prevents
retaining it for a later request.

**Known limit:** at large row counts × many columns, the first thing to run out of
memory is the transductive preprocessing (RBF + polynomial expansion over the whole
table), *upstream* of the transformer. None of the above helps with that.

### Silent degradation

Some fallbacks keep inference alive by handing the model **less than the configured
pipeline promised**, then returning predictions anyway. In serving that is the right
trade — a slightly worse answer beats an exception. In anything *scored* it is the wrong
one: the run produces a plausible number that reads as "this config is weak" rather than
"the pipeline broke", and nothing downstream can tell the two apart.

So none of them are silent. Each warns under its own category, and escalating the
category is how you forbid it:

| warning | raised when | prevented outright by |
|---|---|---|
| `DegradedPipelineWarning` | base class — catch it to mean "any fidelity I did not ask for" | — |
| ↳ `SvdFallbackWarning` | the high-dimensional feature SVD failed, so the model got the **raw** unprojected columns (a `fit` failure) or a **single all-zero column** (a `transform` failure) | — |
| ↳ `ContextSubsampledWarning` | context rows were **dropped** to fit the element budget | `memory_policy={"allow_subsample": False}` |

```python
from synthefy_nori import NoriRegressor, SvdFallbackWarning, strict_pipeline

model = NoriRegressor(model="nori-6m")
model.fit(X_train, y_train)

model.predict(X_test)                     # serving: keep answering, warn if degraded

with strict_pipeline():                   # scored runs: a degraded pipeline raises
    model.predict(X_test)                 # instead of reporting a number

with strict_pipeline(SvdFallbackWarning):  # or just one fallback
    model.predict(X_test)
```

`strict_pipeline()` restores the previous filters on exit, so one strict prediction does
not harden the next one — safe inside a loop over datasets. It is a thin wrapper over
`warnings.simplefilter("error", DegradedPipelineWarning)`, so `-W error::...`,
`PYTHONWARNINGS`, and a `filterwarnings` line in `pytest.ini` work too.

Because the categories form a tree, escalation is inherited: a new fallback adds one
subclass and every caller who already asked for a strict pipeline gets it. **The eval
runner (`synthefy_nori.evaluation`) already wraps every scored predict call in
`strict_pipeline(SvdFallbackWarning)`** — a broken SVD is recorded as a failed row
rather than scored. It deliberately does not escalate `ContextSubsampledWarning`, since
trimming context to a budget is expected on large tables.

Only an SVD inference config can raise `SvdFallbackWarning` at all — the bundled default
and its lower-rank eval variant do; elsewhere the step is a no-op.

### Preprocessing speedups (on by default)

`SYNTHEFY_GPU_SVD`, `SYNTHEFY_CAP_QUANTILES`, and `SYNTHEFY_ADAPTIVE_FIT_SUBSAMPLE`
accelerate the inductive preprocessing pipeline (fit on train, apply to test) and
are enabled by default. They only act on the data shapes named above — most small tables (≤1000 rows,
<256 features) see little or no change. In an internal regression benchmark on a
single H200 they cut end-to-end wall-clock by roughly 1.8× with
mean R² unchanged (0.8087 → 0.8089). A large-scale A/B restricted to the tables
where they actually engage (n>5000) measured a mean ΔR² of +0.00002 (max |Δ|
0.0004) — within run-to-run noise.

### KV caching (on by default)

The cached prediction path is **enabled by default**. It projects the train-side
sequence-attention keys/values **once** and streams the test rows through the
layers reusing that cache, instead of recomputing the train K/V for every test
chunk — measured **~2-3x faster** on multi-chunk inference (the win scales with
the number of chunks). It only activates when the test set is large enough that
inference is already chunking (`n_test > chunk_size`), so it does not change the
chunking and therefore does not change R². We verified `cache == chunked` directly:
identical R², with per-prediction differences at mixed-precision scale (the two paths
reduce in a different order). When the cache will not fit VRAM it is **offloaded to
host RAM or quantized rather than skipped** — see
[Serving memory on large tables](#serving-memory-on-large-tables). Disable it with
`SYNTHEFY_ENABLE_CACHED_INFERENCE=0` or the `SYNTHEFY_DISABLE_CACHED_INFERENCE=1`
kill switch, or `memory_policy={"cache": False}`.

On the 1024-feature QSAR-TID-11 set (single H200), `predict` wall-clock vs.
test-set size with the cache OFF vs. ON shows the OFF time grows linearly
with the chunk count while the ON time stays roughly flat, reaching a **2.57×**
speedup on the full 1723-row test set (10.7 s vs. 27.4 s; ~1.3–1.9× at smaller
sizes). Predictions are effectively identical (max abs diff ~1.5e-3, attributable
to fp16 mixed precision — the cached path is mathematically equivalent). The cache
engages automatically whenever inference already chunks (`n_test > chunk_size`),
which happens readily on many-feature tables or large test sets (here forced via
`SYNTHEFY_MAX_ELEMENTS_BUDGET=1050000`, driving `chunk_size` to its 256-row floor →
7 chunks).

Reproduce (prints the results table and writes `benchmarks/plots/kv_cache_speed.png`):
`uv run python benchmarks/bench_kv_cache.py`

```bash
# All speedups (preprocessing + KV cache) are on by default — nothing to enable.

# To disable them all (e.g. for exact reproducibility / debugging):
SYNTHEFY_GPU_SVD=0 SYNTHEFY_CAP_QUANTILES=0 SYNTHEFY_ADAPTIVE_FIT_SUBSAMPLE=0 \
SYNTHEFY_ENABLE_CACHED_INFERENCE=0 \
python your_inference_script.py
```

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
`--seed` for reproducible runs.

### Training acceleration

The bundled launchers enable native RMSNorm, foreach EMA updates, and regional
dynamic `torch.compile` by default. These preserve the natural shape curriculum;
disable them per run with `NATIVE_RMS_NORM=0`, `EMA_FOREACH=0`, or
`COMPILE_ENCODER_LAYERS=none`. Exact static-shape compilation is also available,
but requires an explicit shape palette that changes the curriculum and must be
validated separately.

See [docs/training.md](docs/training.md) for the full training options and
[the acceleration guide](docs/training_static_compile.md) for compiler modes,
cache setup, measurements, and reproducibility controls.

## Evaluation

```bash
synthefy-nori-eval --checkpoint "Synthefy:path/to/checkpoint.pt"
```

or `bash scripts/evaluate.sh`. See [docs/evaluation.md](docs/evaluation.md) for
benchmark sources and how to evaluate a Nori checkpoint, and
[Reproducing these numbers](#reproducing-these-numbers) for the published
benchmark run.

## Hugging Face

```bash
synthefy-nori-download                                            # fetch default checkpoint
synthefy-nori-upload path/to/checkpoint.pt --repo-id Synthefy/Nori
```

See [docs/huggingface.md](docs/huggingface.md).

## Repository layout

```
libs/synthefy/             Lightweight `synthefy` distribution
  src/synthefy/
    nori_client.py         Explicit remote, SageMaker, and local gateway
    nori_ts/               Backend-neutral forecasting workflow and preparation
src/synthefy_nori/         Heavy `synthefy-nori` distribution
  api.py                   Local NoriRegressor, infer, and predict
  model/                   FeaturesTransformer architecture
  training/                Data generation, trainer, loss, config, CLI
  inference/               Local predictor and preprocessing
  evaluation/              Benchmark runner over public benchmark suites
  hf.py                    Hugging Face download / upload
serving/                   Shared serving core and host-specific packaging
scripts/                   Training, precompile, and evaluation launchers
benchmarks/                Reproducible performance and compiler harnesses
docs/                      Training, inference, evaluation, and release design
examples/                  Runnable local inference and upload scripts
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
  version      = {0.6.0},
  doi          = {10.5281/zenodo.20710462},
  url          = {https://doi.org/10.5281/zenodo.20710462},
}
```

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
