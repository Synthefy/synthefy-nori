# Synthefy Tabular

Synthefy Tabular is a tabular foundation model package for regression and
classification. This repository contains the cleaned public training,
inference, evaluation, and Hugging Face checkpoint tooling.

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

```python
from synthefy_tabular import SynthefyTabularRegressor

model = SynthefyTabularRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

If `model_path` is omitted, the default checkpoint is downloaded from the
Hugging Face Hub. Local checkpoint paths also work:

```python
model = SynthefyTabularRegressor(model_path="checkpoints/best_reg_r2.pt")
```

## Training

```bash
TOTAL_STEPS=2 NPROC_PER_NODE=1 WANDB_MODE=disabled bash scripts/train.sh
```

For the large-table continuation stage:

```bash
RUN_ROOT=checkpoints/synthefy-tabular-train-YYYYMMDD-HHMMSS bash scripts/continue_training.sh
```

## Evaluation

```bash
synthefy-tabular-eval --checkpoint "Synthefy:checkpoints/best_reg_r2.pt"
```

## Hugging Face

```bash
synthefy-tabular-download
synthefy-tabular-upload checkpoints/best_reg_r2.pt --repo-id Synthefy/synthefy-tabular
```
