# Synthefy Tabular

Synthefy Tabular is a tabular foundation model package for regression and
classification. This repository contains the cleaned public training,
inference, evaluation, and Hugging Face checkpoint tooling.

## Install

```bash
uv sync --extra dev
```

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
