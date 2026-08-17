---
name: nori-regression
description: >-
  General tabular regression with Nori, the Synthefy tabular foundation model —
  sklearn-style in-context fit/predict with zero training, native prediction
  intervals (quantiles / full predictive distribution), honest baseline
  comparison under a fixed CV protocol, and SHAP / PDP / feature-selection
  interpretability — plus leak-safe one-step time-series forecasting via lag
  features and a rolling-origin backtest. Use when predicting a continuous
  target from tabular features, when you need uncertainty bands without extra
  machinery, when forecasting a single series one step ahead, or when the
  dataset is too small to train a model on.
---

# Tabular regression with Nori

## What this is and when to use it

A playbook for regression with **Nori** (`synthefy-nori` on PyPI), a ~6M-param
tabular foundation model that predicts by **in-context learning**: `fit()` just
stores your training rows as context, and `predict()` conditions the frozen
model on them. No training loop, no hyperparameters to tune, and the full
predictive distribution comes for free.

Reach for it when:

- you have a **tabular regression** problem (continuous target, feature matrix);
- the dataset is **small-to-mid sized** (tens to a few thousand context rows) —
  exactly where training a model from scratch overfits;
- you want **prediction intervals** without conformal/quantile-regression add-ons;
- you want a strong baseline in ten lines before investing in anything bespoke.

Not a fit: classification (Nori is regression-only), very large datasets
(predict cost grows with context size — subsample the context first), and
unstructured inputs (text/images) unless you featurize them yourself.

> **Time series:** Nori has no time axis, but one-step-ahead forecasting works
> well — frame the series as tabular rows where features at time *t* are built
> only from data *before t* (lags, rolling stats), and evaluate with a
> rolling-origin backtest, never a random split. Full recipe:
> `references/one-step-forecasting.md` + `templates/forecast_one_step.py`.
> For *panels* of related series (SKUs, stores, drugs), use the
> **nori-demand-forecasting** skill instead.

## Mental model (read this first)

- `fit(X, y)` does **no compute** — it stores `X` (float32) and `y` (float64)
  as the context set.
- `predict(X_query)` runs the frozen foundation model, attending from each
  query row to the whole context. All cost is here, and it scales with
  context × query size.
- The default checkpoint has a **999-quantile pinball head**, so beyond point
  predictions (`"mean"` / `"median"` / `"mode"`) you get calibrated-ish
  quantiles (`output_type="quantiles"`) and the whole distribution
  (`output_type="full"`) at no extra cost.

The exact, verified API is in `references/api.md` — read it before writing code.

## Quickstart

```python
from synthefy_nori import NoriRegressor

reg = NoriRegressor(device=None, model="nori-30m")          # None -> cuda:0 if available, else CPU
reg.fit(X_train, y_train)                 # stores context; no training happens
point = reg.predict(X_test)               # (n,) distribution mean
lo, mid, hi = reg.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
```

Install: `pip install synthefy-nori` (add `[interpretability]` for SHAP/PDP).
First `predict` downloads the checkpoint from Hugging Face (`Synthefy/Nori`).

## Workflow

Templates live in `templates/` — copy them into your project and parametrize
via their CLI flags; don't hardcode paths.

1. **Prepare the data.** Numeric matrix in, finite target out: encode
   categoricals (ordinal/one-hot), leave NaN features as NaN (missing is a
   signal Nori accepts), don't bother scaling features. →
   `references/data-preparation.md`
2. **Fit, predict, pick the point estimate.** `"mean"` for symmetric targets,
   `"median"` when the target is skewed or has outliers. →
   `templates/regress.py`, `references/api.md`
3. **Add prediction intervals.** `output_type="quantiles"` for e.g. P10/P50/P90;
   check the empirical coverage of the interval on a held-out split before
   trusting it. → `templates/regress.py`
4. **Compare against baselines honestly.** Same folds, same metric, decided up
   front: `cross_validate` Nori vs Ridge and RandomForest on one shared
   `KFold`. Nori is a scikit-learn estimator, so it drops straight in. →
   `templates/compare_baselines.py`, `references/evaluation.md`
5. **Explain the predictions.** Shapley values (shapiq), partial dependence,
   and sequential feature selection through
   `synthefy_nori.interpretability`. → `templates/explain.py`,
   `references/interpretability.md`
6. **Forecasting a time series?** Build leak-safe lag/rolling/seasonal
   features, refit at every origin (fit is free), and score against last-value
   and seasonal-naive baselines over a rolling backtest — never a random
   split. Need h periods ahead? Use the direct recipe — newest legal lag is
   `lag_h`. → `templates/forecast_one_step.py`,
   `templates/forecast_multi_step.py`, `references/one-step-forecasting.md`
7. **Tune the practical knobs only if needed.** `device` for GPU, `model_path`
   for a local/pinned checkpoint, `HF_HUB_OFFLINE=1` for air-gapped runs,
   context subsampling for big datasets. → `references/api.md`,
   `references/data-preparation.md`

## Guardrails

- **Use only the documented public API.** The constructor and `predict`
  signatures in `references/api.md` are exact — do not invent kwargs, env vars,
  or telemetry flags. Unknown constructor kwargs raise `TypeError`.
- **`quantiles=` is only valid with `output_type="quantiles"`** (else
  `ValueError`); every τ must be strictly inside (0, 1).
- **Fix the evaluation before comparing models.** Choose the split/folds,
  metric, and scope up front and keep them constant across every model you
  try. Iterating the model is science; iterating the eval until you win is
  cheating. → `references/evaluation.md`
- **The target must be finite.** NaN in `y` silently corrupts the stored
  normalization stats — drop or impute target-missing rows before `fit`.
- **No hardcoded paths.** Parametrize scripts with argparse/env and resolve
  relative to `Path(__file__)`. Keep data and checkpoints out of git.
- **Don't ship the checkpoint.** It downloads from the Hugging Face Hub on
  first use and caches under `~/.cache/huggingface/hub`; pass `model_path=`
  only to point at a checkpoint you already have locally.
