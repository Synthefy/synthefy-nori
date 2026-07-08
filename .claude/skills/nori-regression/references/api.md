# Nori public API reference

Verified against `synthefy-nori` 0.9.0 (`src/synthefy_nori/api.py`). This is
the complete public regression surface — anything not listed here is not part
of the public API.

## Install

```bash
pip install synthefy-nori                     # core fit/predict, incl. quantiles
pip install "synthefy-nori[interpretability]" # + shapiq & matplotlib for SHAP/PDP
```

Python ≥ 3.9. Core deps: numpy≥2, pandas≥2, scikit-learn≥1.4, scipy≥1.13,
torch≥2.8, huggingface-hub. `NoriRegressor.fit/predict` — including quantile
output — need **no extra**; `interpretability` is only for SHAP/PDP/feature
selection.

## Construct → fit → predict

```python
from synthefy_nori import NoriRegressor

reg = NoriRegressor(device=None)   # None -> cuda:0 if available, else cpu
reg.fit(X_train, y_train)          # stores context; X -> float32 (n, d), y -> float64 (n,)
point = reg.predict(X_test)        # (n,) distribution mean
```

**Constructor (exact signature):**

```python
NoriRegressor(model_path=None, *, device=None, inference_config=None, token=None,
              augmentations=("yj",), yj_skew_threshold=10.0,
              quantile_collapse="mean", bar_temperature=1.0,
              bar_point_estimator="mean")
```

- `model_path` — local checkpoint path; `None` downloads the default from the
  Hugging Face Hub on first `predict`.
- `device` — `None` → `cuda:0` if available else CPU; or any torch device string.
- `inference_config` — path to a bundled/custom inference config JSON; default
  is `reg_allordinal_poly10_adaptive_svd256.json`. Use
  `synthefy_nori.config_path(filename)` to resolve a bundled one.
- `token` — Hugging Face token, only needed for gated/private checkpoints (the
  default public checkpoint is ungated).
- `augmentations=("yj",)` — input augmentation; `"yj"` applies a Yeo-Johnson
  transform to skewed targets (see `yj_skew_threshold`). Pass `()` to disable.
- `quantile_collapse`, `bar_temperature`, `bar_point_estimator` — advanced
  head/aggregation knobs; leave at defaults unless you know why.

`NoriRegressor` subclasses `RegressorMixin, BaseEstimator`, so `clone`,
`get_params`/`set_params`, `score` (R²), `cross_validate`, and the sklearn
interpretability ecosystem all work directly.

`fit(X, y)` accepts numpy arrays, DataFrames, or lists; returns `self`.
Calling `predict` before `fit` raises `ValueError`.

## `predict` output types

```python
reg.predict(X, *, output_type="mean", quantiles=None)
```

| `output_type` | Returns | Shape | Use for |
|---|---|---|---|
| `"mean"` (default) | distribution mean | `(n,)` | symmetric targets |
| `"median"` | median (τ=0.5) | `(n,)` | skewed targets / outliers |
| `"mode"` | distribution mode | `(n,)` | rarely needed; prefer `"median"` |
| `"quantiles"` | quantiles at `quantiles=[...]` | `(len(taus), n)` | prediction intervals |
| `"full"` | `{"quantiles", "taus", "mean"}` | dict | whole predictive distribution |

```python
# 80% prediction interval + median in one call — note the (3, n) shape
lo, mid, hi = reg.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])

# Full distribution (e.g. for CRPS / calibration curves)
dist = reg.predict(X_test, output_type="full")
Q, taus = dist["quantiles"], dist["taus"]   # Q: (n, 999) ascending per row
```

Rules (all raise informative errors):

- `quantiles=` is **only** valid with `output_type="quantiles"` → else `ValueError`.
- Each τ must be strictly in (0, 1) → else `ValueError`.
- `output_type="main"` → `NotImplementedError`; unknown strings → `ValueError`.
- Returned quantiles are sorted ascending per row (monotone).
- `"quantiles"`/`"full"` need the pinball (quantile-head) checkpoint — the
  default ships one; a `bar_distribution` checkpoint raises `NotImplementedError`.

## One-shot functional API

```python
from synthefy_nori import infer
y_pred = infer(X_train, y_train, X_test)   # mean point predictions only
```

`predict(...)` (module-level) is an alias of `infer`.

## Checkpoint & environment

- Default checkpoint: HF repo `Synthefy/Nori` (override env
  `SYNTHEFY_NORI_HF_REPO`), file `nori.pt` (env `SYNTHEFY_NORI_HF_FILENAME`),
  cached under `~/.cache/huggingface/hub`. The public checkpoint is ungated.
- **Offline:** set `HF_HUB_OFFLINE=1` (a huggingface-hub feature) to serve from
  cache — run once online first, or pass `model_path=` to a local file.
- **Device:** GPU optional; CPU works fine for small/mid contexts. `fit()`
  never touches the device — all compute is in `predict`.
- There are **no** Nori-specific telemetry or feature-flag env vars — don't
  invent any.
