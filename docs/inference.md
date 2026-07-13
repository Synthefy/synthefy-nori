# Inference

Use the public wrappers:

```python
from synthefy_nori import NoriRegressor

reg = NoriRegressor(model_path="checkpoints/best_reg_r2.pt")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

If `model_path` is omitted, the default checkpoint is resolved from Hugging
Face through `synthefy_nori.hf.download_checkpoint()`.

## Output types

`predict` selects what it returns via `output_type`:

| `output_type` | Returns | Shape |
| --- | --- | --- |
| `"mean"` (default) | distribution mean | `(n_samples,)` |
| `"median"` | distribution median (τ=0.5) | `(n_samples,)` |
| `"mode"` | distribution mode | `(n_samples,)` |
| `"quantiles"` | quantiles at `quantiles=[...]` levels | `(n_levels, n_samples)` |
| `"full"` | dict `{"quantiles", "taus", "mean"}` | quantiles `(n_samples, K)` |

```python
# Prediction intervals from quantiles
lo, mid, hi = reg.predict(X_test, output_type="quantiles", quantiles=[0.05, 0.5, 0.95])

# Full predictive distribution (e.g. for CRPS / calibration)
dist = reg.predict(X_test, output_type="full")
Q, taus = dist["quantiles"], dist["taus"]   # Q: (n_samples, 999), ascending per row
```

The default checkpoint exposes a 999-quantile pinball head; quantiles come back
in original-`y` units, sorted per row. `"quantiles"`/`"full"` require the
pinball checkpoint — a `bar_distribution` checkpoint raises `NotImplementedError`.

## Categorical / ordinal targets (`categorical_target=True`)

When the target only takes a small set of discrete values (ratings, counts,
quality scores), declare it and predictions are mapped onto the level lattice
observed in `fit`'s `y`:

```python
reg = NoriRegressor().fit(X_train, y_train)          # y ∈ {3, 4, 5, 6, 7, 8}
labels = reg.predict(X_test, categorical_target=True)               # map-cell
labels = reg.predict(X_test, categorical_target=True,
                     discretize="median-cell")                      # MAE-optimal
```

`discretize` picks the lattice summary; **choose it by the metric you are
scored on** (benchmarked on the K≤10 discrete-target datasets of
TALENT / OpenML-CTR23 / TabArena):

| `discretize` | What it is | Optimal for | Needs quantile bank |
| --- | --- | --- | --- |
| `"map-cell"` (default) | mode of the discrete posterior: integrate the predictive distribution's mass in a cell around each level, take the argmax | accuracy, macro-F1 | yes |
| `"median-cell"` | median of that discrete posterior (cumulative cell mass crosses 0.5) | MAE / ordinal closeness | yes |
| `"snap-mean"` | nearest level to the point mean | quadratically-penalized agreement (QWK) | no |
| `"snap-median"` | nearest level to the distribution median | MAE (bank-free fallback) | no |

Notes:

- **Discretization is strictly opt-in.** Nothing is snapped unless you pass
  `categorical_target=True`. (The predictor's legacy auto-snap for low-K
  targets, `discrete_y_snap_max_unique`, is now **off by default**; re-enable
  it explicitly via `NoriRegressor(discrete_y_snap_max_unique=30)` if you want
  the old always-snap behavior without the flag.)
- **If the task is scored by squared error / R², do not discretize** — the
  continuous mean is optimal and any lattice projection trades R² away
  (~0.05–0.14 R² on the benchmark). `categorical_target=True` is for when you
  need *labels*, not a regression score.
- The gains are largest on **skewed** discrete targets, where the mean falls
  between levels or on a low-probability one; map-cell recovered +48pp
  accuracy over snap-mean on the most skewed benchmark dataset.
- By default levels come from the training `y` (leak-safe). If the context is
  small and may under-cover the true lattice (e.g. a 1–5 scale whose context
  has no 1s), pass the full known level set explicitly:
  `predict(X, categorical_target=True, categorical_levels=[1, 2, 3, 4, 5])`.
- Failed predictions stay visible: a `NaN` point prediction stays `NaN` after
  snapping rather than becoming a confident label.
- `"map-cell"`/`"median-cell"` read the quantile bank, so they share the
  pinball-checkpoint requirement above; the `snap-*` strategies work with any
  checkpoint (`snap-mean` always snaps the *mean*, independent of the
  configured `quantile_collapse`).
- The one-shot helpers accept the same arguments:
  `infer(X_train, y_train, X_test, categorical_target=True)`.
- All three are also **estimator parameters**, so the sklearn ecosystem can
  reach them — `predict` kwargs override per call:

  ```python
  gs = GridSearchCV(NoriRegressor(categorical_target=True),
                    {"discretize": ["map-cell", "median-cell", "snap-mean"]},
                    scoring="accuracy", cv=5)
  ```

The underlying lattice math is importable from `synthefy_nori.discretize`
(`cell_masses`, `discretize_predictions`, `snap_to_levels`).
