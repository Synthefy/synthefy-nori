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

## Large tables and memory (`memory_policy=`)

Nori predicts in context, so the table is *input*: every call reads all of
`X_train` and keeps a per-layer key/value cache over those rows. That cache — not
the ~6M parameters — is what fills a GPU on a big table.

Omit `memory_policy` and nothing changes. The defaults cache at full precision,
drop to int8 only if that is what keeps it on the GPU, spend at most 40% of VRAM
on it, offload to host RAM rather than give up, and shrink the context only as a
last resort.

```python
from synthefy_nori import MemoryPolicy, NoriRegressor

reg = NoriRegressor(model="nori-6m", memory_policy="exact")        # never trade accuracy
reg = NoriRegressor(model="nori-6m", memory_policy="max_context")  # fit the largest table
reg = NoriRegressor(model="nori-6m", memory_policy="off")          # no cache at all
reg = NoriRegressor(model="nori-6m", memory_policy=MemoryPolicy(cache_dtype="int8"))

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(reg.memory_report_["rung"])          # which fallback actually ran
```

When the cache does not fit, inference takes the cheapest step that works:

| Rung | What happens | Accuracy |
| --- | --- | --- |
| `resident_bf16` | full-precision cache in GPU memory | exact |
| `resident_int8` | quantized so it stays resident | ~1.9x smaller, \|ΔR²\| ≈ 6e-6 |
| `offload_bf16` | full precision in host RAM, streamed per layer | exact, slower |
| `offload_int8` | quantized *and* in host RAM | as int8 |
| `context_row_chunk` | cache built in row chunks after an OOM retry | not bit-identical |
| `plain_loop` | no cache; context re-read per batch | exact, much slower |
| `no_cache` | the cached path did not apply | exact |

Only the int8 rungs trade accuracy, and they are reached only when full precision
will not fit; offloading moves bytes rather than approximating. The cache is built
only when the query set spans more than one batch, so a small `X_test` reports
`no_cache` — that is normal, not a degradation.

`memory_report_` also carries `dropped_context_rows`, the one number here that is
a real accuracy loss rather than a rounding-level effect. Set
`allow_subsample=False` to make that case an error instead.

Field-by-field reference, the budget knobs, and the measured savings per lever:
[README, "Serving memory on large tables"](../README.md#serving-memory-on-large-tables).

## Categorical / ordinal targets (`discretize=` / `categorical_levels=`)

When the target only takes a small set of discrete values (ratings, counts,
quality scores), declare it and predictions are mapped onto the level lattice
observed in `fit`'s `y`:

```python
reg = NoriRegressor().fit(X_train, y_train)          # y ∈ {3, 4, 5, 6, 7, 8}
labels = reg.predict(X_test, discretize="map-cell")                 # accuracy-optimal
labels = reg.predict(X_test, discretize="median-cell")              # MAE-optimal
labels = reg.predict(X_test, categorical_levels=[3, 4, 5, 6, 7, 8]) # known lattice, map-cell
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
| `"expected-level"` | Σ level·P(level) — lattice-informed **continuous** expectation (not on-lattice) | analysis / sanity (≈ plain mean) | yes |
| `"prior-match"` | rank rows by point prediction, assign labels to match training priors | calibration experiments — **benchmarked worse** than snap-mean on accuracy (p=0.048) and MAE (p=0.027); prefer map-cell/median-cell | no |

Notes:

- **Discretization is strictly opt-in.** Nothing is snapped unless you ask:
  passing `discretize=` (a strategy) or `categorical_levels=` (a known
  lattice) is the activation — there is no separate flag. `categorical_levels`
  alone uses the default strategy (`DEFAULT_DISCRETIZE_METHOD`, map-cell).
  (The predictor's legacy auto-snap for low-K targets,
  `discrete_y_snap_max_unique`, is now **off by default**; re-enable it via
  `NoriRegressor(discrete_y_snap_max_unique=30)` if you want the old
  always-snap behavior.)
- **If the task is scored by squared error / R², do not discretize** — the
  continuous mean is optimal and any lattice projection trades R² away
  (~0.05–0.14 R² on the benchmark). `discretize=` is for when you need
  *labels*, not a regression score.
- The gains are largest on **skewed** discrete targets, where the mean falls
  between levels or on a low-probability one; map-cell recovered +48pp
  accuracy over snap-mean on the most skewed benchmark dataset.
- `categorical_levels` is the set of values the target can take — the label
  set, in classification terms. It is named *levels* (ordinal terminology)
  rather than *labels* because the values must be numeric and their **order
  matters**: the `map-cell`/`median-cell` strategies build probability cells
  from the midpoints between adjacent values; string/unordered class labels
  are unsupported. Every predicted label is one of these levels. By default
  they come from the training `y` (leak-safe); if the context is small and
  may under-cover the true lattice (e.g. a 1–5 scale whose context has no
  1s), pass the full known set explicitly:
  `predict(X, categorical_levels=[1, 2, 3, 4, 5])`.
- Failed predictions stay visible: a `NaN` point prediction stays `NaN` after
  snapping rather than becoming a confident label.
- `"map-cell"`/`"median-cell"` read the quantile bank, so they share the
  pinball-checkpoint requirement above; the `snap-*` strategies work with any
  checkpoint (`snap-mean` always snaps the *mean*, independent of the
  configured `quantile_collapse`).
- The one-shot helpers accept the same arguments:
  `infer(X_train, y_train, X_test, discretize="map-cell")`.
- Both are also **estimator parameters**, so the sklearn ecosystem can
  reach them — `predict` kwargs override per call:

  ```python
  gs = GridSearchCV(NoriRegressor(),
                    {"discretize": ["map-cell", "median-cell", "snap-mean"]},
                    scoring="accuracy", cv=5)
  ```

The underlying lattice math is importable from `synthefy_nori.discretize`
(`cell_masses`, `discretize_predictions`, `snap_to_levels`).
