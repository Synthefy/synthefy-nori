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

## DataFrame features: numeric, categorical, and text

All local public paths share one fitted DataFrame schema:

```python
reg = NoriRegressor(
    model="nori-30m",
    categorical_columns=["plan", "region"],
    text_columns=["ticket_description"],
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

The estimator API is `fit(X_train, y_train)` followed by `predict(X_test)`.
The one-shot helper is separately `predict(X_train, y_train, X_test, ...)` and
accepts the same feature-preprocessing arguments.

| `categorical_columns` | Behavior for DataFrames |
| --- | --- |
| `"auto"` (default) | Encode every remaining non-numeric, non-text column. |
| sequence of names | Encode exactly those columns; other non-numeric columns raise with their names and dtypes. |
| `None` | Disable categorical inference; remaining columns must be numeric. |

Text and categorical declarations cannot overlap. Column names are learned at
`fit`, and query frames are reordered to that schema; missing or extra query
columns raise directly. Category mappings and text SVD are learned only from
training rows. Ordinal encoding assigns deterministic `0..K-1` codes, preserves
missing values as `NaN`, and maps rare or unseen values to the bounded `K`
`other` code. `categorical_encoding="onehot"` remains available for compatibility.

An automatically inferred string column above
`max_categorical_cardinality=100` is ambiguous—it may be an ID, free text, or a
real high-cardinality categorical—so Nori asks you to declare it rather than
dropping or embedding it. Explicit categoricals retain their top K training
levels and collapse the rest to `other`. Datetime, timedelta, and period columns
must be converted explicitly. Positional lists/arrays remain numeric-only.

Categorical **features** are declared with `categorical_columns`. The
`categorical_levels` option in the target section below applies only to numeric
values of `y` and does not preprocess `X`.

## Execution defaults and exact reproducibility

Standard inference loads checkpoints with PyTorch's native RMSNorm kernel and
skips the feature decoder when its output cannot be observed. Decoder skipping
is output-identical. Native RMSNorm follows the same equation but can move a
mixed-precision result by one bf16 unit in the last place; controlled checks
measured an R² shift no larger than `2e-5` and about a 1.3% end-to-end inference
improvement, with preprocessing dominating total time.

`NoriRegressor` uses these defaults automatically and does not add a new public
switch. Advanced callers that need the exact historical execution path can pass
`native_rms_norm=False` to lower-level `load_model()` or `NoriPredictor`, and
`skip_unused_feature_decoder=False` to `NoriPredictor`. These controls change
execution only; they do not select different weights.

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

Nori predicts in context, so your table is *input*: every call reads all of
`X_train` and keeps a per-layer key/value cache over those rows. That cache — not
the ~6M parameters — is what fills a GPU on a big table.

Omit `memory_policy` and nothing changes. The defaults cache at full precision,
drop to int8 only if that is what keeps the cache on the GPU, spend at most 40% of
VRAM on it, offload to host RAM rather than give up, and shrink the context only as
a last resort.

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
| `plain_loop` | no cache; the context is re-read per batch | exact, much slower |
| `no_cache` | the cached path did not apply | exact |

Only the int8 rungs trade accuracy, and they are reached only when full precision
will not fit; offloading moves bytes rather than approximating. The cache is built
only when the query set spans more than one batch, so a small `X_test` reports
`no_cache` — that is normal, not a degradation.

`memory_report_["dropped_context_rows"]` is the one number here that is a real
accuracy loss rather than a rounding-level effect: it counts context rows discarded
to fit. Set `allow_subsample=False` to make that case an error instead of a silent
shrink.

Field-by-field reference, the budget knobs, and the measured saving per lever:
[README, "Serving memory on large tables"](../README.md#serving-memory-on-large-tables).

## Multi-target regression

Pass a two-dimensional target matrix with at least two columns to use one
`NoriRegressor` for a joint prediction. The default `"copula"` strategy fits
cross-validated probability-integral-transform values and an order-invariant
vine copula over Nori's conditional marginals. Copula support is included in
the standard `synthefy-nori` install.

```python
from synthefy_nori import MultiTargetPredictionPolicy, NoriRegressor

regressor = NoriRegressor(model="nori-6m").fit(X_train, Y_train)
mean = regressor.predict(X_test)  # (n_test, n_targets)
samples = regressor.predict(
    X_test,
    output_type="samples",
    multi_target_prediction_policy=MultiTargetPredictionPolicy(
        n_draws=1_000,
        random_state=42,
    ),
)  # (n_test, 1_000, n_targets)
```

Set `multi_target_prediction_strategy="independent"` for the lowest-cost
product of marginals, or `"autoregressive"` for the strongest measured joint
accuracy at draw- and target-order-scaled inference cost. Scalar targets retain
their existing behavior. The first
release supports joint means and samples; marginal quantiles remain out of scope.

For reproducible autoregressive experiments or domain-informed factorization,
provide complete target-index permutations. Orders control the predictive
factorization and do not imply causality:

```python
policy = MultiTargetPredictionPolicy(
    autoregressive_orders=[[0, 1, 2], [2, 0, 1]],
)
regressor = NoriRegressor(
    model="nori-6m",
    multi_target_prediction_strategy="autoregressive",
    multi_target_prediction_policy=policy,
).fit(X_train, Y_train)
assert regressor.target_orders_ == [(0, 1, 2), (2, 0, 1)]
```

Omitting `autoregressive_orders` generates deterministic unique permutations;
the requested count is capped at `n_targets!` rather than repeating an order.

The lightweight client sends the same operation to a hosted base-model deployment
in one API call:

```python
from synthefy import MultiTargetPredictionPolicy, SynthefyNoriClient

client = SynthefyNoriClient(model="nori-6m")
samples = client.predict(
    X_train,
    Y_train,
    X_test,
    output_type="samples",
    multi_target_prediction_strategy="copula",
    multi_target_prediction_policy=MultiTargetPredictionPolicy(
        n_draws=300,
        random_state=42,
    ),
)
```

Hosted responses cap targets, draws, and total returned sample cells. Thinking and
large-context policies do not yet compose with matrix-valued targets. The server
echoes the honored strategy so the client fails closed against an older deployment.
For autoregressive calls, `client.last_target_orders` records the resolved orders.
When a hosted matrix-target request sets `memory_policy`,
`client.last_multi_target_memory_reports` records one resolved report per internal
marginal or chain call; scalar calls continue to use `client.last_memory_report`.

## Categorical / ordinal targets (`discretize=` / `categorical_levels=`)

When the target only takes a small set of discrete values (ratings, counts,
quality scores), declare it and predictions are mapped onto the level lattice
observed in `fit`'s `y`:

```python
reg = NoriRegressor(model="nori-30m").fit(X_train, y_train)          # y ∈ {3, 4, 5, 6, 7, 8}
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
  gs = GridSearchCV(NoriRegressor(model="nori-30m"),
                    {"discretize": ["map-cell", "median-cell", "snap-mean"]},
                    scoring="accuracy", cv=5)
  ```

The underlying lattice math is importable from `synthefy_nori.discretize`
(`cell_masses`, `discretize_predictions`, `snap_to_levels`).

## Silent degradation (`strict_pipeline`)

Some fallbacks trade fidelity for staying alive and still return a prediction. None of
them are silent — each warns under its own category, and escalating the category forbids
it:

| warning | raised when | also prevented by |
|---|---|---|
| `DegradedPipelineWarning` | base class, i.e. any of the below | — |
| `SvdFallbackWarning` | the high-dimensional SVD failed: raw unprojected columns (`fit`) or one all-zero column (`transform`) | — |
| `ContextSubsampledWarning` | context rows dropped to fit the element budget | `memory_policy={"allow_subsample": False}` |

```python
from synthefy_nori import strict_pipeline

with strict_pipeline():          # evals / benchmarks: refuse to report a degraded run
    y_pred = reg.predict(X_test)
```

Filters are restored on exit, so this is safe in a loop. Equivalent to
`warnings.simplefilter("error", DegradedPipelineWarning)`, so `-W`, `PYTHONWARNINGS` and
pytest's `filterwarnings` work as well. `synthefy_nori.evaluation`'s runner already
wraps every scored predict call in `strict_pipeline(SvdFallbackWarning)`. Full rationale:
README, "Silent degradation".
