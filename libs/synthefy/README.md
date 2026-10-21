# Synthefy Python Client

`synthefy` is the lightweight SDK for Synthefy Nori. It owns the
`SynthefyNoriClient`, shared feature preparation, and backend-neutral
forecasting workflows. Hosted users do not install Torch or model weights;
local execution is supplied by the separate `synthefy-nori` distribution.

The retired ForecastV2 API and `/v2/forecast` endpoint are not part of
`synthefy` 7. Use `SynthefyNoriClient` directly for regression or
`synthefy.nori_ts.NoriTSForecaster` for Nori-backed forecasting.

## Features

- **One regression client**: the same `SynthefyNoriClient` contract runs
  through hosted Baseten, a named SageMaker endpoint, or the local runtime.
- **Nori forecasting**: optional feature preparation and result
  reconstruction use that same regression gateway.
- **Prediction intervals**: quantiles and full predictive distributions come
  from the same forward pass.
- **DataFrame, categorical, and text preparation**: shared preprocessing
  keeps local and hosted numeric requests aligned.
- **Typed errors and requests**: Pydantic request models and one HTTP error
  hierarchy across remote transports.

## Installation

Hosted regression:

```bash
pip install synthefy
```

Local regression:

```bash
pip install synthefy-nori
```

Forecasting:

```bash
pip install "synthefy[forecasting]"       # hosted or SageMaker
pip install "synthefy-nori[forecasting]"  # local runtime
```

Optional text or SageMaker support:

```bash
pip install "synthefy[text]"
pip install "synthefy[aws]"
```

## Nori — DataFrame Forecasting

`NoriTSForecaster` turns time-series DataFrames into regression requests and
runs every request through a configured `SynthefyNoriClient`. That keeps the
forecasting workflow identical across hosted Baseten, SageMaker, and local
execution.

Use `future_df=` when the forecast horizon includes values known in advance.
The target may use a domain-specific name such as `sales`:

```python
import os

import pandas as pd

from synthefy import SynthefyNoriClient
from synthefy.nori_ts import NoriTSForecaster

history = pd.DataFrame({
    "timestamp": pd.date_range("2026-01-01", periods=48, freq="h"),
    "sales": [100.0 + hour for hour in range(48)],
    "promotion": [0.0] * 48,
})
future = pd.DataFrame({
    "timestamp": pd.date_range("2026-01-03", periods=12, freq="h"),
    "promotion": [0.0] * 6 + [1.0] * 6,
})

client = SynthefyNoriClient(
    mode="remote",
    model="nori-6m",
    api_key=os.environ["SYNTHEFY_NORI_API_KEY"],
)
forecaster = NoriTSForecaster(
    client=client,
    quantiles=[0.1, 0.5, 0.9],
)
forecast = forecaster.predict_df(
    history,
    future_df=future,
    target_column="sales",
)
```

`future_df` must contain future timestamps and every numeric covariate used in
history. Its target must be absent or entirely missing; observed future targets
would leak the answer. When there are no future covariates, pass
`prediction_length=` instead and the forecaster generates the horizon.
`target_column` accepts one column name per call; multiple target columns are
not supported yet.

## Nori — Tabular In-Context Regression

`SynthefyNoriClient` is the lightweight client for **Synthefy Nori**, an
in-context learning regressor. Each call supplies labeled context rows
(`X_train`, `y_train`) and query rows (`X_test`); the model returns one predicted
value per query row in a single forward pass — there is no training step.

The same client runs predictions against the hosted endpoint, a named AWS
SageMaker endpoint, or locally. Select `mode="remote"`,
`mode="sagemaker"`, or `mode="local"`; there is no automatic
backend selection and every constructor requires an explicit `model=`.

### Use it from your AI coding assistant

Paste this into Claude Code, Cursor, or any AI coding assistant and it will wire
Nori into your own project:

````text
Look at my code/task/report here and figure out where Nori would best fit — it's
Synthefy's tabular foundation model for regression, used through the `synthefy`
client with no training loop and no hyperparameters. It runs fully on your own
machine (local mode, uses your GPU when one's available), or against the hosted
Synthefy API if you'd rather not run it locally.

1. Install it with this project's package manager, with the local runtime
   (e.g. `uv add "synthefy-nori"`, or `pip install -U "synthefy-nori"`).

2. Use it wherever a tabular regression / prediction step fits:

   ```python
   from synthefy import SynthefyNoriClient

   # model is required -- name a size: "nori-100m" (~98.3M), "nori-30m" (~29.2M),
   # "nori-1-5" (Nori 1.5, ~21M; needs synthefy-nori >= 0.22.0), or "nori-6m" (~6M base).
   client = SynthefyNoriClient(mode="local", model="nori-30m")   # runs on this machine, no API key

   y_pred = client.predict(
       X_train=X_train,   # lists, numpy arrays, or pandas — NaNs OK, imputed for you
       y_train=y_train,   # continuous target
       X_test=X_test,     # rows to score
   )                      # -> list of floats, one per X_test row (as_pandas=True for a Series)

   # Prediction intervals come free — no conformal/quantile add-ons:
   lo, mid, hi = client.predict(X_train, y_train, X_test,
                                output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
   ```

X is a numeric feature matrix (or a pandas DataFrame — non-numeric columns are
encoded for you); y is a continuous target. If I already have a model, wire Nori
up alongside it on the same train/test split and metric so I can compare them. If
the best place to plug Nori in isn't obvious, show me where you'd put it and
confirm with me before making changes.

Prefer not to run it locally? Use the hosted API instead — create a key at
https://docs.synthefy.com/setup/api_key, then:
`client = SynthefyNoriClient(api_key="YOUR_API_KEY", model="nori-30m")` (or set
SYNTHEFY_NORI_API_KEY).
````

### Hosted Usage (`mode="remote"`)

```python
from synthefy import SynthefyNoriClient

# The key is sent as `Authorization: Bearer <key>` (gateway default).
# Pass it explicitly or set the SYNTHEFY_NORI_API_KEY environment variable.
client = SynthefyNoriClient(api_key="your_api_key", model="nori-30m")

predictions = client.predict(
    X_train=[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],  # context features
    y_train=[1.0, 1.0, 2.0],                        # context targets
    X_test=[[2.0, 2.0], [0.5, 0.5]],                # query features
)
print(predictions)  # -> [<float>, <float>]  (one per X_test row)
```

`X_train`, `y_train`, and `X_test` accept Python lists, numpy arrays, or pandas
objects (a `DataFrame` for the feature matrices; a `Series` or single-column
`DataFrame` for `y_train`). When both `X_train` and `X_test` are DataFrames,
**non-numeric columns are encoded for you** — fit on `X_train` and applied
to `X_test` — so you can pass raw categorical columns directly:

```python
import pandas as pd

X_train = pd.DataFrame({"price": [9.99, 4.50, 7.25], "region": ["NW", "SE", "NW"]})
y_train = pd.Series([120.0, 305.0, 180.0])
X_test  = pd.DataFrame({"region": ["SE"], "price": [5.00]})  # order need not match

predictions = client.predict(X_train, y_train, X_test)  # 'region' is encoded
```

By default (`categorical_columns="auto"`) each remaining non-numeric column
becomes a single column of **ordinal codes** learned from `X_train`: retained
categories receive deterministic `0..K-1` codes, a rare or unseen value maps to
the bounded `K` `other` code, and a missing value stays NaN for server-side
imputation. Pass `categorical_columns=["region"]` to encode exactly named
columns and reject other strings, or `categorical_columns=None` to disable
categorical inference. Text and categorical declarations may not overlap. Pass
`categorical_encoding="onehot"` for the previous one-hot behavior (indicator
columns per category; missing values get their own indicator; unseen values map
to an all-zeros group). An automatically inferred column above
`max_categorical_cardinality` (default 100) raises an ambiguity error instead of
being silently dropped or embedded. Explicit categoricals use top-K plus
`other`. Temporal columns require explicit conversion. Numeric columns
(including `bool`) pass through unchanged,
with NaN imputed server-side. Any **object-dtype** column is treated as categorical
(including numeric-looking strings such as IDs or zip codes, and object date
values) — cast genuine numeric columns to a numeric dtype if you want them kept as
magnitudes. (Plain lists/numpy arrays must already be numeric — encoding needs
column names.)

For raw text columns, install `pip install "synthefy[text]"` and name them with
`text_columns=`. The client embeds those columns and optionally reduces them
with SVD before sending the resulting numeric matrix, so this works in both
local and remote modes:

```python
predictions = client.predict(
    X_train,
    y_train,
    X_test,
    text_columns=["review"],
    svd_dim=128,
)
```

Text embedding always happens on the client machine. By default,
`text_device="auto"` uses CUDA/ROCm when available, then Apple MPS, and falls
back to CPU. Pass `text_device="cpu"` (or another PyTorch device such as
`"cuda:1"`) to override automatic selection. The remote service receives only
the widened numeric features; remote mode does not move the sentence encoder to
the server.

Shapes are validated client-side: `X_train` and `y_train` must have the same
number of rows, and `X_test` must have the same number of features as `X_train`.
When both `X_train` and `X_test` are DataFrames, `X_test` is aligned to
`X_train`'s columns **by name** (so column order is irrelevant), and a mismatch
in the column sets raises. **Missing values (NaN) are allowed** — you don't need
to fill them in beforehand; the model imputes them server-side.

`predict` returns a plain `list[float]` by default. Pass **`as_pandas=True`** to
get a pandas `Series` instead — one value per `X_test` row, named after `y_train`
and indexed by `X_test`'s index (when `X_test` is a DataFrame), so predictions
join straight back:

```python
preds = client.predict(X_train, y_train, X_test, as_pandas=True)
# preds is a pd.Series named after y_train, sharing X_test's index
```

The client targets the Baseten inference **gateway**
(`https://inference.baseten.co/predict`); `model=` is required and names a size —
`"nori-100m"` (→ `synthefy/nori-100m`), `"nori-30m"` (→ `synthefy/nori-30m`),
`"nori-1-5"` (→ `synthefy/nori-1-5`), or `"nori-6m"` (→ `synthefy/nori-6m`). The
gateway resolves that slug to a deployment, so you never name a deployment yourself.

`timeout` and `max_retries` are also configurable on the constructor.

#### Authentication

- The only credential is your **Synthefy Nori API key**, created in the Synthefy
  Console. It authenticates against the Baseten-hosted gateway, but you do not
  need a Baseten account.
- Provide it via the `api_key` argument or the `SYNTHEFY_NORI_API_KEY`
  environment variable. It is sent as the header
  `Authorization: Bearer <key>`, which is what the gateway requires.

#### Errors

The Nori client reuses the package's
[exception hierarchy](#exception-hierarchy):

- HTTP `400` → `BadRequestError`, carrying the server's `error` string as the
  message (e.g. a missing field or unsupported task).
- HTTP `401` → `AuthenticationError` (bad or missing key).
- Transient errors (timeouts, connection errors, `429`, `5xx`) are retried with
  exponential backoff, then surface as `RateLimitError` / `InternalServerError` /
  `APITimeoutError` / `APIConnectionError`.

### Amazon SageMaker Usage (`mode="sagemaker"`)

Install the optional AWS transport and invoke a named asynchronous endpoint:

```bash
pip install "synthefy[aws]"
```

```python
from synthefy import SynthefyNoriClient

client = SynthefyNoriClient(
    mode="sagemaker",
    model="nori-30m",
    endpoint_name="nori-prod",
    region_name="us-east-1",
    s3_staging_uri="s3://customer-bucket/nori/invocations/",
)
predictions = client.predict(
    X_train=[[0.0], [1.0]],
    y_train=[0.0, 1.0],
    X_test=[[2.0]],
)
```

The client creates an argument-free `boto3.Session()` and therefore uses
boto3's standard credential chain: environment/shared config, web identity
(including GitHub OIDC), container or instance roles, and SSO profiles. It does
not accept AWS access keys. `model=`, `endpoint_name=`, and a customer-owned
`s3_staging_uri=` prefix are required. One endpoint serves every released peer model, and
an endpoint deployed from a fine-tuning job serves that model under its own name, so
`model=` also accepts a fine-tuned model's name (`ft-` then lowercase letters, digits and
hyphens, 63 characters at most); an endpoint that does not serve the name refuses it with
`SageMakerInvocationError`. The request carries the chosen model and the response identity
is checked so a routing mistake fails closed. Backend selection is always explicit;
installing another package never changes where a request runs.

For every model, the client writes the numeric request as a compact tensor file, uploads
it to the staging prefix, and calls `InvokeEndpointAsync`. `predict()` remains synchronous:
it polls SageMaker's output and failure objects, returns the ordinary typed prediction,
and best-effort deletes only that invocation's temporary objects after a completed response
or a confirmed server failure. A timeout, interrupted poll, or lost submission response does
not cancel work already queued in SageMaker: the client leaves the input and result objects
in place so cleanup cannot race the worker. Configure lifecycle expiration on the staging,
output, and failure prefixes (at least one day, longer than the six-hour queue TTL plus
one-hour processing limit) to remove objects from abandoned or uncertain invocations.
Container failures raise `SageMakerInvocationError`; AWS credential, signing, region, quota,
and throttling errors remain native AWS SDK exceptions. Set timeout/retries on the
constructor. HTTP-only `extra_headers=` are rejected for SageMaker, and per-call `timeout=`
is ignored with a warning.

[SageMaker Asynchronous Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html)
accepts S3 input objects up to 1 GB and allows processing for up to one hour. Those are
transport limits, not a promise that every table fits host/GPU memory. The client does not
split oversized tables because changing the in-context training set can change predictions.

### Local Usage (`mode="local"`, Optional, No Network)

The same prediction can run locally — no network call and no API key — via the
optional [`synthefy-nori`](https://pypi.org/project/synthefy-nori/)
package. Install the local runtime:

```bash
pip install "synthefy-nori"
```

Keep the installed `synthefy-nori` runtime current so it supports the
client options you use and reports recoverable degradation explicitly.

```python
from synthefy import SynthefyNoriClient

client = SynthefyNoriClient(mode="local", model="nori-30m")  # no API key needed
predictions = client.predict(
    X_train=[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    y_train=[1.0, 1.0, 2.0],
    X_test=[[2.0, 2.0]],
)
```

`predict` has the same signature in every mode. The `synthefy-nori` dependency
is imported lazily on first use; if it is not installed, a clear `ImportError` is
raised telling you to `pip install "synthefy-nori"`.

Local mode also preserves `synthefy-nori`'s degradation warnings and their messages.
With `synthefy-nori>=0.13.1`, an SVD failure warns under `SvdFallbackWarning` while
still returning a prediction. Scored or audited runs can turn that warning into an
exception around the client call; the client does not catch, wrap, or rewrite it:

```python
from synthefy_nori import SvdFallbackWarning, strict_pipeline

with strict_pipeline(SvdFallbackWarning):
    predictions = client.predict(X_train, y_train, X_test)
```

Backend selection is explicit. Use `mode="local"` for in-process execution or
`mode="remote"` for the hosted endpoint; installing `synthefy-nori` never changes
an existing client's routing.

### Large Tables and Memory (`memory_policy=`)

Nori does in-context regression, so your table is **input**: one prediction keeps a
per-layer key/value cache over every context row, and that cache — not the
model weights — is what exhausts GPU memory on a big table. `memory_policy=` decides
what to do about it. Omit it and the defaults handle almost every request.

```python
# A preset...
preds = client.predict(X_train, y_train, X_test, memory_policy="exact")        # never quantize
preds = client.predict(X_train, y_train, X_test, memory_policy="max_context")  # fit the largest table

# ...individual fields...
preds = client.predict(X_train, y_train, X_test, memory_policy={"cache_dtype": "int8"})
preds = client.predict(X_train, y_train, X_test,
                       memory_policy={"stream_context": True})  # bounded GPU staging

# ...or the typed model, which ships with the client — no synthefy-nori needed. Validated
# before the request goes out, so a typo or an out-of-range value costs no round trip.
from synthefy import MemoryPolicy
preds = client.predict(X_train, y_train, X_test,
                       memory_policy=MemoryPolicy(cache_dtype="int8", gpu_budget_frac=0.5))

print(client.last_memory_report["rung"])  # e.g. "resident_bf16"
```

`last_memory_report` is how you learn what actually happened, and it is worth reading:
the fallback chosen depends on the replica's free VRAM at that moment, not on your
request, so it is not knowable from your side.

| field | meaning |
|---|---|
| `rung` | which path served it — ordinary `resident_*` / `offload_*`, explicit `stream_bf16` / `stream_int8`, or a lower fallback |
| `est_cache_gb` / `resident_gb` | the cache's full-precision size, and its footprint at the chosen precision |
| `query_chunk` | query rows per forward pass |
| `dropped_context_rows` | context rows discarded to fit, `0` unless subsampling engaged |
| `clamped` | fields the server capped (host-RAM budgets only) |
| `notes` | remarks about the policy you sent, e.g. a budget that cannot take effect |

**Only the int8 rungs quantize the cache.** Ordinary `offload_*` moves bytes to host
RAM rather than approximating, so BF16 offload is bit-identical to staying resident.
Explicit streaming is numerically close but not bit-exact even at BF16 because bounded
online attention changes floating-point reduction order.

Use `memory_policy={"stream_context": True}` when context-attention GPU memory should
stop scaling with context length. It keeps the full context/KV state on the host, reports
`stream_bf16` or `stream_int8`, disables cross-call context reuse, and defaults to a
maximum staged-row cap of 2048. Runtime may use smaller K/V blocks to honor its fixed
FP32 workspace cap; fit-time OOMs retry a bounded 2048 → 1024 → 512 → 256 row ladder.
If the full cache cannot fit the allowed host budget, the request fails clearly
instead of silently switching to `plain_loop`. Set
`memory_policy={"allow_subsample": False}` to turn ordinary element-budget context
shortening into an error as well.

One field behaves differently over the network: **`elements_budget`**. The cache is only
built when the query set spans more than one chunk, and at default settings that needs
far more query rows than the hosted request-body limit (~64 MiB) allows — so lowering
`elements_budget` is what lets a hosted request reach the cached path at all.

In `mode="local"` the same argument works when the installed `synthefy-nori` exposes
the field. An older runtime raises `ImportError` with an upgrade hint before inference.
`last_memory_report` exposes the resolved local report just as it does for hosted calls.

**Nori 1.5 (`model="nori-1-5"`) builds no key/value cache**, so the cache fields and
the presets have nothing to act on. With it, `memory_policy=` may set only
`elements_budget` and `allow_subsample`, which decide whether context rows are dropped
to fit the element budget; anything else raises `ValueError` before a request is sent,
in every mode, and its `last_memory_report` always reports the `no_cache` rung.

### Choosing Context on Large Tables (`large_context_policy=`)

When a context table is larger than one Nori call can use effectively,
`large_context_policy=` makes row selection explicit instead of leaving it to a
memory-pressure subsample. It works in `local`, `remote`, and `sagemaker`
modes:

```python
preds = client.predict(
    X_train,
    y_train,
    X_test,
    large_context_policy="cluster_route",
    large_context_threshold=50_000,
    large_context_seed=0,
)
print(client.last_large_context_report)
```

Here are some commonly used built-in policies:

| policy | hosted status |
| --- | --- |
| `"random"` | supported |
| `"cluster_route"` | supported; recommended default when enabled |
| `"cluster_route_g4"` | supported |
| `"safeboost"` | supported |
| `"boost"` | supported; prefer `safeboost` |
| `"target_rank[cap=N]"` | supported; compare caps through a policy-list gate |

Hosted and SageMaker support built-ins from the installed Nori version. See
[`policies.py`](../../src/synthefy_nori/inference/policies.py) for the complete
current list and configuration options. Hosted modes forward one policy-name string
or a list of up to eight names unchanged, including parameter strings such as
`"safeboost[nu=0.25]"`. Custom
callables and module/file policies remain local-only.
`large_context_cache_entries` is also intentionally absent from the client:
each client call is one-shot, fits the supplied `X_train` again, and hosted
serving retains no customer context across requests.

For the 32k/64k target-rank comparison, send a list and choose the split that
matches row semantics:

```python
preds = client.predict(
    X_train,
    y_train,
    X_test,
    large_context_policy=[
        "target_rank[cap=32768]",
        "target_rank[cap=65536]",
    ],
    large_context_holdout="tail",  # chronological; use "random" for IID
)
```

The response echoes `holdout_strategy`; the client rejects a response that did
not honor it. The global 64k screen was 3.02x slower than 32k and lost 0.0129
mean R² on four LaDe temporal tables, so 64k is not a default.
The direct arms are measured; selecting between them with a tail gate still
needs a frozen temporal replay before that gate should become a default.

`last_large_context_report` is cleared before every call and works in all
three modes. It records whether the policy engaged, the honored policy,
threshold and seed, the context window, internal `nori_calls`, and whether
train-derived state was reused. A hosted client treats a missing or mismatched
report as an unsupported deployment and raises instead of returning a
valid-looking ordinary prediction.

Large-context policies currently return point predictions only
(`output_type="mean"` or `"median"`). Quantile/full distributions and Nori
Thinking variants reject the option before inference. Baseten and SageMaker
share this contract; Snowflake SPCS's positional four-value envelope cannot
carry it and rejects an appended options value.

Hosted use is still a full one-shot upload: `X_train` is sent and policy state
is recomputed on every call. Upload-once/query-many needs a separate,
tenant-isolated session API with authentication, routing, TTL cleanup, storage,
and billing. Also note that `cluster_route` can make up to eight internal model
calls while the existing gateway usage block still meters public request rows
and columns; production enablement therefore requires an explicit pricing/cost
decision after dev latency measurements.

### Prediction Intervals (`output_type=` / `quantiles=`)

Nori's forward pass produces a whole predictive distribution, not just a point
estimate, so **prediction intervals cost nothing extra** — no conformal wrapper,
no separate quantile models:

```python
lo, mid, hi = client.predict(
    X_train, y_train, X_test,
    output_type="quantiles", quantiles=[0.1, 0.5, 0.9],   # an 80% interval
)
```

`output_type` selects what comes back. Shared selectors use the same meanings as
`synthefy-nori`'s `NoriRegressor.predict`:

| `output_type` | Returns | Shape |
| --- | --- | --- |
| `"mean"` (default) | distribution mean — optimal for squared error / R² | `list[float]`, one per `X_test` row |
| `"median"` | distribution median — optimal for MAE | `list[float]` |
| `"quantiles"` | quantiles at the levels in `quantiles=` | `(n_levels, n_query)` — **level-major**, so `lo, mid, hi = ...` unpacks |
| `"full"` | the whole quantile bank | `dict` with `"quantiles"` `(n_query, K)`, `"taus"` `(K,)`, `"mean"` `(n_query,)` |

`quantiles=` takes tau levels strictly inside `(0, 1)`; it is required by — and
valid only with — `output_type="quantiles"`. The returned rows follow **your**
order, so `quantiles=[0.9, 0.1]` gives you high-then-low. Values come back in
original-`y` units, sorted to a valid (monotone) quantile function per row.

`as_pandas=True` returns a `DataFrame` instead: one row per `X_test` row (indexed
by `X_test`, so the bands join straight back) and one column per level, named
`"<target>[<level>]"` — the same convention the forecasting client uses:

```python
bands = client.predict(X_train, y_train, X_test, output_type="quantiles",
                       quantiles=[0.1, 0.9], as_pandas=True)
bands.columns   # ['price[0.1]', 'price[0.9]']  (named after y_train)
```

Use `"full"` for CRPS / interval scoring and calibration work; the bank is the
checkpoint's full quantile head (K = 999 on the default checkpoint), so prefer
`"quantiles"` when you only need a few levels — it keeps the response small.

Capability differs by mode:

- **Local** (`pip install synthefy-nori`): every `output_type` works.
  The installed runtime must support the requested distribution output;
  an older build raises `ImportError` with an upgrade hint. Quantile and
  full output require a compatible pinball checkpoint.
- **Remote**: needs a hosted deployment that serves distribution output. The
  server echoes back the `output_type` it honored, and the client **raises**
  rather than accept a mismatch:

  ```text
  ValueError: The hosted deployment did not serve output_type='median': it omitted
  the output_type field entirely, so it predates distribution output. Such a
  deployment answers with the distribution mean, which is indistinguishable from a
  real 'median' result, so this is raised rather than returning means as if they
  were what you asked for. Use local mode (pip install "synthefy-nori", then
  mode="local"), or point base_url/endpoint at a deployment that serves
  distribution output.
  ```

  That handshake is the point: a deployment that ignores `output_type` answers
  with means, which look exactly like a valid `"median"` result — so silence here
  would be a confidently wrong answer, not a missing feature.

`output_type`/`quantiles=` cannot be combined with `discretize=` /
`categorical_levels=` (below): discrete labels and a distribution summary are
different answers, so asking for both raises `ValueError`. An ordinary
`predict(...)` call is unaffected by any of this — the request body it sends is
byte-for-byte what it always was.

### Categorical / Ordinal Targets (`discretize=` / `categorical_levels=`)

When the target only takes a small set of discrete values (a 1–5 rating, a
count, a quality score), pass `discretize=` and every returned prediction is
one of the target's own levels instead of a continuous estimate:

```python
labels = client.predict(X_train, y_train, X_test, discretize="map-cell")
labels = client.predict(
    X_train, y_train, X_test,
    discretize="map-cell",
    categorical_levels=[1, 2, 3, 4, 5],   # the full scale, if the context may under-cover it
)
```

Discretization is **strictly opt-in** — nothing is snapped unless you ask.
`categorical_levels` is the set of values the target can take (numeric; order
and duplicates don't matter); it defaults to the distinct values of `y_train`, which is
leak-safe. A `NaN` prediction stays `NaN` rather than becoming a confident
label.

The strategy is the choice that matters, and the right one depends on the metric
you are scored on: `"map-cell"` (accuracy-optimal, the default when only
`categorical_levels` is given), `"median-cell"` (MAE-optimal), `"snap-mean"`
(QWK), `"snap-median"`, `"expected-level"` (a lattice-informed *continuous*
value, for analysis), `"prior-match"` (calibration experiments). See the
`synthefy-nori` docs for the benchmark behind those.

Every strategy works in every mode. Remote and SageMaker send the request to the
deployment, which owns the predictive distribution the distribution-based
strategies need; local mode runs the same code in your own process. The response
names the strategy that ran, and the client raises rather than hand back
continuous predictions from a deployment too old to know the field. Local mode
needs a `synthefy-nori` recent enough to ship `synthefy_nori.discretize`, and
raises an `ImportError` with an upgrade hint otherwise.

Two exceptions. Nori Thinking deployments refuse `discretize`: they fit and
predict in one call and own their regressors internally, so an estimator-level
option never reaches them. And the Snowflake SPCS batch envelope carries no
per-row options at all, so it always serves continuous predictions.

If your task is scored by squared error / R², don't discretize — the
continuous mean is already optimal for those metrics.

## Nori: Fine-tuning on Amazon SageMaker

`SynthefyNoriFineTuner` fine-tunes Nori on your table in your own AWS account, deploys the result as a SageMaker
asynchronous endpoint and gives you a `SynthefyNoriClient` for it. The job, the endpoint and your data stay in your
account: no Synthefy service sits in front of them, and Synthefy can neither see nor stop them.

### What you need

```bash
pip install -U "synthefy[aws]>=7.3.0"
```

Subscribe to the Nori fine-tuning Algorithm in AWS Marketplace (the listing's View purchase options, then Subscribe at
the bottom of that page). Once the subscription is active, choose Launch your software on the same page and copy the ARN
for your region under Algorithm ARNs; to come back to it later, open the listing and choose View purchase options, then
Launch your software. Your bucket, the training job, the endpoint and every `region_name` you pass must be in that
region. The client refuses an Algorithm ARN from another region before any AWS call.

SageMaker runs the job and the endpoint as an execution role. The quickest setup gives it AWS's
`AmazonSageMakerFullAccess` policy: in the IAM console choose Create role, AWS service, then SageMaker as the use case,
and name it `nori-fine-tune-execution`. Or create it, and a bucket, from the command line:

```bash
aws iam create-role --role-name nori-fine-tune-execution --assume-role-policy-document \
  '{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name nori-fine-tune-execution --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws s3 mb s3://sagemaker-us-east-1-ACCOUNT_ID --region us-east-1
```

`AmazonSageMakerFullAccess` grants object access to buckets whose names contain `sagemaker`, such as the one above; a
bucket named otherwise is refused. The identity that runs your code needs SageMaker and S3 access too, and
`AmazonSageMakerFullAccess` covers every call the client makes. To grant both identities less, use the policies under
Least-privilege IAM below.

You need quota for one training and one endpoint instance; a job's default type follows its base:

| Use | Instance types | Default |
| --- | --- | --- |
| Train `nori-6m` | `ml.g5.xlarge`, `ml.g5.2xlarge`, `ml.g6.xlarge` (24 GB); `ml.g6e.xlarge`, `ml.g6e.2xlarge` (48 GB) | `ml.g5.xlarge` |
| Train `nori-100m` | `ml.g6e.xlarge`, `ml.g6e.2xlarge` only | `ml.g6e.xlarge` |
| Endpoint | `ml.g5.xlarge`, `ml.g5.2xlarge`, `ml.g6.xlarge` | `ml.g5.xlarge` |

The quotas are named like `ml.g5.xlarge for training job usage` and `ml.g5.xlarge for endpoint usage`. At 0 SageMaker
refuses the create call (`ResourceLimitExceeded`) before anything bills, and the client raises `FineTuningError` naming
the quota; ask for more in the Service Quotas console. The same quotas set how much runs at once: N jobs on
`ml.g5.xlarge` need its training quota at N or more, and N endpoint instances of a type need its endpoint quota at N
or more. New accounts often start at 0 for GPU types; request what you need before the first job, since approval can
take a few days. Check your quota:

```bash
aws service-quotas list-service-quotas --service-code sagemaker --region us-east-1 \
  --query "Quotas[?contains(QuotaName, 'ml.g5.xlarge')].[QuotaName,Value]" --output table
# lists the training and the endpoint quota; for nori-100m, run it again with ml.g6e.xlarge
```

### Your table

Fine-tuning is regression only, on a CSV table with one header row. Every column the job trains on must be numeric;
features may have missing values, the target must be finite in every row. Encode categorical columns as numbers before
you upload and keep that encoding for prediction: a text column fails the job.

`training_data` is one object ending in `.csv` or a folder ending in `/`; the client refuses anything else, because S3
matches prefixes as strings (`.../nori/data/train` would also read `train-old/`). For one object, `fine_tune()` also
lists the bucket and refuses it when another key starts with its key: SageMaker would read `train.csv.backup.csv` as
well. Under a folder the job reads every uncompressed file ending in `.csv` in the folder and its sub-folders (an
`old/` sub-folder is read too), and those files must share one header; it ignores other files and fails on a Parquet
file. `validation_data` is optional, takes the same form and has exactly the training table's columns. The training
table needs at least 1,111 rows, or 1,000 with a validation channel, so that 1,000 remain after the job sets aside its
report rows. Above 200,000 rows or 30 million cells (rows x features), the job trains on a subsample drawn with its
`seed`.

### The whole flow

```python
import logging
import boto3
import pandas as pd
from synthefy import FineTuneHyperparameters, FineTuneJobRequest, SynthefyNoriFineTuner

logging.basicConfig(level=logging.INFO)  # wait() logs each status change
REGION = "us-east-1"
ALGORITHM_ARN = "arn:aws:sagemaker:us-east-1:..."  # copied from your subscription, for REGION
ROLE_ARN = "arn:aws:iam::ACCOUNT_ID:role/nori-fine-tune-execution"
BUCKET, TARGET = "sagemaker-us-east-1-ACCOUNT_ID", "y"

table = pd.read_csv("train.csv")
features = [column for column in table.columns if column != TARGET]  # the order the job trains on
boto3.client("s3", region_name=REGION).upload_file("train.csv", BUCKET, "nori/data/train/train.csv")

with SynthefyNoriFineTuner(region_name=REGION) as tuner:
    request = FineTuneJobRequest(
        algorithm=ALGORITHM_ARN,
        role_arn=ROLE_ARN,
        training_data=f"s3://{BUCKET}/nori/data/train/",
        output_path=f"s3://{BUCKET}/nori/output/",
        hyperparameters=FineTuneHyperparameters(target_column=TARGET, model_name="ft-acme-churn-v1", bag=2, steps=30),
    )
    print(request.instance_type, request.max_runtime_s)  # ml.g5.xlarge 4500
    job = tuner.fine_tune(request)
    print(job.job_name)  # watch it from another session
    job = tuner.wait(job.job_name)
    print(job.status, job.secondary_status, job.status_message, job.billable_seconds)
    if job.status == "Failed":
        raise SystemExit(job.failure_reason)
    report = tuner.get_training_report(job.job_name)
    print(report.val_mse_before, report.val_mse_after, report.cut_short)
    endpoint = tuner.deploy(job)  # asynchronous endpoint on ml.g5.xlarge; returns once it is InService
    try:
        with tuner.create_client(endpoint, s3_staging_uri=f"s3://{BUCKET}/nori/invocations/") as client:
            context = table.sample(n=min(len(table), 2000), random_state=0)
            new_rows = pd.read_csv("new.csv")[features]  # rows the job never saw: same columns, same order
            print(client.predict(X_train=context[features], y_train=context[TARGET], X_test=new_rows,
                                 categorical_columns=None))
            assert client.last_model_version == report.model_version
    finally:
        tuner.undeploy(endpoint)
```

`bag=2, steps=30` is a small run. With capacity available, the flow takes about 10 minutes: the job 3 to 5 minutes, the model package under a minute and the endpoint 4 to 6 minutes. Capacity waits add to that (below). The endpoint does not store your table, so every `predict()` still sends context rows and uploads
them to your staging prefix: the flow caps them at 2,000 to keep each request small. Send the training
feature columns in training order, without the target, with `categorical_columns=None`, so a text column raises in the
client instead of being encoded in a way the job never saw. `last_model_version` equal to `report.model_version` shows
the fine-tuned model answered; the staging prefix works as in Amazon SageMaker Usage above. Expire `nori/invocations/`
and `nori/output/ft-acme-churn-v1/async/` after at least one day, never `nori/output/` as a whole: the model package reads
its model from there. `undeploy()` deletes the endpoint, its endpoint configs and the SageMaker model, and keeps the
package for a new deploy. The endpoint bills every hour from creation until `undeploy()`, whether or not you send
requests; prices per instance type and region are on [SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/).

Other hyperparameters, with defaults: `base_model` (`nori-6m`, or `nori-100m`); `feature_columns` (all but the target, in
table order); `bag` (8: the number of children the job trains, 1 to 8; the fine-tuned model is built from them, and each
child has its own steps and time budget); `steps` (100: maximum training steps per child); `patience` (25: a child stops
after this many steps without improvement); `budget_s` (900 s per child); `lr` (1e-5); `seed` (0). `model_name` starts with `ft-`, then 1 to 60
lowercase letters, digits and `-` starting with a letter or digit, and cannot be a released model's name. End it with a
letter or digit as well: `deploy()` names the endpoint after it, and SageMaker refuses an endpoint name that ends with `-`.

### Watching a job

`wait()` blocks, so watch the job from a second session with the name `fine_tune()` printed:

```python
import pandas as pd
from synthefy import SynthefyNoriFineTuner

JOB = "nori-ft-acme-churn-v1-..."  # printed by fine_tune()
with SynthefyNoriFineTuner(region_name="us-east-1") as tuner:
    job = tuner.get_status(JOB)  # one DescribeTrainingJob
    print(job.status, job.secondary_status, job.status_message)
    for line in tuner.get_logs(JOB, follow=True):  # returns when the job ends
        print(line)
    curves = pd.DataFrame(tuner.get_training_curves(JOB))  # columns child, step, train_loss, val_mse
```

After every validated step the job prints `child=<k> step=<n> train_loss=<x> val_mse=<x>`. `get_training_curves()` turns each such
line into one row, and falls back to the report's curves once the log stream is gone. The console's metric charts come
from the same lines, sampled per minute, so a short job shows only a handful of points there.

### Reading the result

`report.val_mse_after` against `report.val_mse_before` is the result: the fine-tuned model and the base, scored on the
same rows with the same context. Lower is better: the distance between the two is what fine-tuning gained on your table,
so always read them together. Both are mean squared errors in standardized target units (the target scaled by the
training rows' mean and standard deviation); multiply the square root by that standard deviation for your own units.
Without a validation channel the job sets aside min(10 % of the rows, 4,096) before training, rows no training step
sees, so both scores are out of sample. With one, its rows (up to 4,096) are scored and also steer the training, so
`val_mse_after` runs a little optimistic there. `validation_source` names the mode, `n_val_rows` counts the rows
scored, and `n_rows_used` the rows kept after the 200,000-row and 30-million-cell limits; without a validation channel
that count includes the report rows, which no training step sees.

`job.latest_metrics` and the console charts show the last child's last step: progress, not the result. Each child's
`val_mse` in `get_training_curves()` is that child's own validation score, the one that decides when it stops, not the
report's score, so compare the report's two numbers instead.

### Several models and endpoints

A job trains one model, and an endpoint serves the one model it was deployed from: a request naming another model is
refused. For several models, run one job per `model_name` and deploy each:

```python
endpoints = {}
for name, target in [("ft-acme-churn-v1", "churn"), ("ft-acme-ltv-v1", "ltv")]:
    hp = FineTuneHyperparameters(target_column=target, model_name=name)
    # max_runtime_s=None: the copy derives its runtime from these hyperparameters
    per_model = request.model_copy(update={"hyperparameters": hp, "max_runtime_s": None})
    job = tuner.wait(tuner.fine_tune(per_model).job_name)
    endpoints[name] = tuner.deploy(job)  # one endpoint per model, named after it
```

Jobs run at the same time when your training quota allows it (see What you need).

One model can sit behind several endpoints: pass a different `endpoint_name` to each `deploy()` call (the default is
the model's name). The model package is reused, each endpoint gets its own async output prefix under `output_path`, and
each bills on its own. Separate endpoints isolate workloads, for example one per team or per environment.

For throughput, add instances to one endpoint instead: each instance handles one request at a time and extra requests
wait in the endpoint's queue, so N instances answer N requests at once. `deploy()` creates the endpoint with one
instance; raise the count from the SageMaker AI console (the endpoint's configuration) or with
`aws sagemaker update-endpoint` and a new endpoint config, within your endpoint quota.

### Stopping and failures

`tuner.stop(job.job_name)` stops a job: SageMaker sends it SIGTERM, then SIGKILL 120 s later. The job is built to use
that window: it writes a model from the training done so far when any of it improved on the base, and ends `Stopped`
with `cut_short` true; `val_mse_after` is scored only if that fits within 60 s of
the signal, else null. `deploy()` accepts a `Stopped` job that has a model and warns that it was cut short. At `max_runtime_s` SageMaker
stops the job the same way, so it ends `Stopped`, not `Failed`; the default is 1.5 x `bag` x `budget_s` + 1,800 s
(12,600 s with default settings), and the client refuses less than `bag` x `budget_s`. SageMaker's own ceiling is 28
days per job; for a long run, raise `budget_s` and `steps` per child and `max_runtime_s` together. A job keeps no
checkpoints and cannot resume: a new job starts again from the base.

| You see | What to do |
| --- | --- |
| `deploy()` or `get_training_report()` raises `FineTuningError` saying the `Stopped` job has no model artifact | Nothing had improved on the base when the job stopped, so it wrote no model. Start a new job. |
| `Failed`, and `failure_reason` names your table or settings | The job refused the input once its instance was up. Fix it and start a new job; `tuner.get_logs()` has any traceback. |
| `InProgress` in `Pending` for a long time with the message `Training job waiting for capacity`, or `Failed` with `CapacityError:` | AWS has no spare capacity for that type. The wait is unbilled and can last hours. Stop it and start a new job on another type. |
| `fine_tune()` raises `FineTuningError` saying SageMaker started the job but reading its status failed | The job runs and bills. Call `tuner.get_status(error.job_name)` to follow it, not `fine_tune()` again, which starts a second job. |
| `wait()` raises `APITimeoutError` | It waited the time limit plus 7,200 s. The job still runs: call `wait()` again, or `stop()`. |
| `deploy()` raises `FineTuningError` naming `InsufficientInstanceCapacity` | AWS found no instance of that type in time. Call `tuner.undeploy("ft-acme-churn-v1")`, then `tuner.deploy(job, instance_type=...)` with another endpoint type; it reuses the model package. |
| `deploy()` raises `APITimeoutError` | The endpoint was not InService within `timeout_s` (3,600 s by default) and is still starting and billing. Call `tuner.undeploy("ft-acme-churn-v1")`, then deploy again with a larger `timeout_s`. |

### Least-privilege IAM

For production, replace `AmazonSageMakerFullAccess` with two policies that grant each identity only what it uses; with
them the bucket can have any name. Save this permission document as `execution-policy.json`, replacing `BUCKET`,
`REGION`, `ACCOUNT_ID` and `ALGORITHM_ARN`. It lets the job read `nori/data/` and write `nori/output/`, lets the endpoint
read `nori/invocations/` and write under `nori/output/`, and lets SageMaker describe the Algorithm and your model package
(named after the job, `nori-ft-...` unless you set `job_name`; if you do, use its prefix here and in the EventBridge rule
under Alarms) as the execution role.

```json
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"], "Resource": "arn:aws:s3:::BUCKET"},
  {"Effect": "Allow", "Action": "s3:GetObject", "Resource": "arn:aws:s3:::BUCKET/nori/data/*"},
  {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
   "Resource": ["arn:aws:s3:::BUCKET/nori/output/*", "arn:aws:s3:::BUCKET/nori/invocations/*"]},
  {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
   "Resource": ["arn:aws:logs:REGION:ACCOUNT_ID:log-group:/aws/sagemaker/TrainingJobs*",
                "arn:aws:logs:REGION:ACCOUNT_ID:log-group:/aws/sagemaker/Endpoints/*"]},
  {"Effect": "Allow", "Action": "cloudwatch:PutMetricData", "Resource": "*"},
  {"Effect": "Allow", "Action": ["sagemaker:DescribeAlgorithm", "sagemaker:DescribeModelPackage"],
   "Resource": ["ALGORITHM_ARN", "arn:aws:sagemaker:REGION:ACCOUNT_ID:model-package/nori-ft-*"]}]}
```

Put it on the role in place of the managed policy:

```bash
aws iam put-role-policy --role-name nori-fine-tune-execution --policy-name nori --policy-document file://execution-policy.json
aws iam detach-role-policy --role-name nori-fine-tune-execution --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

The identity that runs your code needs the policy below. Save it as `caller-policy.json`, replacing `BUCKET`, `REGION`
and `ACCOUNT_ID`, and attach it to that identity; for a role, `aws iam put-role-policy --role-name YOUR_ROLE
--policy-name nori-caller --policy-document file://caller-policy.json`. `s3:ListBucket` makes a missing object read as
not found, which the client relies on while it waits for a result, and lets `fine_tune()` check a one-object channel.
With `kms_key_id`, the key's policy must let both identities use it.

```json
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Resource": "*", "Action": [
    "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob", "sagemaker:DescribeAlgorithm",
    "sagemaker:CreateModelPackage", "sagemaker:DescribeModelPackage", "sagemaker:DeleteModelPackage", "sagemaker:CreateModel",
    "sagemaker:DescribeModel", "sagemaker:DeleteModel", "sagemaker:CreateEndpointConfig", "sagemaker:DescribeEndpointConfig",
    "sagemaker:ListEndpointConfigs", "sagemaker:DeleteEndpointConfig", "sagemaker:CreateEndpoint", "sagemaker:DescribeEndpoint",
    "sagemaker:DeleteEndpoint", "sagemaker:InvokeEndpointAsync"]},
  {"Effect": "Allow", "Action": "iam:PassRole", "Resource": "arn:aws:iam::ACCOUNT_ID:role/nori-fine-tune-execution",
   "Condition": {"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}}},
  {"Effect": "Allow", "Action": ["logs:DescribeLogStreams", "logs:GetLogEvents"],
   "Resource": ["arn:aws:logs:REGION:ACCOUNT_ID:log-group:/aws/sagemaker/TrainingJobs",
                "arn:aws:logs:REGION:ACCOUNT_ID:log-group:/aws/sagemaker/TrainingJobs:*"]},
  {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::BUCKET"},
  {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], "Resource": "arn:aws:s3:::BUCKET/nori/*"}]}
```

### Alarms

An EventBridge rule can mail you when a job ends `Failed` or `Stopped` (including a stop you asked for), and two
CloudWatch alarms watch an endpoint you keep running (the flow above deletes its own). The CLI adds no SNS topic policy,
so save this one as `topic-policy.json`, replacing `REGION` and `ACCOUNT_ID`. Run the commands below as an identity that
may create SNS topics, EventBridge rules and CloudWatch alarms.

```json
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Principal": {"Service": "events.amazonaws.com"}, "Action": "sns:Publish",
   "Resource": "arn:aws:sns:REGION:ACCOUNT_ID:nori-alerts",
   "Condition": {"ArnEquals": {"aws:SourceArn": "arn:aws:events:REGION:ACCOUNT_ID:rule/nori-fine-tune-ended"}}},
  {"Effect": "Allow", "Principal": {"Service": "cloudwatch.amazonaws.com"}, "Action": "sns:Publish",
   "Resource": "arn:aws:sns:REGION:ACCOUNT_ID:nori-alerts", "Condition": {"StringEquals": {"aws:SourceAccount": "ACCOUNT_ID"}}}]}
```

```bash
export AWS_DEFAULT_REGION=us-east-1  # the Algorithm's region
TOPIC=$(aws sns create-topic --name nori-alerts --query TopicArn --output text)
aws sns set-topic-attributes --topic-arn "$TOPIC" --attribute-name Policy --attribute-value file://topic-policy.json
aws sns subscribe --topic-arn "$TOPIC" --protocol email --notification-endpoint you@example.com
# confirm the subscription from the email AWS sends
aws events put-rule --name nori-fine-tune-ended --event-pattern '{"source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Training Job State Change"],
  "detail": {"TrainingJobStatus": ["Failed", "Stopped"], "TrainingJobName": [{"prefix": "nori-ft-"}]}}'
aws events put-targets --rule nori-fine-tune-ended --targets "Id=sns,Arn=$TOPIC"
aws cloudwatch put-metric-alarm --alarm-name ft-acme-churn-v1-failures --namespace AWS/SageMaker --metric-name InvocationFailures \
  --dimensions Name=EndpointName,Value=ft-acme-churn-v1 Name=VariantName,Value=AllTraffic --statistic Sum --period 300 \
  --evaluation-periods 1 --threshold 0 --comparison-operator GreaterThanThreshold --treat-missing-data notBreaching --alarm-actions "$TOPIC"
aws cloudwatch put-metric-alarm --alarm-name ft-acme-churn-v1-queue-age --namespace AWS/SageMaker \
  --metric-name ApproximateAgeOfOldestRequest --dimensions Name=EndpointName,Value=ft-acme-churn-v1 --statistic Maximum --period 300 \
  --evaluation-periods 1 --threshold 1800 --comparison-operator GreaterThanThreshold --treat-missing-data notBreaching --alarm-actions "$TOPIC"
aws cloudwatch list-metrics --namespace AWS/SageMaker --dimensions Name=EndpointName,Value=ft-acme-churn-v1
```

The endpoint alarms use SageMaker's asynchronous inference metrics: `InvocationFailures` per endpoint and variant
(`deploy()` names the variant `AllTraffic`), `ApproximateAgeOfOldestRequest` per endpoint. Both appear in `list-metrics` with exactly these dimensions once the endpoint has served a request. The queue alarm fires when a
request has waited half an hour; the client lets a request wait at most six hours before SageMaker expires it. An
alarm on a wrong dimension set never fires and still shows `OK`, because `notBreaching` reads a metric without data as
healthy; `list-metrics` (the last command) shows a metric only after its first data point. To make `InvocationFailures` appear, and to test that alarm, call `predict()`
once through a `SynthefyNoriClient` built with `mode="sagemaker"`, `model="nori-6m"`,
`endpoint_name="ft-acme-churn-v1"` and your `s3_staging_uri`: the endpoint serves only your model, so it refuses the request, the client raises
`SageMakerInvocationError`, and SageMaker records one failure.

## API Reference

### SynthefyNoriClient (Tabular Regression)

- `SynthefyNoriClient(api_key=None, *, mode="remote", timeout=300.0, max_retries=2, base_url=..., endpoint=..., model, user_agent=None, endpoint_name=None, region_name=None, s3_staging_uri=None)`: `model` is **required everywhere** and accepts the released Nori variants (`nori-6m`, `nori-1-5`, `nori-30m`, `nori-100m`, and `nori-30m-thinking-medium`) or an explicit custom HTTP slug; in SageMaker mode it also accepts a fine-tuned model's name (`ft-` then lowercase letters, digits and hyphens, 63 characters at most). There is no `None`/default model path. SageMaker requires `endpoint_name` plus a customer-owned `s3_staging_uri` and uses asynchronous inference for every variant while `predict()` still returns one normal result.
  - `mode`: `"remote"` (hosted, default), `"local"` (in-process via
    `synthefy-nori`), or `"sagemaker"` (a named SageMaker endpoint using the AWS
    credential chain).
  - `api_key` (remote mode) falls back to the `SYNTHEFY_NORI_API_KEY`
    environment variable. Not required in local mode.
  - Hosted Nori is reached by gateway slug — that is the path Synthefy meters,
    rate-limits and grants per key. To target a single-model endpoint you host
    yourself, pass your own `base_url`/`endpoint` and an explicit custom model slug.
- `predict(X_train, y_train, X_test, task="regression", *, output_type="mean", quantiles=None, categorical_columns="auto", max_categorical_cardinality=100, categorical_encoding="ordinal", text_columns=None, svd_dim=128, embedder="minilm", text_device="auto", timeout=None, extra_headers=None) -> List[float]`
  - Returns one predicted value per row of `X_test`. `timeout`/`extra_headers`
    apply to remote mode only.
  - `output_type=` picks what comes back from the predictive distribution:
    `"mean"` (default), `"median"`, `"quantiles"` (with
    `quantiles=[...]`, returns `(n_levels, n_query)`), or `"full"` (the whole
    quantile bank as a dict). See
    [Prediction Intervals](#prediction-intervals-output_type--quantiles).
    Everything other than `"mean"` needs local mode or a hosted deployment that
    serves distribution output.
  - Inputs accept Python lists, numpy arrays, or pandas DataFrames/Series.
    Lists/arrays must be numeric. DataFrame `X_test` is aligned to `X_train`
    by column name and named categorical/text roles are replayed from training;
    `categorical_encoding="ordinal"` is the default and `"onehot"` is available;
    missing values (NaN) are imputed server-side.
  - `categorical_columns` is `"auto"`, an exact sequence of names, or `None` to
    disable inference. `max_categorical_cardinality` (default 100) bounds
    retained levels; ambiguous auto columns above it raise, while explicitly
    named categoricals use top-K plus `other`.
  - `text_columns` embeds named raw-text DataFrame columns client-side. The
    default `text_device="auto"` prefers CUDA/ROCm, then Apple MPS, then CPU;
    install the `text` extra and pass `text_device="cpu"` or another PyTorch
    device string to override it.
  - `as_pandas=True` returns a pandas `Series` (named after `y_train`, indexed by
    `X_test`) instead of the default `list[float]` — or a `DataFrame` with one
    column per level (`"<target>[<level>]"`) for
    `output_type="quantiles"`/`"full"`.
  - `discretize=` / `categorical_levels=` map predictions onto a discrete
    target's levels (see
    [Categorical / Ordinal Targets](#categorical--ordinal-targets-discretize--categorical_levels));
    the full strategy set is available in every mode.
- `mode`: the explicitly selected execution mode.
- `last_model_version`: the `sha256:<hex>` `model_version` the deployment reported on
  the most recent remote or SageMaker `predict()`, cleared at the start of every call
  and left `None` by a call that raises, so a refused answer never sets it; `None` in
  local mode and from a deployment that predates the field. Compare it with a
  fine-tuning job's `TrainingReport.model_version` to prove which model answered.
- `close()` / context manager support (`with SynthefyNoriClient(...) as client:`).

### Exception Hierarchy

Import these exceptions from `synthefy.errors`; all inherit from `SynthefyError`:

- `APITimeoutError`: Request timed out
- `APIConnectionError`: Network/connection issues
- `FineTuningError`: A fine-tuning job, or the deployment of its result, cannot give you a model; carries `job_name`, `job_arn`, `status`, `failure_reason`, and `created` (what a failed deployment had already made)
- `APIStatusError`: Base class for HTTP status errors
  - `BadRequestError` (400, 422): Invalid request data
  - `AuthenticationError` (401): Invalid API key
  - `PermissionDeniedError` (403): Access denied
  - `NotFoundError` (404): Resource not found
  - `RateLimitError` (429): Rate limit exceeded
  - `InternalServerError` (5xx): Server errors

Each status error includes:
- `status_code`: HTTP status code
- `request_id`: Request ID for debugging (if available)
- `error_code`: API-specific error code (if available)
- `response_body`: Raw response body

## Configuration

### Environment Variables

- `SYNTHEFY_NORI_API_KEY`: Your hosted-Nori API key (`SynthefyNoriClient`)

## Support

For support and questions:
- Email: contact@synthefy.com

## License

Apache License 2.0 - see LICENSE file for details.
