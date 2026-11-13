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
pip install -U "synthefy[aws]>=9.0.0"
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
| Train `nori-6m` or `nori-1-5` | One GPU: `ml.g5.xlarge`, `ml.g5.2xlarge`, `ml.g6.xlarge` (24 GB); `ml.g6e.xlarge`, `ml.g6e.2xlarge` (48 GB). Several GPUs: `ml.g5.12xlarge` (4 x 24 GB), `ml.g5.48xlarge` (8 x 24 GB), `ml.g6e.12xlarge` (4 x 48 GB), `ml.g6e.48xlarge` (8 x 48 GB) | `ml.g5.xlarge` |
| Train `nori-100m` | The 48 GB types only: `ml.g6e.xlarge`, `ml.g6e.2xlarge`, `ml.g6e.12xlarge` (4 GPUs), `ml.g6e.48xlarge` (8 GPUs) | `ml.g6e.xlarge` |
| Endpoint | `ml.g5.xlarge`, `ml.g5.2xlarge`, `ml.g6.xlarge` | `ml.g5.xlarge` |

A job trains one model per GPU at a time, so the four multi-GPU types train several models side by side:
`ml.g5.12xlarge` and `ml.g6e.12xlarge` have 4 GPUs, `ml.g5.48xlarge` and `ml.g6e.48xlarge` have 8
(`synthefy.nori_data_models.MULTI_GPU_INSTANCE_TYPES`). On a type with N GPUs the `bag` models train in
ceil(`bag` / N) rounds, against `bag` rounds on one GPU: the default `bag=8` takes two on a 4-GPU type and one on an
8-GPU type. A multi-GPU type finishes sooner but bills more per GPU-hour than the one-GPU types above with the same
GPU (us-east-1 on-demand prices, October 2026), so the same job costs more on it. The report's `workers`, `gpus` and
`failed_children` say what ran (see Reading the result).

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

Fine-tuning is regression only, on a table in Parquet files (`.parquet` or `.pq`). Every column the job trains on must
be numeric; features may have missing or infinite values, and both count as missing; the target must be finite in
every row. Encode categorical columns as numbers before you upload and keep that encoding for prediction: a text
column fails the job. The `aws` extra installs `pyarrow`, so pandas writes a table as Parquet with
`table.to_parquet("train.parquet", index=False)`, and a CSV file converts with
`pd.read_csv("train.csv").to_parquet("train.parquet", index=False)`.

`training_data` is one object ending in `.parquet` or `.pq`, or a folder ending in `/`. The client refuses anything else
before any AWS call: a `.csv` object, because the job reads Parquet only, and a folder without its `/`, because S3
matches prefixes as strings (`.../nori/data/train` would also read `train-old/`). For one object, `fine_tune()` and
`get_training_plan()` also list the bucket and refuse it when another key starts with its key: SageMaker would read
`train.parquet.backup.parquet` as well. Under a folder the job reads every `.parquet` and `.pq` file in the folder and
its sub-folders (an `old/` sub-folder is read too), and those files must have the same columns. It skips empty files
and every file or folder whose name starts with `_` or `.`, such as the `_SUCCESS` marker Spark writes, and fails on
any other file, a `.csv` file among them, before it reads any row. So a folder holds this table's Parquet files only:
delete any other file from it, a `train.csv` among them, before you start a job. Point a channel at plain Parquet
files, not at a Delta Lake or Iceberg table's folder: its data files still hold rows the table has deleted or
replaced, and the job reads every file. A column that exists only in folder names (`region=1/`) is not read; keep
every column in the files. `validation_data` is optional, takes the same form and has exactly the training table's
columns.

SageMaker copies the tables to the instance's local disk before the job starts (its File input mode, the one a
SageMaker Algorithm's channels take). Every instance type the Algorithm offers has at least 250 GB of local storage,
which holds the tables and the artifact; `volume_size_gb` sizes the disk only on types without it. The job reads the
tables one row group of a Parquet file at a time, so the instance's memory bounds the size of a row group, not of the
table. A larger table takes longer to copy and to read, and both count toward `max_runtime_s`. A row group may hold at most 2,000,000 rows and 200,000,000 cells of the columns the job reads (rows x
the feature and target columns); a file with a larger one fails the job before it reads any row, naming the file, the
row group and a row group size that fits: write it again with smaller row groups, for example
`table.to_parquet("train.parquet", index=False, row_group_size=1_000_000)` for a table of up to 200 columns.

Each model trains on at least 1,000 rows. With `bag` 1 the training table needs about 1,400 rows, or 1,000 with a
validation channel, as the job sets aside its report rows and, without a validation channel, a validation fold. With
`bag` above 1 under `"shard"` the table needs about 1,150 rows per model, or 1,050 with a validation channel (2,300 for
`bag` 2, 9,200 for `bag` 8). Each model trains on its share of the training rows. With the default
`row_partition_strategy="shard"` the `bag` models split the training rows between them, each row in one model's share;
with `"folds"` each model trains on every training row outside its own validation fold. A share holds at least 1,000
rows: on a table too small for `bag` shards of that size the job trains fewer models, and the report's `bag` gives the
number it trained. A model takes the training steps that read each row of its share `n_passes_per_row` times (unset by
default: each row once), and at least 100; `n_steps_per_model` sets an exact number of steps and is refused together
with `n_passes_per_row`. So by default the plan reads the whole table once. A model stops sooner after `patience`
validation scores in a row without a better one (25 by default) or when the job reaches `max_training_hours`, and
each model's `stop_reason` in the report's `child_curves` says which; `patience=None` lets every model take every
step of its plan.

Patience is a wait in training steps: `patience` scores x `validation_every_n_steps`, the steps between two scores.
When you leave `validation_every_n_steps` unset, the default keeps that wait at 100 steps or fewer: a model is scored
every hundredth of its steps, but at most every 100 / `patience` steps (4 at patience 25). A plan of fewer than 500
steps waits 25 scores a hundredth of the plan apart (75 steps on a plan of 341), and a longer one waits 100 steps (25
scores every 4 steps), where a plan of 5,000 or 50,000 steps used to wait 1,250 or 12,500. The 100 steps are
provisional, to be set from the stopping rules replayed on saved validation curves. A `validation_every_n_steps` you
set is kept as it is, and so is the wait it gives: patience 25 with a score every 10 steps waits 250 steps. The plan's
`patience_steps` gives the wait for your table. A model always returns the weights of its best validation score, so
the wait changes how long a model trains, not which of the weights it scored are kept.
With `bag` 1 and no validation channel, the model trains on four fifths of the training rows and is scored on up to
4,096 rows of the other fifth.

### Several tasks or a time series

One fine-tuned model can learn several regression tasks at once, such as one per store, product or series. Nori reads
each task from the context rows a prediction sends, so the fine-tuned model serves every task: predict a store by
sending that store's rows as context.

**One table with a task column.** Stack the tasks in one table with a column that names each row's task (numbers or
text). The task column is not a feature.

```python
hyperparameters = FineTuneHyperparameters(
    target_column="sales", model_name="ft-acme-stores-v1", task_column="store_id"
)
```

**Tables with their own targets.** List the tables in `tasks` instead of `target_column`. Each is a file or a folder of
the training data (a folder holds every file under it) with its own target, and may set its own `feature_columns`,
`task_column` and `time_column`. Every file belongs to exactly one table, and tables need not share columns.

```python
from synthefy import TaskTable

hyperparameters = FineTuneHyperparameters(
    model_name="ft-acme-v1",
    tasks=[
        TaskTable(path="sales.parquet", target_column="sales"),
        TaskTable(path="returns/", target_column="units_returned"),
        TaskTable(path="stores/", target_column="sales", task_column="store_id"),
    ],
)
```

A task column the files lack can come from folder names: `df.to_parquet("stores/", partition_cols=["store_id"])`
writes `stores/store_id=7/...`, and `task_column="store_id"` reads the 7 from there. A list of one table that is one
task is refused: that is `target_column`.

**Time series.** `time_column` names a timestamp, date or integer column. It makes each task of its table a time
series: the rows are sorted by it, every training step predicts the `query_rows` rows right after `context_rows` rows
of history, as a forecast does, and each task is validated and scored on its latest rows. A time repeated within a
task is refused.

```python
hyperparameters = FineTuneHyperparameters(
    target_column="load",
    model_name="ft-grid-v1",
    task_column="meter",
    time_column="hour",
    query_rows=24,
    context_rows={168: 1, 336: 1, 672: 1},
)
```

`context_rows` and `query_rows` count rows, as a number or as `{rows: weight}`, from which each step draws one count
with probability proportional to its weight. Set them to how the model will be used. `context_rows` is the history a
prediction sends, at least 16 rows. `query_rows`, which a time series needs, is how far past its context the furthest
prediction sits; the model forecasts that far only when every feature of a row is known that many rows before it,
such as lags of at least `query_rows` rows, which the job cannot check. Rows are not durations: with gaps in time, 24
rows can span more than 24 hours. Both settings apply to tabular tasks too, where `query_rows` only sets how many
targets a step scores, and a table's own settings override the job's.

Any of `tasks`, `task_column`, `time_column`, `context_rows` or `query_rows` makes a job of tasks, which differs from a
job on one table:

- It reads its tables into memory whole: at most 1,000,000 rows together, fewer above 100 features (the rows x
  features stay under 100,000,000). For a larger table of one task, leave all five unset; that job streams the table.
- The tasks need at least 1,000 training rows together. Each task needs, besides its held-out rows, at least 8 rows
  to validate on and 24 to train on in every model: 32 rows for a time series or with `bag` 1; with `bag` above 1
  each model of a tabular task validates on 1/`bag` of its rows (48 rows with `bag` 2, 64 with `bag` 8). A time
  series holds out at least its largest `query_rows`. The job names a task that has fewer.
- Each model trains on every task: a tabular task's rows outside its validation fold (`row_partition_strategy` does
  not apply), a time series' training rows with its own step draws. The models train one after the other on one GPU.
- `validation_data` holds the same tables at the same paths. A time series' rows there must all come after its
  training rows; it validates on the first 4,096 of them.
- Every model featurizes every task before it trains, and every validation score runs one forward pass per task, so
  thousands of tasks take hours before the first step.
- It has no plan before the job: `get_training_plan()` checks the tables, then raises `ValueError`, and `fine_tune()`
  attaches none; the job logs its plan once it has read the tables.
- `checkpoint_uri` is refused, and a model that fails fails the job.
- A task whose target never changes is refused; drop it.
- `tasks` travels as one hyperparameter value of at most 2,500 characters, about 15 time-series tables or 5 tables
  that each list 30 feature names. More tasks fit as one table with a task column.
- The report's `tasks` lists each task's table, rows and MSE before and after; `val_mse_before` and `val_mse_after`
  combine them with each task weighted by one over its MSE before, so every task counts by its relative change.

Only a newer version of the Algorithm reads these settings; an earlier one refuses them.

### Large tables: what a job reads and what it costs

A training step reads a number of rows set by the table's width, and a model takes the steps that read its share of
the table `n_passes_per_row` times, so a larger table takes more steps and more time. A model takes at least 100 steps,
which read about 790,000 rows of a table of 20 numeric feature columns and about 220,000 of a table of 100, from row
groups at random places in its share; a smaller share is read more than once. A model that `patience` (25 by default) or
`max_training_hours` stops sooner reads less: the report's `steps_run` gives the steps each model took and
`n_passes_per_row_completed` how many times the models read their rows.

The table gives, for every type the Algorithm offers, one pass of one `nori-6m` model over 50,000,000 rows of 20 numeric
feature columns: the hours it takes, its dollars at the on-demand SageMaker training prices for us-east-1 (AWS Price
List API, 2026-10-27), and the steps that make the pass.

| Instance type | GPUs | Rows per step, 20 columns | Rows per step, 100 columns | Seconds per step, 20 columns | Hours for the pass | Dollars for the pass | Steps for the pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ml.g5.xlarge | 1 | about 7,950 | about 2,210 | 2.085 (measured) | 5.3 | $7.5 | 6,293 |
| ml.g5.2xlarge | 1 | about 7,950 | about 2,210 | 2.085 (estimate) | 5.3 | $8.1 | 6,293 |
| ml.g6.xlarge | 1 | about 7,950 | about 2,210 | 2.897 (estimate) | 7.4 | $8.3 | 6,293 |
| ml.g6e.xlarge | 1 | about 7,950 | about 2,210 | not measured | not measured | not measured | 6,293 |
| ml.g6e.2xlarge | 1 | about 7,950 | about 2,210 | not measured | not measured | not measured | 6,293 |
| ml.g5.12xlarge | 4 | about 7,950 | about 2,210 | 2.085 (measured) | 5.3 | $38 | 6,293 |
| ml.g5.48xlarge | 8 | about 7,950 | about 2,210 | 2.085 (estimate) | 5.3 | $109 | 6,293 |
| ml.g6e.12xlarge | 4 | about 7,950 | about 2,210 | not measured | not measured | not measured | 6,293 |
| ml.g6e.48xlarge | 8 | about 7,950 | about 2,210 | not measured | not measured | not measured | 6,293 |

A narrower table, or fewer rows per model, takes less time a step; the plan's `seconds_per_step` gives the time for
your table.

The rows per step are the average over a model's steps, for feature columns of continuous values and a share of the
table above about 20,000 rows, so the steps times the rows per step is what a model that runs all its steps reads; one
step reads 4,605 to 11,290 rows at 20 columns and 1,268 to 3,153 at 100. The hours count the steps, the validation
scores between them (one every 4 steps, by default at patience 25 on a plan of 400 steps or more) and a few minutes
of reading and fitting. At 20 columns a score takes about 1.8 steps' time, so a score every 4 steps makes a pass about
40% longer than a score every hundredth of its steps did; patience can stop a model 100 steps after its best score.
`ml.g5.2xlarge` is an estimate taken from `ml.g5.xlarge`, and `ml.g5.48xlarge` one taken from `ml.g5.12xlarge`: the
same GPUs, one model per GPU. On `ml.g6.xlarge` a step and a validation score were timed together and their seconds
split between the two as on `ml.g5.xlarge`, so its row is an estimate as well. The `ml.g6e` types are not measured;
the report of a short job on one (its `seconds_per_step`) gives that type's figure. On a type with N GPUs a model's pass
takes the same time as on one GPU of the same kind, and the dollars are the whole instance's: N models make their passes
at once for that price. `nori-1-5`, the default base, takes 6.0 s a step on the same table, measured with four models
on `ml.g5.12xlarge`; every `ml.g5` type takes that figure, and its plan gives the hours and dollars there. It has no
`ml.g6` or `ml.g6e` figure yet, so on those types its plan gives the steps and rows per step and no hours or dollars.

The default job (`bag` 8, `"shard"`) splits the table among its eight models, and each reads its eighth once with
validation scores of its own, every 4 steps as in the table, so on a type with N GPUs it takes about ceil(8 / N) / 8 of
the hours and dollars in the table: up to about 5.3 h and $7.4 on `ml.g5.xlarge` and 1.3 h and $9.3 on
`ml.g5.12xlarge`, plus the few minutes of reading and fitting, and less when `patience` stops a model sooner. `bag=2` makes two passes over halves:
about the same hours and dollars on a one-GPU type, half of them on a 4-GPU or 8-GPU type. With
`row_partition_strategy="folds"` each model trains on every row outside its validation fold, seven eighths of the table
with `bag` 8, so each model's pass is seven times as long as under `"shard"`. The hours and dollars grow with
`n_passes_per_row`; `n_passes_per_row` below 1, or a smaller `n_steps_per_model`, reads part of each share.

`max_training_hours` (24 by default, at most 119.5) caps the job from its start, the read of the table included: each
model stops at it, and a model not started by then does not train (the report lists it in `unstarted_children`, and
`n_rows_unread` counts the rows of the models' shares that no model read). For a table whose pass takes longer, raise it
or set it to `None`; `max_runtime_s` follows it unless you set it (see Stopping and failures). The report's
`training_plan` gives the steps, hours and dollars the job worked out for your table and type before it trained, with a
warning when the cap ends the training before every planned step; with `patience` set (25 by default) its hours and
dollars are a ceiling, since a model can stop sooner.

A training model holds in host memory the row group it is reading (at most about 7 GB for the largest row group the job
accepts), its buffer of rows and about 2 GB besides: on the 50,000,000-row table of 20 columns, in row groups of
1,000,000 rows, a model peaked at 3.5 GiB. Every type the Algorithm offers has at least 16 GiB of host memory per GPU,
so every type holds any table within the row group limits up to 10,000 feature columns, and the choice of type is one of
time and price. A row group above 2,000,000 rows or 200,000,000 cells fails the job before it reads any row (see Your
table). The validation channel is used whole up to 4,096 rows; a larger one is sampled to 4,096 rows.

### The plan before the job

```python
with SynthefyNoriFineTuner(region_name=REGION) as tuner:
    plan = tuner.get_training_plan(request)  # the request you would pass to fine_tune()
    print(plan.get_summary())
```

`get_training_plan` gives the plan the job would make for your table and instance type, without starting it: the models
and the rows each trains on, the steps, the share of the table they read, the training hours, their dollars at the
prices listed under Large tables, and a warning for a share the steps leave unread or a `max_training_hours` that ends
the training early. It reads only the end of each Parquet file, the last 256 KiB or the whole footer when that is
longer: well under a second for one file, and a few seconds for a folder of a thousand files in the bucket's AWS Region;
elsewhere, about the time to download 256 KiB of each file. The rows per step depend on the width of a featurized row,
which the plan estimates from the feature columns (`rows_per_step_is_estimate` is true): the steps, hours and dollars
can differ from the job's own plan, which the job logs and the report's `training_plan` holds, by the difference in that
width. It raises `ValueError` for a `target_column` or `feature_columns` entry the training table lacks, a validation
table with other columns, a file that names a column twice, a channel with no Parquet file or with a file of another
extension, a `.parquet` or `.pq` file that cannot be read as Parquet, and a table too small for each model's 1,000 rows
(on the fixed-steps version below, for fewer than 1,000 training rows in the table); also for a footer above 256 MiB,
which `fine_tune()` submits without a plan.

`fine_tune()` makes the same plan, logs its summary and returns it as the job's `training_plan`; it refuses the table
before creating the job for the same reasons. Any other failure to make the plan (an AWS error such as missing
credentials or denied access, or a footer above 256 MiB) logs a warning, and the job is submitted without a plan. It
warns when the plan's training hours leave the job less than the 30 minutes it needs besides training within
SageMaker's 120-hour limit on a training job and the request has no `checkpoint_uri`.

An Algorithm version that trains each model for a fixed number of steps reads none of `row_partition_strategy`,
`n_passes_per_row`, `max_training_hours` and `validation_every_n_steps`: on that version `fine_tune()` refuses a request
that sets one, or that sets `patience=None`, naming each. Otherwise it trains each model for `n_steps_per_model` steps
(100 by default) or 900 seconds, whichever ends first, and the plan counts the steps that fit. `fine_tune()` logs a
warning naming the Algorithm and its steps per model, and the plan warns when the 900 seconds stop each model before
those steps. `patience` counts validation scores on that version too, 25 by default, and its wait has no 100-step
limit. On that version `fine_tune()`
sends a `max_runtime_s` you set as it is, and one left unset as ceil(`bag` / N) x 1,350 + 1,800 s on an instance type
with N GPUs, at most 432,000 s (12,600 s for `bag` 8 on a one-GPU type). A newer version of the Algorithm reads all four
settings. When the Algorithm cannot be described, the job gets only the settings every version of the Algorithm reads,
and a request that sets `row_partition_strategy`, `n_passes_per_row`, `max_training_hours`, `validation_every_n_steps`,
`n_steps_per_model` or `patience=None` is refused, naming each.

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

table = pd.read_parquet("train.parquet")
features = [column for column in table.columns if column != TARGET]  # the order the job trains on
boto3.client("s3", region_name=REGION).upload_file("train.parquet", BUCKET, "nori/data/train/train.parquet")

with SynthefyNoriFineTuner(region_name=REGION) as tuner:
    request = FineTuneJobRequest(
        algorithm=ALGORITHM_ARN,
        role_arn=ROLE_ARN,
        training_data=f"s3://{BUCKET}/nori/data/train/",
        output_path=f"s3://{BUCKET}/nori/output/",
        hyperparameters=FineTuneHyperparameters(
            target_column=TARGET, model_name="ft-acme-churn-v1", bag=2, n_steps_per_model=30
        ),
    )
    print(request.instance_type, request.max_runtime_s)  # ml.g5.xlarge 88200, from max_training_hours (24)
    job = tuner.fine_tune(request)
    print(job.job_name)  # watch it from another session
    job = tuner.wait(job.job_name)  # logs the report's summary once the job completes
    print(job.status, job.secondary_status, job.status_message, job.billable_seconds)
    if job.status == "Failed":
        raise SystemExit(job.failure_reason)
    report = tuner.get_training_report(job.job_name)
    print(report.val_mse_before, report.val_mse_after, report.cut_short)
    endpoint = tuner.deploy(job)  # asynchronous endpoint on ml.g5.xlarge; returns once it is InService
    try:
        with tuner.create_client(endpoint, s3_staging_uri=f"s3://{BUCKET}/nori/invocations/") as client:
            context = table.sample(n=min(len(table), 2000), random_state=0)
            new_rows = pd.read_parquet("new.parquet")[features]  # rows the job never saw: same columns, same order
            print(client.predict(X_train=context[features], y_train=context[TARGET], X_test=new_rows,
                                 categorical_columns=None))
            assert client.last_model_version == report.model_version
    finally:
        tuner.undeploy(endpoint)
```

`bag=2, n_steps_per_model=30` is a small run. With capacity available, the flow takes about 10 minutes: the job 3 to 5 minutes, the model package under a minute and the endpoint 4 to 6 minutes. Capacity waits add to that (below). The endpoint does not store your table, so every `predict()` still sends context rows and uploads
them to your staging prefix: the flow caps them at 2,000 to keep each request small. Send the training
feature columns in training order, without the target, with `categorical_columns=None`, so a text column raises in the
client instead of being encoded in a way the job never saw. `last_model_version` equal to `report.model_version` shows
the fine-tuned model answered; the staging prefix works as in Amazon SageMaker Usage above. Expire `nori/invocations/`
and `nori/output/ft-acme-churn-v1/async/` after at least one day, never `nori/output/` as a whole: the model package reads
its model from there. `undeploy()` deletes the endpoint, its endpoint configs and the SageMaker model, and keeps the
package for a new deploy. The endpoint bills every hour from creation until `undeploy()`, whether or not you send
requests; prices per instance type and region are on [SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/).

Other hyperparameters, with defaults: `base_model` (`nori-1-5`, `nori-6m` or `nori-100m`; a model fine-tuned from
`nori-1-5` keeps no context cache, like its base; `fine_tune()` raises for a base the Algorithm's version does not
list, and an unset `base_model` on a version published before `nori-1-5` trains that version's default, `nori-6m`); `feature_columns` (all but the target, in
table order); `bag` (8: how many models the job trains, 1 to 8; the fine-tuned model averages them, and models train
one per GPU at a time); `row_partition_strategy` (`"shard"`: each model trains on its own 1/`bag` of the training rows;
`"folds"`: each model trains on every training row outside its validation fold); `n_passes_per_row` (`None`, one read of
each row: how many times each model reads each training row it trains on, up to 100; at 20 feature columns about 13%
of the rows a pass reads are prediction targets, so more passes make more rows targets, about 1 - 0.87^passes of them
(1 pass 13%, 2 passes 24%, 5 passes 50%, 10 passes 75%), as the plan's `fraction_of_rows_as_targets` shows);
`n_steps_per_model` (`None`: the exact number of training steps each model takes; not together with `n_passes_per_row`);
`patience` (25: the validation scores in a row without a better one that stop a model, a wait of `patience` x
`validation_every_n_steps` training steps; `None`: a model takes every step of its plan); `max_training_hours` (24.0, at most 119.5: the cap, in hours from the job's start, on reading the
table and training the models; `None`: no cap);
`validation_every_n_steps` (the steps per model divided by 100, rounded down, at most 100 divided by `patience`, rounded
down, and at least 1); `lr` (1e-5); `seed` (0).
`model_name` starts with `ft-`, then 1 to 60 lowercase letters, digits and `-`
starting with a letter or digit, and cannot be a released model's name. End it with a letter or digit as well:
`deploy()` names the endpoint after it, and SageMaker refuses an endpoint name that ends with `-`.

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
Without a validation channel the job sets aside up to 4,096 rows (about 10 % of a smaller table) before training, rows
no training step sees, so both scores are out of sample. With one, its rows (up to 4,096) are scored and also steer the
training, so `val_mse_after` runs a little optimistic there. `validation_source` names the mode, `n_val_rows` counts
the rows scored, `n_rows` the rows of the training table and `n_rows_used` the rows training could draw from: the table
minus the report rows, when the job sets them aside. With `bag` 1 and no validation channel, the model trains on four
fifths of these rows and is scored on up to 4,096 rows of the other fifth.

`rows_visited` counts the rows the models' training steps used, a row used in two passes twice; each curve in
`child_curves` has its model's own count. `fraction_visited` sets it against the shares of every model the plan set,
those that stopped early, failed or did not start included (1.0 for one pass over each), `n_passes_per_row_completed`
gives the same as passes per row, and `steps_per_pass` is the number of steps the longest of the models' passes takes.
`fraction_of_rows_as_targets` is the share of the training rows used as a prediction target in at least one training
step, the steps after a model's best validation score included. `validation_interval` is the number of training steps
between two validation scores in a model's curve (`validation_every_n_steps`), so `patience` scores without a better
one are `patience` times that many steps (the plan's `patience_steps`). `read_seconds` is the time the job's first read of the training
table took, and `seconds_per_step` the seconds a training step took with its share of the validation scores, the figure
the plan's `seconds_per_step` estimates.

`stop_reason` says what ended the training, the first that applies of `stop_request` (a stop, below),
`max_training_hours`, `non_finite_loss`, `patience` (every model stopped early) and `plan_done` (otherwise: every model
took its steps, or some did and `patience` stopped the rest); each curve in `child_curves` has its model's own. `bag`
and `row_partition_strategy` are the values the job used, `unstarted_children` the models `max_training_hours` left
untrained, and `n_rows_unread` the rows of the models' shares that no model read. `training_plan` is what the job worked
out before it trained, with its warnings. `report.get_summary()` gives these figures in a few lines, and `wait()` logs
it when a job completes.

`workers` is the number of worker processes that trained the models at the same time (one per GPU on a type with
several GPUs) and `gpus` the number of GPUs they trained on: both are 1 on a one-GPU type. An error in one model fails
that model alone, on any type: `failed_children` lists it, its curve in `child_curves` has `failed` true and the end
of its error message in `failure`, and the fine-tuned model averages the other models. On a type with several
GPUs, running out of host memory can end every unfinished model on one GPU at once, and `failed_children` lists them
all. A job whose models all fail ends `Failed`, and `failure_reason` names them.

`job.latest_metrics` and the console charts follow the models' validated steps as they are printed, several
models side by side on a type with several GPUs (the `child` metric names each point's model): progress, not the
result. Once the job ends, `job.latest_metrics` holds the last step of whichever model finished last. Each model's
`val_mse` in `get_training_curves()` is that model's own validation score, the one that decides when it stops, not the
report's score, so compare the report's two numbers instead.

### Several models and endpoints

A job trains one model, and an endpoint serves the one model it was deployed from: a request naming another model is
refused. For several models, run one job per `model_name` and deploy each:

```python
endpoints = {}
for name, target in [("ft-acme-churn-v1", "churn"), ("ft-acme-ltv-v1", "ltv")]:
    hp = FineTuneHyperparameters(target_column=target, model_name=name)
    # the copy derives its max_runtime_s from these hyperparameters
    per_model = request.model_copy(update={"hyperparameters": hp})
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
stops the job the same way, so it ends `Stopped`, not `Failed`; the default is `max_training_hours` x 3,600 + 1,800 s
(88,200 s with the default 24 hours), and 432,000 s (120 hours, the limit SageMaker sets by default) when
`max_training_hours` is `None`, except on the fixed-steps version (see The plan before the job). At `max_training_hours`
the job itself ends the training and still writes its model and report, so leave `max_runtime_s` to follow it: a request
copied with new hyperparameters derives `max_runtime_s` from them, and a `max_runtime_s` you set is sent as it is.
Unless the Algorithm's version trains each model for a fixed number of steps, `fine_tune()` and `get_training_plan()`
refuse one below `max_training_hours` x 3,600 + 1,800 s, which would stop the job before the cap ends its training.

A job given `checkpoint_uri`, an S3 folder ending in `/` such as `s3://BUCKET/nori/output/checkpoints/<name>/`, saves
each model it finishes there, and the progress of a model it is training every 15 minutes or 500 steps and when it
stops. After a stop or a failure, `tuner.resume(job)` starts a new job with that job's request under a new name
(`instance_type` and `max_training_hours` may change; the fixed-steps version refuses `max_training_hours`, see The plan
before the job) that trains only the models not saved in the folder, continuing a model from its last saved progress,
and the report marks the finished ones with `ChildCurve.restored` and counts each step once. The resumed job keeps the
job's `model_name`; the job's tags are not copied. `resume()` makes the plan as `fine_tune()` does, logs it and returns
it as the new job's `training_plan`: the plan of the whole request, so the most the new job takes, since that job
trains only what the folder does not hold. `resume()` refuses a job still running, a job without
`checkpoint_uri` or whose folder holds no checkpoint file, a job created with managed spot training, a VPC or a volume
KMS key (a request carries none of them), and a completed job that left no model to train or whose training report
cannot be read. `fine_tune()` refuses a `checkpoint_uri` folder that holds any file,
naming the models and steps it holds and the job that wrote it, before creating the job: resume that job, give another
`checkpoint_uri`, or pass `fine_tune(request, overwrite=True)`, which deletes the folder's checkpoint files and then
creates the job. `overwrite=True` deletes nothing, and refuses, when the folder holds a file the job does not write,
more than 1,000 files, or the files of a job still running; a folder that holds only files a job does not write is
refused as not a checkpoint folder. A folder resumes only on the Algorithm version that wrote
it and with the same table and hyperparameters: a job refuses a folder written
by another version, with other hyperparameters or for another table, naming what differs, and an edited table needs an
empty folder. Run one job per folder at a time: `fine_tune()` and `resume()` refuse, with nothing deleted and no job
created, while another `InProgress` or `Stopping` training job of the account saves to the folder. The folder holds
each finished model and a few saved states of each
model, at most about 0.4 GB per model for `nori-6m` (1.4 GB for `nori-1-5`, 6.1 GB for `nori-100m`). A job given the
folder first copies all of it. A job deletes saved states it does not need, so the execution role needs
`s3:DeleteObject` on the folder. A job that completes with every model saved deletes the saved states and keeps the
finished models, about 0.025 GB per model for `nori-6m` (0.085 GB for `nori-1-5`, 0.38 GB for `nori-100m`); a stopped or
failed job keeps saved states too, up to that 0.4 GB, 1.4 GB or 6.1 GB per model. S3 stores them until you delete the
folder: delete it once a job has completed. A job that cannot continue a saved state, usually because the table changed
since it was saved, fails naming the model, the state and what differs, and leaves that model's files in the folder
unchanged; to train from the start, use an empty folder. Without `checkpoint_uri` a job keeps no checkpoints: a new job
starts again from the base.

| You see | What to do |
| --- | --- |
| `deploy()` or `get_training_report()` raises `FineTuningError` saying the `Stopped` job has no model artifact | Nothing had improved on the base when the job stopped, so it wrote no model. Start a new job. |
| `Failed`, and `failure_reason` names your table or settings | The job refused the input once its instance was up. Fix it and start a new job; `tuner.get_logs()` has any traceback. |
| `Failed`, and `failure_reason` contains `killed while` | The job's process was killed, most likely because the instance ran out of memory. Start a new job on an instance type with more memory, or write the table with smaller row groups (see Your table). |
| `InProgress` in `Pending` for a long time with the message `Training job waiting for capacity`, or `Failed` with `CapacityError:` | AWS has no spare capacity for that type. The wait is unbilled and can last hours. Stop it and start a new job on another type. |
| `fine_tune()` raises `FineTuningError` saying a new job needs an empty folder | An earlier job saved models in the `checkpoint_uri` folder. Call `tuner.resume(<the job it names>)` to continue that job, give another `checkpoint_uri`, or pass `overwrite=True` to delete its checkpoint files and train every model from the start. |
| `fine_tune()` or `resume()` raises `FineTuningError` saying a training job saves to the checkpoint folder | That job, for example one an earlier `resume()` started, still runs on the folder. Wait for it to end, or `tuner.stop()` it, then call again; or give another `checkpoint_uri`. |
| `Failed`, and `failure_reason` says the checkpoint folder holds another job's files and `resume` is not set | The folder was not empty when the job started. Call `tuner.resume(<the job it names>)` to continue that job, or start the job again with another `checkpoint_uri` or `overwrite=True`. |
| `fine_tune()` raises `FineTuningError` saying SageMaker started the job but reading its status failed | The job runs and bills. Call `tuner.get_status(error.job_name)` to follow it, not `fine_tune()` again, which starts a second job. |
| `wait()` raises `APITimeoutError` | It waited the time limit plus 7,200 s. The job still runs: call `wait()` again, or `stop()`. |
| `deploy()` raises `FineTuningError` naming `InsufficientInstanceCapacity` | AWS found no instance of that type in time. Call `tuner.undeploy("ft-acme-churn-v1")`, then `tuner.deploy(job, instance_type=...)` with another endpoint type; it reuses the model package. |
| `deploy()` raises `APITimeoutError` | The endpoint was not InService within `timeout_s` (3,600 s by default); it is left in place, still `Creating` and billing. Call `tuner.undeploy("ft-acme-churn-v1")`: it waits for the endpoint to leave `Creating` (up to its own `timeout_s`, 3,600 s by default), then deletes it. Then deploy again, on another endpoint type (`ml.g5.2xlarge` or `ml.g6.xlarge`) if it keeps happening. |
| `undeploy()` raises `APITimeoutError` saying the endpoint is still `Creating` | SageMaker cannot delete an endpoint until it leaves `Creating`, `Updating`, `SystemUpdating` or `RollingBack`, and `undeploy()` waited its `timeout_s` (3,600 s by default) for that. Call `undeploy()` again later, or pass a larger `timeout_s` to wait longer. |

### Least-privilege IAM

For production, replace `AmazonSageMakerFullAccess` with two policies that grant each identity only what it uses; with
them the bucket can have any name. Save this permission document as `execution-policy.json`, replacing `BUCKET`,
`REGION`, `ACCOUNT_ID` and `ALGORITHM_ARN`. It lets the job read `nori/data/`, write `nori/output/` and delete what it
saves under `nori/output/checkpoints/`, lets the endpoint read `nori/invocations/` and write under `nori/output/`, and
lets SageMaker describe the Algorithm and your model package
(named after the job, `nori-ft-...` unless you set `job_name`; if you do, use its prefix here and in the EventBridge rule
under Alarms) as the execution role.

```json
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"], "Resource": "arn:aws:s3:::BUCKET"},
  {"Effect": "Allow", "Action": "s3:GetObject", "Resource": "arn:aws:s3:::BUCKET/nori/data/*"},
  {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
   "Resource": ["arn:aws:s3:::BUCKET/nori/output/*", "arn:aws:s3:::BUCKET/nori/invocations/*"]},
  {"Effect": "Allow", "Action": "s3:DeleteObject", "Resource": "arn:aws:s3:::BUCKET/nori/output/checkpoints/*"},
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
not found, which the client relies on while it waits for a result, and lets `fine_tune()` check a one-object channel
and list the `checkpoint_uri` folder; `s3:GetObject` reads the folder's records, and `s3:DeleteObject` lets
`fine_tune(request, overwrite=True)` empty it. `sagemaker:ListTrainingJobs` lets `fine_tune()` with a
`checkpoint_uri`, and `resume()`, find a job that still saves to the folder; both refuse without it.
With `kms_key_id`, the key's policy must let both identities use it.

```json
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Resource": "*", "Action": [
    "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:ListTrainingJobs", "sagemaker:StopTrainingJob",
    "sagemaker:DescribeAlgorithm",
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
