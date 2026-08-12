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

   # model is required -- name a size: "nori-30m" (~29.2M) or "nori-6m" (~6M base).
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

By default each categorical column becomes a single column of **ordinal codes**
(categories from `X_train` in sorted order — the model's own server-side
convention): a value seen only in `X_test` maps to `-1`, and a missing value
(NaN) stays NaN for server-side imputation. Pass
`categorical_encoding="onehot"` for the previous one-hot behavior (indicator
columns per category; missing values get their own indicator; unseen values map
to an all-zeros group). Datetime columns and categorical columns with more than
`max_categorical_cardinality` (default 100) distinct training values are dropped
with a warning; `timedelta` columns are unsupported and raise (convert them to a
number or string first). Numeric columns (including `bool`) pass through unchanged,
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
`"nori-30m"` (→ `synthefy/nori-30m`) or `"nori-6m"` (→ `synthefy/nori-6m`). The
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

Install the optional AWS transport and invoke a named real-time endpoint:

```bash
pip install "synthefy[aws]"
```

```python
from synthefy import SynthefyNoriClient

client = SynthefyNoriClient(
    mode="sagemaker",
    model="nori-30m",
    endpoint_name="nori-30m-prod",
    region_name="us-east-1",
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
not accept AWS access keys. `model=` and `endpoint_name=` are required: the
endpoint selects the deployed model specification, while the request model is
checked against it so a routing mistake fails closed. Backend selection is always
explicit; installing another package never changes where a request runs.

SageMaker's request is the same Nori JSON contract used by the hosted transport,
sent through `InvokeEndpointWithResponseStream` with `application/json` for all three
models. The server emits 15-second heartbeat chunks and one final JSON result, which
the client buffers into the normal `predict()` return value. This lets large 30M
requests use SageMaker's streaming processing window (up to eight minutes) instead of
the regular invocation's 60-second limit. Container errors retain
their original status/message through the normal Synthefy exception hierarchy;
AWS credential, signing, region, quota, and throttling errors remain native AWS
SDK exceptions. The constructor timeout is SageMaker's per-read inactivity timeout,
not a total stream deadline. Set timeout/retries on the constructor. HTTP-only
`extra_headers=` are rejected for SageMaker. Per-call `timeout=` is ignored with
a warning.

Streaming does not increase AWS Marketplace's 25,000,000-byte SageMaker endpoint
request-body limit. The client checks the final encoded JSON before invoking the
endpoint. It does not split oversized tables because every query must use the same
complete in-context training set, so splitting can change the prediction. The planned
large-input path is an explicit S3-backed SageMaker Asynchronous Inference API rather
than a silent fallback; [AWS documents](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html)
payloads up to 1 GB and processing up to one hour for that service.

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
~6M-parameter model — is what exhausts GPU memory on a big table. `memory_policy=` decides
what to do about it. Omit it and the defaults handle almost every request.

```python
# A preset...
preds = client.predict(X_train, y_train, X_test, memory_policy="exact")        # never quantize
preds = client.predict(X_train, y_train, X_test, memory_policy="max_context")  # fit the largest table

# ...individual fields...
preds = client.predict(X_train, y_train, X_test, memory_policy={"cache_dtype": "int8"})

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
| `rung` | which fallback served it — `resident_bf16` → `resident_int8` → `offload_bf16` → `offload_int8` → `plain_loop` → `no_cache` |
| `est_cache_gb` / `resident_gb` | the cache's full-precision size, and what stayed in GPU memory |
| `query_chunk` | query rows per forward pass |
| `dropped_context_rows` | context rows discarded to fit — **the one accuracy loss worth checking**, `0` unless subsampling engaged |
| `clamped` | fields the server capped (host-RAM budgets only) |
| `notes` | remarks about the policy you sent, e.g. a budget that cannot take effect |

**Only the int8 rungs trade accuracy**, and they are reached only when full precision
will not fit. `offload_*` moves bytes to host RAM rather than approximating, so it is
bit-identical to staying resident. Set `memory_policy={"allow_subsample": False}` to turn a
silently shortened context into an error instead.

One field behaves differently over the network: **`elements_budget`**. The cache is only
built when the query set spans more than one chunk, and at default settings that needs
far more query rows than the hosted request-body limit (~64 MiB) allows — so lowering
`elements_budget` is what lets a hosted request reach the cached path at all.

In `mode="local"` the same argument works, but needs `synthefy-nori >= 0.13.0`; older
builds raise `ImportError` with an upgrade hint. `last_memory_report` stays `None`
locally — use `NoriRegressor` and read `memory_report_` if you need it there.

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
labels = client.predict(X_train, y_train, X_test, discretize="snap-mean")
labels = client.predict(
    X_train, y_train, X_test,
    discretize="snap-mean",
    categorical_levels=[1, 2, 3, 4, 5],   # the full scale, if the context may under-cover it
)
```

Discretization is **strictly opt-in** — nothing is snapped unless you ask.
`categorical_levels` is the set of values the target can take (numeric; order
and duplicates don't matter); it defaults to the distinct values of `y_train`, which is
leak-safe. A `NaN` prediction stays `NaN` rather than becoming a confident
label.

Capability differs by mode:

- **Remote**: the hosted endpoint returns point predictions (the distribution
  mean), so the supported strategy is `discretize="snap-mean"` — the nearest
  level to the point prediction, computed client-side and identical to local
  `"snap-mean"`. Other strategies raise a `ValueError` pointing here.
- **Local** (`pip install "synthefy-nori"`, with a `synthefy-nori` recent
  enough to ship `synthefy_nori.discretize`): the full strategy set is
  forwarded — `"map-cell"` (accuracy-optimal), `"median-cell"` (MAE-optimal),
  `"snap-mean"` (QWK), `"snap-median"`, `"expected-level"`, `"prior-match"`.
  Choose by the metric you are scored on; see the `synthefy-nori` docs. An
  older `synthefy-nori` raises an `ImportError` with an upgrade hint.

If your task is scored by squared error / R², don't discretize — the
continuous mean is already optimal for those metrics.

## API Reference

### SynthefyNoriClient (Tabular Regression)

- `SynthefyNoriClient(api_key=None, *, mode="remote", timeout=300.0, max_retries=2, base_url=..., endpoint=..., model, user_agent=None, endpoint_name=None, region_name=None)` — `model` is **required everywhere** and accepts the three released Nori variants (`nori-6m`, `nori-30m`, and `nori-30m-thinking-medium`) or an explicit custom HTTP slug; there is no `None`/default model path. SageMaker uses response streaming for all three so large 30M requests can run beyond the regular-response limit while `predict()` still returns one normal result.
  - `mode`: `"remote"` (hosted, default), `"local"` (in-process via
    `synthefy-nori`), or `"sagemaker"` (a named SageMaker endpoint using the AWS
    credential chain).
  - `api_key` (remote mode) falls back to the `SYNTHEFY_NORI_API_KEY`
    environment variable. Not required in local mode.
  - Hosted Nori is reached by gateway slug — that is the path Synthefy meters,
    rate-limits and grants per key. To target a single-model endpoint you host
    yourself, pass your own `base_url`/`endpoint` and an explicit custom model slug.
- `predict(X_train, y_train, X_test, task="regression", *, output_type="mean", quantiles=None, text_columns=None, svd_dim=128, embedder="minilm", text_device="auto", timeout=None, extra_headers=None) -> List[float]`
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
    Feature columns must be numeric; DataFrame `X_test` is aligned to `X_train`
    by column name; non-numeric columns are encoded (fit on `X_train`;
    `categorical_encoding="ordinal"` by default, `"onehot"` available);
    missing values (NaN) are imputed server-side.
  - `max_categorical_cardinality` (default 100): non-numeric columns with more
    distinct training values than this — and datetime columns — are dropped with
    a warning instead of encoded.
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
    remote mode supports `discretize="snap-mean"`, local mode the full
    strategy set of the installed `synthefy-nori`.
- `mode`: the explicitly selected execution mode.
- `close()` / context manager support (`with SynthefyNoriClient(...) as client:`).

### Exception Hierarchy

Import these exceptions from `synthefy.errors`; all inherit from `SynthefyError`:

- `APITimeoutError`: Request timed out
- `APIConnectionError`: Network/connection issues
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
