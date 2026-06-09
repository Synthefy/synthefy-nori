# Design: `SynthefyTabularClient` and consolidating the tabular SDK into `synthefy`

**Status:** Draft / for review
**Author:** (design doc — implementation not yet started)
**Scope:** How to expose a `SynthefyTabularClient` that is importable as
`from synthefy import SynthefyTabularClient`, reuses the current regressor
signature, and folds this repo's nascent tabular SDK into **one** published
package, [`synthefy`](https://pypi.org/project/synthefy/).

---

## 1. Goal

Today the tabular foundation model ships as its own distribution,
`synthefy-tabular`, whose public surface is a single scikit-learn–style
estimator, `SynthefyTabularRegressor` (see
[`src/synthefy_tabular/api.py`](../../src/synthefy_tabular/api.py)). Separately,
the company already publishes [`synthefy`](https://pypi.org/project/synthefy/)
(v3.0.0) — a **remote API client** for time-series forecasting that exposes
`SynthefyAPIClient` / `SynthefyAsyncAPIClient`.

The objective is to **consolidate the tabular capability into a single
`synthefy` package** under a new client class, so a user can write it **the same way they import the existing clients** — from
the `synthefy.api_client` module:

```python
from synthefy.api_client import SynthefyTabularClient
# ...exactly mirroring today's:
from synthefy.api_client import SynthefyAPIClient, SynthefyAsyncAPIClient
```

and get tabular regression through the same package they already use for
forecasting — while **reusing the current regressor signature** unchanged. (The
class is also re-exported at the top level, so `from synthefy import
SynthefyTabularClient` works too.)

**Hard requirements:**
1. **One distribution.** There must be **no separate `synthefy-tabular`
   package** that `synthefy` depends on; the model code is absorbed into
   `synthefy` itself.
2. **Dual-mode inference.** Tabular inference must support **both a local mode**
   (in-process model) **and a remote mode** (hosted endpoint) as first-class,
   supported paths — not one with the other as a future stretch. Both expose the
   same signature; they differ only in where the forward pass runs.

This document covers: (a) what "compatible" means for the current signature
(scikit-learn vs AutoGluon), (b) the shape of `SynthefyTabularClient`, and
(c) how to merge a heavy local-inference library into one package **without**
forcing torch on every `synthefy` user.

---

## 2. Current state (the two things being merged)

### 2.1 `synthefy-tabular` (this repo) — local model, sklearn-style

```python
from synthefy_tabular import SynthefyTabularRegressor

model = SynthefyTabularRegressor()          # downloads public HF weights on first use
model.fit(X_train, y_train)                 # "fit" stores the in-context rows
pred = model.predict(X_test)                # one forward pass
pred = model.predict(X_test, output_type="median")
```

- **Contract:** `fit(X, y)` then `predict(X, *, output_type="mean"|"median"|"mode", quantiles=None)`.
  Numpy/array-in, numpy-out. This mirrors the **scikit-learn estimator
  protocol** and the **TabPFN** `TabPFNRegressor.predict` contract.
- **Runtime:** local PyTorch inference; weights pulled from the Hugging Face Hub
  (now public — no token needed); bundled JSON inference configs.
- **Weight:** heavy dependency set (`torch`, `scikit-learn`, `scipy`, `pandas`,
  `huggingface-hub`, …).
- **Also available:** a hosted **Baseten endpoint** serving the same model over
  HTTP with a JSON contract roughly
  `{"task","X_train","y_train","X_test"} → {"task","predictions"}`
  (HTTP 400 on invalid input). This is the "remote backend" that already exists.

### 2.2 `synthefy` (PyPI v3.0.0) — remote API client, DataFrame-style

```python
from synthefy import SynthefyAPIClient

with SynthefyAPIClient(api_key="...") as client:
    forecast_dfs = client.forecast_dfs(
        history_dfs=[history_df], target_dfs=[target_df],
        target_col="sales", timestamp_col="date", model="sfm-moe-v1",
    )
```

- **Contract:** thin HTTP client; `api_key` auth; pandas-DataFrame in/out; sync
  **and** async variants.
- **Weight:** lightweight (HTTP + pandas); no ML framework. `import synthefy` is
  fast and torch-free.

**The central tension:** `synthefy` is a small networked client; the tabular
model is a large local-inference library. Folding them into one distribution
must not turn `import synthefy` into a multi-hundred-MB torch import for users
who only want the forecasting API. Section 5 shows how a single package still
keeps torch optional.

---

## 3. Is the current signature "AutoGluon compatible"? (and what that means)

Short answer: **No — the current signature is *scikit-learn* compatible, which
is a different convention from AutoGluon.** "Compatible" is not a single thing;
it means "follows the input/output protocol that a given ecosystem's tooling
expects." There are two relevant protocols:

### 3.1 scikit-learn protocol (what we have, and what TabPFN uses)

- Construct an estimator, then `fit(X, y)` where `X` is a 2-D feature
  array/DataFrame and `y` is a **separate** target array.
- `predict(X)` returns an array aligned to `X`'s rows.
- Enables drop-in use in `sklearn.pipeline.Pipeline`, `cross_val_score`,
  `GridSearchCV`, etc., and makes us a **drop-in replacement for
  `TabPFNRegressor`**.

Our `SynthefyTabularRegressor` already satisfies this (the `output_type` /
`quantiles` keyword-only extensions are additive and don't break the protocol).

### 3.2 AutoGluon protocol (what we do *not* match)

AutoGluon's `TabularPredictor` is **DataFrame-and-label-column** centric, not
`(X, y)` centric:

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label="target", problem_type="regression")
predictor.fit(train_data)            # train_data is a DataFrame that INCLUDES the label column
preds = predictor.predict(test_data) # DataFrame in, pandas Series out
```

Key differences from the sklearn/our convention:

| Aspect | scikit-learn / ours / TabPFN | AutoGluon `TabularPredictor` |
|---|---|---|
| Target | separate `y` passed to `fit(X, y)` | a **column name** (`label=`) inside the training DataFrame |
| `fit` input | `X` (features) + `y` (target) | one DataFrame containing features **and** label |
| `predict` input | `X` features only | a DataFrame (label column optional/ignored) |
| `predict` output | numpy array | pandas `Series` (or DataFrame) |
| Problem type | inferred by the wrapper / fixed per class | explicit `problem_type` arg (`regression`/`binary`/`multiclass`/`quantile`) |
| Quantiles | `predict(..., output_type="quantiles", quantiles=[...])` | `problem_type="quantile"` + `quantile_levels=[...]`; `predict` returns one column per level |

So an AutoGluon user's muscle memory (`TabularPredictor(label=...).fit(df)`)
will **not** work against our estimator, and vice-versa. Neither is "more
correct" — they're different ecosystem conventions.

### 3.3 Recommendation on compatibility

1. **Keep the scikit-learn/TabPFN signature as the canonical core.** It's the
   broadest-reach standard, it makes us a TabPFN drop-in, and it composes with
   the entire sklearn ecosystem. `SynthefyTabularClient` exposes exactly this
   signature (Section 4).
2. **AutoGluon parity is out of scope** (decided — see §7). We will not build an
   AutoGluon-style `label=`/DataFrame adapter. The comparison above stands only
   to answer the question "is the current signature AutoGluon-compatible?" — it
   is not, and that's fine; we standardize on the sklearn/TabPFN convention.

---

## 4. `SynthefyTabularClient` design

### 4.1 What it is

A façade in the `synthefy` namespace that **wraps** the existing regressor and
gives `synthefy` users a tabular entry point consistent with the package's
client-oriented naming (`SynthefyAPIClient`, `SynthefyAsyncAPIClient`,
`SynthefyTabularClient`). It reuses the current signature verbatim — it does
**not** invent a new prediction contract.

Per hard requirement #2, `SynthefyTabularClient` supports **both** inference
modes as first-class paths, selected by a `backend` argument:

- `backend="local"` — runs the regressor in-process (requires the `tabular`
  extra; see Section 5). Best for offline use, no per-call network, full control
  of the weights/device.
- `backend="remote"` — calls the hosted tabular endpoint with an `api_key`,
  mirroring `SynthefyAPIClient` (no torch needed). Best for a lightweight
  install, no local GPU, centrally-managed model versions.

Both backends present the **identical** `fit`/`predict`/`forecast` signature, so
switching between them is a one-argument change with no other code edits. A
lightweight `pip install synthefy` can already do remote inference;
`pip install synthefy[tabular]` additionally unlocks local inference — one
class, two backends, one signature.

### 4.2 Surface (sketch — not final)

```python
from synthefy.api_client import SynthefyTabularClient

# Local backend (requires `pip install synthefy[tabular]`)
client = SynthefyTabularClient(backend="local")          # or backend="auto"
client.fit(X_train, y_train)
y = client.predict(X_test, output_type="mean")           # same signature as the regressor

# Remote backend (lightweight install; hits the hosted endpoint)
client = SynthefyTabularClient(backend="remote", api_key="...")
client.fit(X_train, y_train)                             # stores context client-side
y = client.predict(X_test)                              # one HTTP call: X_train/y_train/X_test -> predictions

# One-shot convenience (no separate fit), array- or DataFrame-friendly
y = SynthefyTabularClient(backend="local").forecast(X_train, y_train, X_test, output_type="median")
```

Proposed class skeleton:

```python
class SynthefyTabularClient:
    """Brand-level tabular regression client.

    Reuses the SynthefyTabularRegressor contract: fit(X, y) then
    predict(X, *, output_type="mean"|"median"|"mode", quantiles=None).
    """

    def __init__(
        self,
        *,
        backend: str = "auto",          # "auto" | "local" | "remote"
        api_key: str | None = None,     # required for remote
        model_path: str | None = None,  # local-only passthroughs...
        device=None,
        token: str | bool | None = None,
        # plus the regressor's tuning knobs (inference_config, augmentations, ...)
        **regressor_kwargs,
    ) -> None: ...

    def fit(self, X, y): ...                       # sklearn-style
    def predict(self, X, *, output_type="mean", quantiles=None): ...
    def forecast(self, X_train, y_train, X_test, *, output_type="mean", quantiles=None): ...
```

Notes:
- **Lives in `synthefy/api_client.py`**, alongside `SynthefyAPIClient` /
  `SynthefyAsyncAPIClient`, so it imports the same way
  (`from synthefy.api_client import SynthefyTabularClient`); `synthefy/__init__.py`
  re-exports it to the top level. Critically, defining it there must **not** make
  `synthefy.api_client` import torch at module load — the heavy model lives in
  the separate `synthefy.tabular` subpackage and is imported lazily inside
  methods (see §5.4).
- The **method signatures are copied from the regressor** so there is exactly
  one contract to learn and document. For the local backend, `predict` delegates
  straight to the regressor's `predict`.
- `backend="auto"` (the **default**, decided): use `local` if the `tabular`
  extra is importable, else `remote` if an `api_key` is available, else raise a
  clear, actionable error. It **logs one line** stating which backend was
  selected, so the choice is never silent.
- **No async client.** Unlike `SynthefyAsyncAPIClient`, there is no
  `SynthefyAsyncTabularClient` (decided — out of scope). The client is
  synchronous only.

### 4.3 Why reuse the signature (not redesign)

- Zero new concepts for users already on `SynthefyTabularRegressor` or TabPFN.
- The remote backend can honor the identical signature because the Baseten
  contract already takes `X_train/y_train/X_test` and returns `predictions`;
  `fit` just buffers the context rows and `predict` issues one request.
- Keeps the door open to the distributional outputs (`output_type="quantiles"`)
  later without changing the client API.

---

## 5. Single-package consolidation strategy

### 5.1 The constraint

One distribution (`synthefy`), **and** `import synthefy` must stay **fast and
torch-free** for forecasting-only users. These are not in conflict — see below.

### 5.2 Code and dependencies are decoupled in a wheel

The insight that makes "single package" and "lightweight import" coexist: a
wheel separates **what code ships** from **what dependencies get installed**.

- The pure-Python tabular modules and the small bundled JSON configs **always
  ship** inside the `synthefy` wheel (they cost kilobytes).
- The heavy third-party libraries (`torch`, `scikit-learn`, `scipy`,
  `huggingface-hub`, …) live in an **optional extra**, so they are installed
  only on request:

| Command | Installs | Capabilities |
|---|---|---|
| `pip install synthefy` | base client only (HTTP + pandas) | forecasting API **+ remote tabular**; no torch |
| `pip install synthefy[tabular]` | base **+** torch/sklearn/scipy/hf-hub | the above **+ local tabular inference** |

This is exactly how libraries like `transformers` work: all the code ships in
one package, but `transformers[torch]` adds the framework. We are doing the
same — **one distribution, opt-in dependency footprint.**

### 5.3 Layout — absorb the model code into `synthefy`

The model code's canonical home moves from `src/synthefy_tabular/` into the
`synthefy` package as a `synthefy.tabular` subpackage:

```
synthefy/
  __init__.py            # re-exports SynthefyAPIClient, SynthefyAsyncAPIClient, SynthefyTabularClient  (NO torch import)
  api_client.py          # EXISTING module: SynthefyAPIClient/SynthefyAsyncAPIClient + new SynthefyTabularClient
                         #   (lazy-imports synthefy.tabular inside methods; no torch at module load)
  tabular/               # <- the absorbed model code (was src/synthefy_tabular/)
    api.py               #    SynthefyTabularRegressor
    inference/  model/  configs/*.json  ...
```

`pyproject.toml` of the **single** `synthefy` distribution:

```toml
[project]
name = "synthefy"
dependencies = [                 # base stays light
    "httpx>=...", "pandas>=...",        # whatever the API client already needs
]

[project.optional-dependencies]
tabular = [                      # the heavy stack lives HERE, not in a separate package
    "torch>=2.0", "scikit-learn>=1.4", "scipy>=1.13",
    "huggingface-hub>=1.0", "einops>=0.7", "kditransform>=1.0",
    "numpy>=2.0", "tqdm>=4.65",
]

[tool.setuptools.package-data]
"synthefy.tabular.configs" = ["*.json"]   # configs ship in the one wheel
```

### 5.4 Import-time safety (keeping `import synthefy` torch-free)

- `synthefy/__init__.py` must **never** import torch. `SynthefyTabularClient`
  itself is pure Python and can be imported/constructed with no heavy deps; only
  its **methods** touch the model, via a lazy import that fails with an
  actionable message when the extra is missing:

  ```python
  # synthefy/api_client.py  (illustrative — same module as SynthefyAPIClient)
  class SynthefyTabularClient:
      def _local_regressor(self):
          try:
              from synthefy.tabular import SynthefyTabularRegressor   # imports torch lazily, like today
          except ModuleNotFoundError as exc:
              raise ModuleNotFoundError(
                  "Local tabular inference needs the optional extra: "
                  "pip install 'synthefy[tabular]'"
              ) from exc
          ...
  ```

- The absorbed code keeps its existing **lazy `numpy`/`torch` imports** (already
  the pattern in [`api.py`](../../src/synthefy_tabular/api.py)), so even with the
  extra installed, nothing heavy loads until a model is actually used.
- A CI test should assert `import synthefy` leaves `"torch" not in sys.modules`.
- Missing-extra and missing-`api_key` errors must be explicit and name the fix.

---

## 6. Implementation plan (phased)

1. **Contract locked** — decisions recorded in §7.
2. **Absorb the code:** move `src/synthefy_tabular/` → `synthefy/tabular/`
   (renaming internal imports `synthefy_tabular` → `synthefy.tabular`), and move
   its heavy dependencies into the `synthefy` package's `[tabular]` extra. Keep
   the bundled configs as `synthefy.tabular.configs` package-data.
3. **Add the client:** implement `SynthefyTabularClient` in `synthefy.api_client`
   (local backend first), with lazy imports and the top-level re-export. Add a CI
   test asserting `import synthefy` pulls **no** torch.
4. **Remote backend:** implement `backend="remote"` against the hosted tabular
   endpoint, reusing `synthefy`'s existing HTTP/auth plumbing; add the
   `forecast()` one-shot and the `backend="auto"` selection + one-line log.
5. **Parity tests:** local vs remote produce matching predictions on a fixture;
   `output_type` matrix; clear errors for missing extra / api_key; the
   end-to-end public-weights download path (as already validated for v0.2.0).
6. **Docs:** a "Tabular" page in the `synthefy` docs reusing the regressor's
   signature; note TabPFN-compatibility.
7. **Release:** `synthefy` minor bump (e.g. 3.1.0) — additive, non-breaking for
   existing `synthefy` users (base footprint unchanged; tabular is opt-in).

---

## 7. Decisions (resolved)

- **D1 — AutoGluon parity:** **Not needed.** Standardize on the sklearn/TabPFN
  signature; no AutoGluon adapter (§3.3).
- **D2 — Async:** **Not needed.** No `SynthefyAsyncTabularClient`; the client is
  synchronous only.
- **D3 — Default backend:** **`"auto"`** — local if the `tabular` extra is
  importable, else remote if an `api_key` is present — and it **logs one line**
  naming the backend it picked.
- **D4 — Re-exports:** **Only the client.** Do **not** re-export
  `SynthefyTabularRegressor` from `synthefy`; `SynthefyTabularClient` is the sole
  public tabular entry point (the regressor stays an internal implementation
  detail under `synthefy.tabular`).

---

## 8. Summary

- The current signature is **scikit-learn / TabPFN compatible**, **not**
  AutoGluon-native; keep it as the canonical contract (no AutoGluon adapter).
- Introduce `SynthefyTabularClient` as a **backend-pluggable façade** (local
  torch inference vs hosted endpoint) that **reuses the regressor signature
  verbatim**. Define it in the existing `synthefy.api_client` module next to the
  other clients, so `from synthefy.api_client import SynthefyTabularClient`
  mirrors today's imports (also re-exported at the top level).
- Consolidate into **one distribution**: absorb the model code into `synthefy`
  as `synthefy.tabular`, and move the heavy stack into a `synthefy[tabular]`
  **extra**. Because a wheel decouples shipped code from installed dependencies,
  `import synthefy` stays lightweight and torch-free while the whole tabular SDK
  lives inside the single published `synthefy` package.
