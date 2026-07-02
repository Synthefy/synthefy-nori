# One-step time-series forecasting with Nori

Nori has no time axis — no horizon, freq, or timestamp arguments. You forecast
by **framing the series as tabular rows**: one row per period, target = that
period's value, features built **only from strictly earlier periods**. Done
right, one-step-ahead forecasting is just regression with a stricter split
discipline.

## The one rule: no leakage

Features for period *t* may use only data `< t`. Concretely: every derived
column must go through `shift(1)` (or a lag) before any rolling/aggregation.
And **never evaluate with a random split** — rows are time-ordered, so a random
split puts the future in the context. Use a rolling-origin backtest instead.

## Feature recipe (start here, sweep later)

| feature | construction | why |
|---|---|---|
| `lag1, lag2, lag3` | `y.shift(k)` | local level & short dynamics |
| `lag<season>` | `y.shift(season)` (12 for monthly, 7 for daily) | seasonality |
| `roll_mean, roll_std` | `y.shift(1).rolling(season).mean() / .std()` | trailing level & volatility |
| `trend` | row index | long-run drift |
| `phase_sin, phase_cos` | `sin/cos(2π · (t mod season)/season)` | smooth seasonal phase |

Early rows have NaN lags — that's fine, Nori accepts NaN features (leave them;
don't zero-fill). The target still must be finite.

## Fit/predict pattern (expanding context)

`fit()` is free (it just stores context), so refit at every origin and reuse
one estimator object:

```python
reg = NoriRegressor(device="cpu")            # construct ONCE, refit per origin
for t in test_origins:                       # expanding window
    reg.fit(X[:t], y[:t])                    # context = everything before t
    point = reg.predict(X[t:t+1], output_type="median")
    q10, q90 = reg.predict(X[t:t+1], output_type="quantiles", quantiles=[0.1, 0.9])
```

`"median"` is the robust default for skewed series (demand, traffic, revenue);
for such positive skewed targets also consider fitting on `np.log1p(y)` and
inverting with `np.expm1` — quantiles invert directly through any monotone
transform, unlike the mean.

## Evaluate against naive baselines — always

A forecast that can't beat `y[t-1]` (last value) or `y[t-season]` (seasonal
naive) isn't earning its keep. Report MAE/WAPE for Nori **and both baselines**
over the same rolling origins, plus the empirical coverage of your [q10, q90]
band (nominal 0.80). `templates/forecast_one_step.py` runs this end-to-end.

## Beyond one step

- **Multi-step:** either *recursive* (feed predictions back as lags — simple,
  but errors compound and the bands stop being honest) or *direct* (train a
  separate frame with target `y.shift(-h)` per horizon `h` — one Nori fit per
  horizon, honest quantiles). Prefer direct when the horizon matters.
- **Many related series (a panel):** pool rows across series with a shared
  feature schema and a scale-free target so one model serves them all — that,
  plus cold-start handling, exogenous signals, and ensembling, is the
  **nori-demand-forecasting** skill's territory. Reach for it when you have
  SKUs/stores/products rather than a single series.
