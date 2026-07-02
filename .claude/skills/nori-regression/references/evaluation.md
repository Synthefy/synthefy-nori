# Evaluating Nori honestly

The single rule: **fix the yardstick before you compare anything.** Decide the
folds, the metric(s), and the data scope up front — then score every model,
every variant, on exactly that. Iterating the model is science; iterating the
eval (resplitting, reslicing, metric-shopping) until a favorite wins is
cheating yourself.

## The protocol

```python
from sklearn.model_selection import KFold, cross_validate

cv = KFold(n_splits=5, shuffle=True, random_state=0)   # built ONCE, shared by all models
scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}

res = cross_validate(model, X, y, cv=cv, scoring=scoring)   # same cv object for every model
```

- `NoriRegressor` is a scikit-learn estimator (`clone`-safe), so it drops into
  `cross_validate` directly — see `templates/compare_baselines.py`.
- Report **mean ± std over folds**, not just the mean. Two models whose means
  differ by less than the fold noise are **tied** — say so instead of
  declaring a winner.
- Always include at least two cheap baselines with real signal:
  `Ridge` (linear floor) and `RandomForestRegressor` (nonlinear floor). If
  Nori can't beat Ridge, the relationship is probably linear — that is a
  finding, not a failure.
- Pick the metric to match the decision: R² for variance-explained framing,
  MAE for robust absolute error, RMSE when large errors are disproportionately
  costly. Decide **before** looking at results; report the others as secondary.

## Prediction-interval calibration

Quantile bands are only useful if their coverage is honest. Check empirically
on held-out data:

```python
lo, hi = reg.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.9])
coverage = ((y_test >= lo) & (y_test <= hi)).mean()   # nominal: 0.80
```

- Coverage well **below** nominal → intervals overconfident; widen the taus
  you act on (e.g. use [0.05, 0.95] where you report "80%") or recalibrate.
- Coverage near 1.0 → intervals are so wide they say nothing.
- With `output_type="full"` you can trace the whole calibration curve:
  empirical coverage of `[τ_lo, τ_hi]` vs nominal across many τ pairs.

## Knowing when to stop

If several sensible variants (feature subsets, target transform on/off,
context subsampling) all land within the fold-noise band, you're at the
information floor of the dataset — more tuning finds noise, not signal. Say
so and stop; only genuinely new information (new features, more rows) moves
the number from there.
