# Interpretability

Needs the extra: `pip install "synthefy-nori[interpretability]"` (pulls in
shapiq + matplotlib). `NoriRegressor` subclasses `RegressorMixin`/`BaseEstimator`,
so it also works with the wider sklearn interpretability ecosystem directly.
All methods operate on the regression predictive mean.

## Shapley values & interactions (shapiq)

`get_nori_imputation_explainer` builds a `shapiq.TabularExplainer` that removes
features by **imputation** against a background set. The fitted context stays
fixed across coalitions; each coalition costs one `predict` call, so total cost
is set by `budget`.

```python
from synthefy_nori import NoriRegressor
from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

reg = NoriRegressor(model="nori-30m").fit(X_train, y_train)

# Plain first-order Shapley values:
explainer = get_nori_imputation_explainer(reg, X_train, index="SV", max_order=1)
iv = explainer.explain(X_test[0:1], budget=128)     # shapiq InteractionValues

# Pairwise interactions on top:
explainer = get_nori_imputation_explainer(reg, X_train, index="k-SII", max_order=2)
iv = explainer.explain(X_test[0:1], budget=128)
iv.plot_waterfall()                                  # additive contribution waterfall
```

- **Budget guidance:** start at 128 and raise only if attributions look noisy
  across reruns. `<10` features → 64–128; 10–20 → 128–512; 20+ → 512–2048.
- **Global importance:** Shapley is per-row; for a dataset-level view, average
  `|φ|` over a sample of query rows — `templates/explain.py` does exactly this
  and prints a sorted importance table.

## Partial dependence / ICE

```python
from synthefy_nori.interpretability.pdp import partial_dependence_plots

partial_dependence_plots(reg, X_test, features=[0, 2], kind="average")
# kind="individual" for ICE curves, "both" for overlay
```

## Feature selection

```python
from synthefy_nori.interpretability.feature_selection import feature_selection

res = feature_selection(reg, X_train, y_train, n_features_to_select=5, cv=3,
                        feature_names=list(feature_names))
print(res.selected_names)
print(res.baseline_score_mean, "->", res.selected_score_mean)   # CV R²
```

Sequential selection re-runs in-context prediction on every CV split ×
candidate feature, so it is the **slow** tool here — keep it to a few thousand
rows and a modest feature count, and run it once you already have a working
model, not before.

Runnable end-to-end example: `examples/interpretability_regression.py` in this
repo; skill template: `templates/explain.py`.
