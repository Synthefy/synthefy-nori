# Interpretability

`NoriRegressor` subclasses scikit-learn's `RegressorMixin`/`BaseEstimator`, so it
works out of the box with [shapiq](https://github.com/mmschlk/shapiq) and the
sklearn interpretability ecosystem — no wrappers needed beyond the thin
convenience adapters in `synthefy_nori.interpretability`.

```bash
pip install "synthefy-nori[interpretability]"   # pulls in shapiq
```

Nori is **regression-only**, so all methods below operate on the regression
predictive mean.

## Shapley values & interactions (shapiq, recommended)

`get_nori_imputation_explainer` builds a `shapiq.TabularExplainer` that removes
features by **imputation** against a background set — the training context is
fixed across coalitions, so each coalition is a single `predict` call (cost set
by `budget`).

```python
from synthefy_nori import NoriRegressor
from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

model = NoriRegressor().fit(X_train, y_train)

# index="k-SII", max_order=2 captures pairwise interactions;
# use index="SV", max_order=1 for plain Shapley values.
explainer = get_nori_imputation_explainer(model, X_train, index="k-SII", max_order=2)
sv = explainer.explain(X_test[:1], budget=128)
print(sv)
sv.plot_waterfall()
```

**Budget:** start at `128` and raise only if explanations look noisy. Guide:
`<10` features → 64–128; 10–20 → 128–512; 20+ → 512–2048.

> Note: TabPFN's fast Shapley path reuses a KV-cache across coalitions; Nori's
> public package does not ship that cache, so explanations run one forward per
> coalition — correct and budget-controlled, just not cache-accelerated.

## Partial dependence / ICE

```python
from synthefy_nori.interpretability.pdp import partial_dependence_plots
partial_dependence_plots(model, X_test, features=[0, 2], kind="average")  # "individual"/"both" for ICE
```

## Feature selection

```python
from synthefy_nori.interpretability.feature_selection import feature_selection
res = feature_selection(model, X_train, y_train, n_features_to_select=5, cv=3)
print(res.selected_indices, res.selected_score_mean)
```

Sequential selection re-fits in-context on every CV split, so keep it to a few
thousand samples and a modest feature count.

Full runnable example: [`examples/interpretability_regression.py`](../examples/interpretability_regression.py).
