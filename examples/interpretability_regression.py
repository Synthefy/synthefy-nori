"""Interpretability on a regression task with Nori.

    pip install "synthefy-nori[interpretability]"
    python examples/interpretability_regression.py

Shows the three methods: shapiq Shapley values for one prediction, a partial
dependence plot, and sequential feature selection.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from synthefy_nori import NoriRegressor
from synthefy_nori.interpretability.feature_selection import feature_selection
from synthefy_nori.interpretability.pdp import partial_dependence_plots
from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = NoriRegressor()           # downloads weights from the HF Hub on first use
model.fit(X_train.values, y_train.values)

# 1) Explain a single prediction with shapiq (Shapley values + pairwise interactions).
explainer = get_nori_imputation_explainer(model, X_train.values, index="k-SII", max_order=2)
sv = explainer.explain(X_test.values[0:1], budget=128)
print(sv)                          # top contributions / interactions
sv.plot_waterfall()                # additive contribution waterfall

# 2) Global feature effect: partial dependence for two features.
partial_dependence_plots(model, X_test.values, features=[0, 2], kind="average")

# 3) Which features can we drop? (small/slow — ICL runs per CV fit)
result = feature_selection(model, X_train.values, y_train.values,
                           n_features_to_select=5, cv=3,
                           feature_names=list(X.columns))
print("selected:", result.selected_names)
print(f"CV R2: all-features {result.baseline_score_mean:.3f} -> "
      f"selected {result.selected_score_mean:.3f}")
