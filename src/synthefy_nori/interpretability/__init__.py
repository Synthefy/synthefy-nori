"""Interpretability for Nori regression: Shapley values & interactions (shapiq),
partial dependence / ICE plots, and sequential feature selection.

Nori's public regressor (:class:`synthefy_nori.NoriRegressor`) is a scikit-learn
estimator, so these are thin, well-tested adapters over shapiq and sklearn — no
bespoke attribution math. Requires the optional extra:

    pip install "synthefy-nori[interpretability]"

Import from the submodules (so ``import synthefy_nori.interpretability`` stays
light and never eagerly imports shapiq):

    from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer
    from synthefy_nori.interpretability.pdp import partial_dependence_plots
    from synthefy_nori.interpretability.feature_selection import feature_selection
"""
