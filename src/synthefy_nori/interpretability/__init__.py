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

The :mod:`~synthefy_nori.interpretability.glassbox` subpackage adds the one-call
:class:`NoriInterpreter` — Nori feature importance (permutation / SHAP) distilled
into a glass-box EBM. It is re-exported here for convenience but imported lazily,
so ``import synthefy_nori.interpretability`` never eagerly pulls in ``interpret``:

    from synthefy_nori.interpretability import NoriInterpreter
"""

__all__ = ["NoriInterpreter"]


def __getattr__(name):
    # PEP 562 lazy re-export: keep the package import light (glassbox pulls in
    # interpret-core), while still exposing NoriInterpreter at the top level.
    if name == "NoriInterpreter":
        from synthefy_nori.interpretability.glassbox import NoriInterpreter

        return NoriInterpreter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
