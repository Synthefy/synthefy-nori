"""Explainability for Nori regression/classification: model-faithful feature
importance + distillation into a glass-box EBM (GA²M).

The idea: attribute importance to the *raw input columns* of a fitted
:class:`synthefy_nori.NoriRegressor` (so the scores honour Nori's full internal
preprocessing), keep the fewest top features that retain ≥95 % of Nori's skill,
then fit an interpretable Explainable Boosting Machine on exactly those features.
The EBM is a fully transparent additive model whose shape functions can be read
off directly — often matching (or beating) the full-feature EBM.

Requires the optional extra::

    pip install "synthefy-nori[explainability]"

One-call, scikit-learn-style API — pass the FULL table; it makes an internal 70/30
split and runs importance -> pruning -> glass-box EBM, storing every artifact::

    from synthefy_nori.interpretability.glassbox import NoriInterpreter
    interp = NoriInterpreter().fit(X, y)
    interp.feature_importances_    # per-feature importance
    interp.selected_features_      # the pruned feature set
    interp.ebm_                    # the fitted glass-box model
    interp.plot_model()           # the model diagram

Lower-level functions (import from the submodules; keeps ``import
synthefy_nori.interpretability.glassbox`` light and never eagerly imports ``interpret`` /
``shapiq`` / ``matplotlib``)::

    from synthefy_nori.interpretability.glassbox.importance import nori_permutation_importance, nori_shap_importance
    from synthefy_nori.interpretability.glassbox.ebm import fit_ebm, ebm_structure
    from synthefy_nori.interpretability.glassbox.pipeline import run           # end-to-end importance -> EBM
    from synthefy_nori.interpretability.glassbox.viz import plot_ebm_model     # the model diagram

End-to-end from the command line (runs on a bundled sklearn demo out of the box)::

    python -m synthefy_nori.interpretability.glassbox.pipeline --demo diabetes
    python -m synthefy_nori.interpretability.glassbox.pipeline --npz mydata.npz
"""

from synthefy_nori.interpretability.glassbox.interpreter import NoriInterpreter

__all__ = ["NoriInterpreter"]
