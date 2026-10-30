"""Deprecated import shim — see :mod:`synthefy_nori.interpretability.glassbox.importance`.

Kept so ``from synthefy_nori.explainability.importance import nori_permutation_importance``
(and ``nori_shap_importance``) keeps working after the move. Prefer:

    from synthefy_nori.interpretability.glassbox.importance import nori_permutation_importance
"""

from synthefy_nori.interpretability.glassbox.importance import (  # noqa: F401
    first_order_vector,
    nori_permutation_importance,
    nori_shap_importance,
)

__all__ = ["nori_permutation_importance", "nori_shap_importance", "first_order_vector"]
