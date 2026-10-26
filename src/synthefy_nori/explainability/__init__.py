"""Deprecated import shim — the explainability tools moved under interpretability.

``synthefy_nori.explainability`` was folded into
:mod:`synthefy_nori.interpretability.glassbox` so all interpretability lives in one
package. This module re-exports the public surface so existing imports keep working:

    from synthefy_nori.explainability import NoriInterpreter          # -> interpretability.glassbox
    from synthefy_nori.explainability.importance import nori_permutation_importance

Prefer the new paths:

    from synthefy_nori.interpretability import NoriInterpreter
    from synthefy_nori.interpretability.glassbox.importance import nori_permutation_importance
"""

import warnings as _warnings

from synthefy_nori.interpretability.glassbox import NoriInterpreter

_warnings.warn(
    "synthefy_nori.explainability has moved to synthefy_nori.interpretability.glassbox; "
    "import NoriInterpreter from synthefy_nori.interpretability instead. This shim will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["NoriInterpreter"]
