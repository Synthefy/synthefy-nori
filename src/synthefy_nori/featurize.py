"""Compatibility exports for client-side tabular featurization.

The lightweight :mod:`synthefy.featurize` module is the canonical owner in
Synthefy 7. This module preserves the established
``synthefy_nori.featurize`` import path for local users while the released
6.3 client remains in the compatibility lane.
"""

from synthefy.featurize import (
    CATEGORICAL_ENCODINGS,
    DEFAULT_CATEGORICAL_ENCODING,
    DEFAULT_MAX_CARDINALITY,
    align_and_featurize,
)

__all__ = [
    "CATEGORICAL_ENCODINGS",
    "DEFAULT_CATEGORICAL_ENCODING",
    "DEFAULT_MAX_CARDINALITY",
    "align_and_featurize",
]
