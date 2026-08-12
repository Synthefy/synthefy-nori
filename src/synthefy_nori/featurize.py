"""Compatibility exports for client-side tabular featurization.

The lightweight :mod:`synthefy.featurize` module is the canonical owner in
Synthefy 7. This module preserves the established
``synthefy_nori.featurize`` import path for local users while the released
6.3 client remains in the compatibility lane.
"""

try:
    from synthefy.featurize import (
        CATEGORICAL_ENCODINGS,
        DEFAULT_CATEGORICAL_ENCODING,
        DEFAULT_MAX_CARDINALITY,
        _featurize_frames,
        _has_encodable_columns,
        _numeric_categories_to_values,
        align_and_featurize,
    )
except ModuleNotFoundError as exc:
    if exc.name != "synthefy.featurize":
        raise
    from synthefy_nori._legacy_featurize import (
        CATEGORICAL_ENCODINGS,
        DEFAULT_CATEGORICAL_ENCODING,
        DEFAULT_MAX_CARDINALITY,
        _featurize_frames,
        _has_encodable_columns,
        _numeric_categories_to_values,
        align_and_featurize,
    )

__all__ = [
    "CATEGORICAL_ENCODINGS",
    "DEFAULT_CATEGORICAL_ENCODING",
    "DEFAULT_MAX_CARDINALITY",
    "align_and_featurize",
]
