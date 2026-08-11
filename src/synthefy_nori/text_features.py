"""Backward-compatible text-feature imports.

The implementation is owned by the lightweight :mod:`synthefy` package so the
hosted client can prepare text features without installing the local Nori model.
This module keeps the established ``synthefy_nori.text_features`` import path
working for local users and existing pickles.
"""

try:
    from synthefy.text_features import (
        MODELS,
        _MISSING,
        MultimodalPreprocessor,
        _canon_cat,
        _make_encoder,
        build_paragraphs,
    )
except ModuleNotFoundError as exc:
    if exc.name != "synthefy.text_features":
        raise
    from synthefy_nori._legacy_text_features import (
        MODELS,
        _MISSING,
        MultimodalPreprocessor,
        _canon_cat,
        _make_encoder,
        build_paragraphs,
    )

__all__ = ["MODELS", "MultimodalPreprocessor", "build_paragraphs"]
