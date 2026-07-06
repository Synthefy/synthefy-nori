"""Embedding extraction for Nori regression.

Nori's encoder produces a learned representation of every row that downstream
models (kNN, linear probes, clustering, retrieval) can consume.
:class:`NoriEmbedding` is a scikit-learn ``TransformerMixin`` that wraps
:class:`synthefy_nori.NoriRegressor` and exposes those embeddings, with an
optional out-of-fold mode for leakage-free training embeddings.

    from synthefy_nori.embedding import NoriEmbedding
"""

from synthefy_nori.embedding.embedding import NoriEmbedding

__all__ = ["NoriEmbedding"]
