"""Public API for Nori."""

from __future__ import annotations

from synthefy_nori import discretize
from synthefy_nori.inference.degradation import (
    ContextSubsampledWarning,
    DegradedPipelineWarning,
    SvdFallbackWarning,
    strict_pipeline,
)
from synthefy_nori.inference.memory_policy import ContextTooLargeError, MemoryPolicy
from synthefy_nori.api import (
    NoriRegressor,
    config_path,
    infer,
    predict,
)
from synthefy_nori.embedding import NoriEmbedding
from synthefy_nori.pricing import billable_price

__version__ = "0.14.0"

__all__ = [
    "ContextSubsampledWarning",
    "ContextTooLargeError",
    "DegradedPipelineWarning",
    "MemoryPolicy",
    "SvdFallbackWarning",
    "strict_pipeline",
    "NoriRegressor",
    "discretize",
    "NoriEmbedding",
    "billable_price",
    "config_path",
    "infer",
    "predict",
]
