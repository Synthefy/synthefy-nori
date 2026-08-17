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
from synthefy_nori.configs import DEFAULT_INFERENCE_CONFIG, DEFAULT_MODEL_CONFIG
from synthefy_nori.api import (
    NoriRegressor,
    config_path,
    infer,
    predict,
)
from synthefy_nori.embedding import NoriEmbedding
from synthefy_nori.pricing import billable_price

__version__ = "0.17.3"

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
    "DEFAULT_INFERENCE_CONFIG",
    "DEFAULT_MODEL_CONFIG",
    "config_path",
    "infer",
    "predict",
]
