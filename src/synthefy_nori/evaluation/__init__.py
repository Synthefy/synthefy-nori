"""Nori Evaluation Framework.

Evaluate Nori checkpoints across the TabArena, TALENT, and
OpenML regression benchmarks.
"""

from __future__ import annotations


def __getattr__(name):
    """Lazy imports to avoid dependency issues at import time."""
    if name == "DatasetRegistry":
        from synthefy_nori.evaluation.datasets import DatasetRegistry

        return DatasetRegistry
    if name == "DatasetEntry":
        from synthefy_nori.evaluation.datasets import DatasetEntry

        return DatasetEntry
    if name == "ModelRegistry":
        from synthefy_nori.evaluation.models import ModelRegistry

        return ModelRegistry
    if name == "ModelEntry":
        from synthefy_nori.evaluation.models import ModelEntry

        return ModelEntry
    if name == "EvalRunner":
        from synthefy_nori.evaluation.runner import EvalRunner

        return EvalRunner
    if name == "EvalAnalyzer":
        from synthefy_nori.evaluation.analysis import EvalAnalyzer

        return EvalAnalyzer
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")
