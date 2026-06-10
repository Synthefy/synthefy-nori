"""Synthefy Tabular Evaluation Framework.

Evaluate Synthefy Tabular checkpoints across TabArena, RelBench CTU,
OpenML-CC18, and other benchmarks.
"""

from __future__ import annotations


def __getattr__(name):
    """Lazy imports to avoid dependency issues at import time."""
    if name == "DatasetRegistry":
        from synthefy_tabular.evaluation.datasets import DatasetRegistry
        return DatasetRegistry
    if name == "DatasetEntry":
        from synthefy_tabular.evaluation.datasets import DatasetEntry
        return DatasetEntry
    if name == "ModelRegistry":
        from synthefy_tabular.evaluation.models import ModelRegistry
        return ModelRegistry
    if name == "ModelEntry":
        from synthefy_tabular.evaluation.models import ModelEntry
        return ModelEntry
    if name == "EvalRunner":
        from synthefy_tabular.evaluation.runner import EvalRunner
        return EvalRunner
    if name == "EvalAnalyzer":
        from synthefy_tabular.evaluation.analysis import EvalAnalyzer
        return EvalAnalyzer
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")
