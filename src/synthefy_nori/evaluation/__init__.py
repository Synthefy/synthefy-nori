"""Official TALENT, OpenML-CTR23, and TabArena regression evaluation."""

from __future__ import annotations


def __getattr__(name):
    if name in {"BenchmarkEvalUnit", "BenchmarkLoader", "MaterializedSplit", "UnitMeta"}:
        from synthefy_nori.evaluation import protocol

        return getattr(protocol, name)
    if name in {"EvalResultRow", "OFFICIAL_PROTOCOL", "run_benchmark"}:
        from synthefy_nori.evaluation import harness

        return getattr(harness, name)
    if name in {"ModelEntry", "ModelRegistry"}:
        from synthefy_nori.evaluation import models

        return getattr(models, name)
    if name in {"OpenMLTaskLoader", "TabArenaLoader", "TalentNativeLoader"}:
        from synthefy_nori.evaluation import loaders

        return getattr(loaders, name)
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")
