"""Official public benchmark loaders (imported lazily for optional OpenML)."""

from __future__ import annotations


def __getattr__(name):
    if name == "TalentNativeLoader":
        from synthefy_nori.evaluation.loaders.talent_native import TalentNativeLoader

        return TalentNativeLoader
    if name == "OpenMLTaskLoader":
        from synthefy_nori.evaluation.loaders.openml_task import OpenMLTaskLoader

        return OpenMLTaskLoader
    if name == "TabArenaLoader":
        from synthefy_nori.evaluation.loaders.tabarena import TabArenaLoader

        return TabArenaLoader
    raise AttributeError(f"module 'loaders' has no attribute {name!r}")
