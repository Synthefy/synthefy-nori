"""TabArena-v0.1 regression tasks with their official outer-CV policy."""

from __future__ import annotations

import hashlib
from typing import Iterator, Optional

from synthefy_nori.evaluation.loaders.openml_task import OpenMLTaskLoader, _read_task_id_list
from synthefy_nori.evaluation.protocol import BenchmarkEvalUnit, MaterializedSplit
from synthefy_nori.featurize import DEFAULT_MAX_CARDINALITY


def _tabarena_num_repeats(total_instances: int) -> int:
    """Mirror TabArena's official repeat count for a regression dataset."""
    if total_instances < 2_500:
        return 10
    if total_instances > 250_000:
        return 1
    return 3


class TabArenaLoader:
    """Thirteen pinned TabArena regression tasks and their official OpenML splits."""

    name = "tabarena"

    def __init__(
        self,
        task_ids=None,
        *,
        cache_dir: Optional[str] = None,
        max_categorical_cardinality: int = DEFAULT_MAX_CARDINALITY,
    ):
        ids = task_ids if task_ids is not None else _read_task_id_list("tabarena_reg")
        self._openml = OpenMLTaskLoader(
            ids,
            cache_dir=cache_dir,
            max_categorical_cardinality=max_categorical_cardinality,
            n_repeats_policy=_tabarena_num_repeats,
        )

    @property
    def task_ids(self):
        return self._openml.task_ids

    def fingerprint(self) -> str:
        payload = f"tabarena-v0.1|{self._openml.fingerprint()}|2500:10|250000:3|above:1"
        return hashlib.sha256(payload.encode()).hexdigest()

    def units(self) -> Iterator[BenchmarkEvalUnit]:
        for unit in self._openml.units():
            yield unit.model_copy(update={"source": self.name})

    def materialize(self, unit: BenchmarkEvalUnit) -> MaterializedSplit:
        return self._openml.materialize(unit)
