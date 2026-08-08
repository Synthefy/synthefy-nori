"""Loader for official OpenML regression task splits."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from typing import Callable, Iterator, Optional

from synthefy_nori.evaluation.dataframe_prep import featurize_split, validate_target
from synthefy_nori.evaluation.protocol import (
    BenchmarkEvalUnit,
    MaterializedSplit,
    UnitMeta,
    compose,
    decompose,
)
from synthefy_nori.featurize import DEFAULT_MAX_CARDINALITY


def _openml():
    try:
        import openml
    except ImportError as exc:
        raise ImportError("OpenML benchmarks require `pip install 'synthefy-nori[eval]'`.") from exc
    return openml


def _set_openml_cache(cache_dir: Optional[str]) -> None:
    if cache_dir:
        _openml().config.set_root_cache_directory(cache_dir)


def _read_task_id_list(list_name: str) -> list[int]:
    text = (files("synthefy_nori.evaluation.benchmark_lists") / f"{list_name}.csv").read_text()
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        last = line.split(",")[-1].strip()
        if last.isdigit():
            ids.append(int(last))
    if not ids:
        raise ValueError(f"OpenML benchmark list {list_name!r} has no task ids")
    return ids


class OpenMLTaskLoader:
    """Enumerate and materialize every official fold registered on OpenML tasks."""

    name = "openml"

    def __init__(
        self,
        task_ids,
        *,
        cache_dir: Optional[str] = None,
        max_categorical_cardinality: int = DEFAULT_MAX_CARDINALITY,
        name: Optional[str] = None,
        n_repeats_policy: Optional[Callable[[int], int]] = None,
    ):
        if isinstance(task_ids, (int, str)):
            task_ids = [task_ids]
        self.task_ids = [int(task_id) for task_id in task_ids]
        if not self.task_ids:
            raise ValueError("OpenMLTaskLoader requires at least one task id")
        self.cache_dir = cache_dir
        self.max_categorical_cardinality = max_categorical_cardinality
        self.n_repeats_policy = n_repeats_policy
        if name is not None:
            self.name = name
        self._meta = {}
        self._xy = {}

    def fingerprint(self) -> str:
        """Hash the pinned task membership and repeat-policy implementation."""
        policy = getattr(self.n_repeats_policy, "__qualname__", None)
        payload = (
            f"{self.name}|"
            f"{','.join(map(str, self.task_ids))}|"
            f"{policy}|"
            f"max-categorical-cardinality={self.max_categorical_cardinality}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_pinned(cls, list_name: str, *, name: Optional[str] = None, **kwargs):
        """Build from a package-pinned task list without a mutable suite query."""
        return cls(
            _read_task_id_list(list_name),
            name=name or list_name.replace("_", "-"),
            **kwargs,
        )

    @classmethod
    def from_ctr23(cls, **kwargs):
        """OpenML-CTR23: suite 353's 35 regression tasks, pinned by task id."""
        return cls.from_pinned("openml_ctr23", **kwargs)

    def _task_meta(self, task_id: int):
        if task_id not in self._meta:
            openml = _openml()
            _set_openml_cache(self.cache_dir)
            task = openml.tasks.get_task(task_id, download_splits=True)
            task_type = str(getattr(task, "task_type", "") or "")
            if "regression" not in task_type.lower():
                raise ValueError(f"OpenML task {task_id} is {task_type!r}, not regression")
            dataset = task.get_dataset()
            n_repeats, n_folds, _ = task.get_split_dimensions()
            if self.n_repeats_policy is not None:
                total_instances = int(dataset.qualities["NumberOfInstances"])
                n_repeats = min(int(n_repeats), self.n_repeats_policy(total_instances))
            self._meta[task_id] = {
                "task": task,
                "name": dataset.name,
                "did": getattr(task, "dataset_id", None),
                "n_repeats": int(n_repeats),
                "n_folds": int(n_folds),
            }
        return self._meta[task_id]

    def _task_xy(self, task_id: int):
        if task_id not in self._xy:
            self._xy[task_id] = self._task_meta(task_id)["task"].get_X_and_y(dataset_format="dataframe")
        return self._xy[task_id]

    def units(self) -> Iterator[BenchmarkEvalUnit]:
        """Yield units from the live task metadata; never hardcode fold totals."""
        for task_id in self.task_ids:
            meta = self._task_meta(task_id)
            n_total = meta["n_repeats"] * meta["n_folds"]
            for repeat in range(meta["n_repeats"]):
                for fold_in_repeat in range(meta["n_folds"]):
                    yield BenchmarkEvalUnit(
                        source=self.name,
                        dataset=meta["name"],
                        fold=compose(repeat, fold_in_repeat, meta["n_folds"]),
                        n_folds=n_total,
                        meta=UnitMeta(
                            openml_task_id=task_id,
                            openml_dataset_id=meta["did"],
                            repeat=repeat,
                            fold_in_repeat=fold_in_repeat,
                            n_repeats=meta["n_repeats"],
                            n_folds_per_repeat=meta["n_folds"],
                        ),
                    )

    def materialize(self, unit: BenchmarkEvalUnit) -> MaterializedSplit:
        task_id = unit.meta.openml_task_id
        if task_id is None:
            raise ValueError(f"OpenML unit {unit.dataset!r} has no task id")
        meta = self._task_meta(task_id)
        X, y = self._task_xy(task_id)
        repeat, fold_in_repeat = decompose(unit.fold, meta["n_folds"])
        train_idx, test_idx = meta["task"].get_train_test_split_indices(
            repeat=repeat,
            fold=fold_in_repeat,
            sample=0,
        )

        y_train, train_ok = validate_target(y.iloc[train_idx], meta["name"], self.name)
        y_test, test_ok = validate_target(y.iloc[test_idx], meta["name"], self.name)
        X_train_df = X.iloc[train_idx][train_ok]
        X_test_df = X.iloc[test_idx][test_ok]
        y_train, y_test = y_train[train_ok], y_test[test_ok]
        if len(y_train) < 2 or len(y_test) < 1:
            raise ValueError(f"OpenML task {task_id} fold {unit.fold} has too few finite targets")

        X_train, X_test, n_features = featurize_split(
            X_train_df,
            X_test_df,
            max_categorical_cardinality=self.max_categorical_cardinality,
        )
        return MaterializedSplit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_features=n_features,
        )
