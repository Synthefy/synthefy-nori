"""Lazy contracts shared by the three official regression benchmark loaders.

A :class:`BenchmarkEvalUnit` is a cheap dataset/fold identity.  A loader only
downloads and materializes the arrays for that unit when the harness reaches it,
so repeated OpenML protocols do not need to live in memory at once.
"""

from __future__ import annotations

from typing import Iterator, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


def decompose(fold: int, n_folds_per_repeat: int) -> Tuple[int, int]:
    """Return ``(repeat, fold_in_repeat)`` for a flat fold index."""
    if n_folds_per_repeat < 1:
        raise ValueError("n_folds_per_repeat must be >= 1")
    if fold < 0:
        raise ValueError("fold must be >= 0")
    return divmod(fold, n_folds_per_repeat)


def compose(repeat: int, fold_in_repeat: int, n_folds_per_repeat: int) -> int:
    """Return the flat fold index for ``(repeat, fold_in_repeat)``."""
    if n_folds_per_repeat < 1:
        raise ValueError("n_folds_per_repeat must be >= 1")
    if repeat < 0:
        raise ValueError("repeat must be >= 0")
    if not 0 <= fold_in_repeat < n_folds_per_repeat:
        raise ValueError("fold_in_repeat must be in [0, n_folds_per_repeat)")
    return repeat * n_folds_per_repeat + fold_in_repeat


class UnitMeta(BaseModel):
    """Typed fold and source identity for an evaluation unit."""

    model_config = ConfigDict(frozen=True)

    repeat: int = 0
    fold_in_repeat: int = 0
    n_repeats: int = 1
    n_folds_per_repeat: int = 1
    openml_task_id: Optional[int] = None
    openml_dataset_id: Optional[int] = None
    source_path: Optional[str] = None


class BenchmarkEvalUnit(BaseModel):
    """One official ``(dataset, fold)`` evaluation identity."""

    model_config = ConfigDict(frozen=True)

    source: str
    dataset: str
    fold: int = Field(default=0, ge=0)
    n_folds: int = Field(default=1, ge=1)
    meta: UnitMeta = Field(default_factory=UnitMeta)


class MaterializedSplit(BaseModel):
    """Model-ready arrays for one unit, with NaNs preserved for harness policy."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_features: int

    @model_validator(mode="after")
    def _check_shapes(self) -> "MaterializedSplit":
        if self.X_train.ndim != 2 or self.X_test.ndim != 2:
            raise ValueError("X_train and X_test must be 2-D")
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X_train/y_train row mismatch")
        if self.X_test.shape[0] != self.y_test.shape[0]:
            raise ValueError("X_test/y_test row mismatch")
        if self.X_train.shape[1] != self.X_test.shape[1] or self.X_train.shape[1] != self.n_features:
            raise ValueError("train/test/n_features column mismatch")
        return self


@runtime_checkable
class BenchmarkLoader(Protocol):
    """Structural contract implemented by every official benchmark loader."""

    name: str

    def units(self) -> Iterator[BenchmarkEvalUnit]:
        ...

    def materialize(self, unit: BenchmarkEvalUnit) -> MaterializedSplit:
        ...
