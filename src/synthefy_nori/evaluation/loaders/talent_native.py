"""Official TALENT-100 regression loader for the native January 2025 archive.

TALENT provides train/validation/test NumPy arrays per dataset.  The official
Nori protocol uses train+validation as context, the provided test split for
scoring, and caps those sets at 10,000 and 20,000 rows respectively.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from synthefy_nori.evaluation.protocol import BenchmarkEvalUnit, MaterializedSplit, UnitMeta

DEFAULT_ROOT = os.environ.get("TALENT_NATIVE_ROOT", "cache/talent/data")


def _regression_folders(root: str) -> list[Path]:
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(
            f"TALENT data root not found: {path}. Download/extract the native archive as documented in docs/evaluation.md."
        )
    folders = []
    for folder in path.iterdir():
        info_path = folder / "info.json"
        if info_path.is_file() and json.loads(info_path.read_text()).get("task_type") == "regression":
            folders.append(folder)
    return sorted(folders)


def _load_split(folder: Path, split: str):
    blocks = []
    numeric_path = folder / f"N_{split}.npy"
    categorical_path = folder / f"C_{split}.npy"
    if numeric_path.exists():
        blocks.append(("numeric", np.load(numeric_path, allow_pickle=True).astype(np.float64)))
    if categorical_path.exists():
        blocks.append(("categorical", np.load(categorical_path, allow_pickle=True)))
    target = np.load(folder / f"y_{split}.npy", allow_pickle=True).astype(np.float64).reshape(-1)
    return blocks, target


def _assemble(folder: Path):
    train_blocks, y_train = _load_split(folder, "train")
    val_blocks, y_val = _load_split(folder, "val")
    test_blocks, y_test = _load_split(folder, "test")

    def block(blocks, kind):
        return next((array for block_kind, array in blocks if block_kind == kind), None)

    def context(kind):
        train = block(train_blocks, kind)
        val = block(val_blocks, kind)
        if train is None:
            return None
        return np.vstack([train, val]) if val is not None else train

    numeric_context = context("numeric")
    categorical_context = context("categorical")
    numeric_test = block(test_blocks, "numeric")
    categorical_test = block(test_blocks, "categorical")
    y_context = np.concatenate([y_train, y_val])

    context_columns, test_columns = [], []
    if numeric_context is not None:
        context_columns.append(numeric_context)
        test_columns.append(numeric_test)
    if categorical_context is not None:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        ).fit(categorical_context.astype(str))
        context_columns.append(encoder.transform(categorical_context.astype(str)))
        test_columns.append(encoder.transform(categorical_test.astype(str)))
    if not context_columns:
        raise ValueError(f"TALENT dataset {folder.name!r} has no feature arrays")

    return (
        np.hstack(context_columns).astype(np.float32),
        y_context,
        np.hstack(test_columns).astype(np.float32),
        y_test,
    )


class TalentNativeLoader:
    """Lazy loader for the 100 official TALENT regression splits."""

    name = "talent"
    ctx_cap = 10_000
    test_cap = 20_000

    def __init__(self, root: str = DEFAULT_ROOT, *, expected_datasets: int | None = 100):
        self.root = root
        self.expected_datasets = expected_datasets
        self._fingerprint = None

    def fingerprint(self) -> str:
        """Hash the actual extracted benchmark inputs used by this loader."""
        if self._fingerprint is None:
            root = Path(self.root)
            digest = hashlib.sha256()
            folders = _regression_folders(self.root)
            paths = (
                file
                for folder in folders
                for file in folder.iterdir()
                if file.suffix in {".json", ".npy"}
            )
            for path in sorted(paths):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(b"\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            self._fingerprint = digest.hexdigest()
        return self._fingerprint

    def units(self) -> Iterator[BenchmarkEvalUnit]:
        folders = _regression_folders(self.root)
        if self.expected_datasets is not None and len(folders) != self.expected_datasets:
            raise ValueError(
                f"expected {self.expected_datasets} TALENT regression datasets, found {len(folders)} under {self.root}"
            )
        for folder in folders:
            yield BenchmarkEvalUnit(
                source=self.name,
                dataset=folder.name,
                meta=UnitMeta(source_path=str(folder)),
            )

    def materialize(self, unit: BenchmarkEvalUnit) -> MaterializedSplit:
        if not unit.meta.source_path:
            raise ValueError(f"TALENT unit {unit.dataset!r} has no source path")
        X_train, y_train, X_test, y_test = _assemble(Path(unit.meta.source_path))
        return MaterializedSplit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_features=X_train.shape[1],
        )
