"""Unified, crash-resumable harness for the official regression protocols."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from synthefy_nori import __version__
from synthefy_nori.evaluation.metrics import compute_reg_metrics
from synthefy_nori.inference.degradation import SvdFallbackWarning, strict_pipeline

OFFICIAL_PROTOCOL = "official-regression-v1"
OFFICIAL_ELEMENTS_BUDGET = 8_000_000
OFFICIAL_ALLOW_SUBSAMPLE = False
OFFICIAL_IMPUTE = "median"
OFFICIAL_SEED = 0
OFFICIAL_SUITE_COUNTS = {
    "talent": (100, 100),
    "openml-ctr23": (35, 800),
    "tabarena": (13, 222),
}

_IDENTITY_FIELDS = (
    "protocol",
    "source",
    "dataset",
    "fold",
    "n_folds",
    "repeat",
    "fold_in_repeat",
    "n_repeats",
    "n_folds_per_repeat",
    "openml_task_id",
    "openml_dataset_id",
    "data_fingerprint",
    "model",
    "checkpoint_sha256",
    "reg_config_sha256",
    "synthefy_nori_version",
    "source_tree_sha256",
    "seed",
    "impute",
    "context_cap",
    "test_cap",
    "elements_budget",
    "allow_subsample",
)


class EvalResultRow(BaseModel):
    """One fully identified ``unit x model`` result written to JSONL."""

    protocol: str = OFFICIAL_PROTOCOL
    model: str
    source: str
    dataset: str
    fold: int
    n_folds: int
    repeat: int
    fold_in_repeat: int
    n_repeats: int
    n_folds_per_repeat: int
    openml_task_id: Optional[int] = None
    openml_dataset_id: Optional[int] = None
    data_fingerprint: Optional[str] = None
    n_train: Optional[int] = None
    n_test: Optional[int] = None
    n_features: Optional[int] = None
    r2: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    error: Optional[str] = None
    seed: int
    rng_mode: str = "per_unit"
    impute: str
    context_cap: Optional[int] = None
    test_cap: Optional[int] = None
    checkpoint_sha256: Optional[str] = None
    reg_config_sha256: Optional[str] = None
    synthefy_nori_version: str = __version__
    source_tree_sha256: str
    elements_budget: Optional[int] = None
    allow_subsample: Optional[bool] = None
    model_metadata: dict


def _unit_seed(source: str, dataset: str, fold: int, base_seed: int) -> int:
    digest = hashlib.sha256(f"{source}|{dataset}|{fold}|{base_seed}".encode()).hexdigest()
    return int(digest[:16], 16)


def _subsample(X, y, cap: Optional[int], rng: np.random.Generator):
    if cap is None or len(y) <= cap:
        return X, y
    idx = rng.choice(len(y), cap, replace=False)
    return X[idx], y[idx]


def _apply_impute(X_train: np.ndarray, X_test: np.ndarray, policy: str):
    if policy == "keep":
        return X_train, X_test
    if policy not in ("median", "mean"):
        raise ValueError("impute must be 'median', 'mean', or 'keep'")
    stat_fn = np.nanmedian if policy == "median" else np.nanmean
    finite_train = np.where(np.isfinite(X_train), X_train, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        column_stat = stat_fn(finite_train, axis=0)
    column_stat = np.where(np.isfinite(column_stat), column_stat, 0.0)
    return (
        np.where(np.isfinite(X_train), X_train, column_stat).astype(np.float32),
        np.where(np.isfinite(X_test), X_test, column_stat).astype(np.float32),
    )


def iter_units(loaders: Iterable, limit: Optional[int] = None, fold_stride: int = 1):
    """Yield the exact ordered unit selection used by the harness and CLI."""
    if fold_stride < 1:
        raise ValueError("fold_stride must be >= 1")
    seen = 0
    for loader in loaders:
        for unit in loader.units():
            if unit.fold % fold_stride:
                continue
            if limit is not None and seen >= limit:
                return
            seen += 1
            yield loader, unit


def validate_protocol_units(suite: str, units: list) -> None:
    """Reject incomplete or changed membership for an official suite."""
    if suite not in OFFICIAL_SUITE_COUNTS:
        raise ValueError(f"unknown official suite: {suite!r}")
    expected_datasets, expected_units = OFFICIAL_SUITE_COUNTS[suite]
    actual_datasets = len({unit.dataset for unit in units})
    identities = {
        (unit.source, unit.dataset, unit.fold, unit.meta.openml_task_id)
        for unit in units
    }
    if any(unit.source != suite for unit in units) or len(identities) != len(units):
        raise RuntimeError(f"{suite} contains duplicate or mismatched unit identities")
    if (actual_datasets, len(units)) != (expected_datasets, expected_units):
        raise RuntimeError(
            f"{suite} protocol changed: found {actual_datasets} datasets/{len(units)} units; "
            f"expected {expected_datasets}/{expected_units}"
        )


@lru_cache(maxsize=1)
def _source_tree_sha256() -> str:
    """Hash shipped package code/config/list files so dirty code cannot resume."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".csv"}
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _identity(record: dict) -> tuple:
    return tuple(record.get(field) for field in _IDENTITY_FIELDS)


def _read_existing(path: Path) -> tuple[set[tuple], list[dict]]:
    if not path.exists():
        return set(), []
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    valid_lines = []
    for index, line in enumerate(lines):
        if not line.strip():
            valid_lines.append(line)
            continue
        try:
            rows.append(json.loads(line))
            valid_lines.append(line)
        except json.JSONDecodeError:
            is_partial_tail = index == len(lines) - 1 and not line.endswith(("\n", "\r"))
            if is_partial_tail:
                # A process may die between write(2) calls. Remove only that
                # unterminated tail before appending resumed records.
                path.write_text("".join(valid_lines))
                break
            raise
    if valid_lines and not valid_lines[-1].endswith(("\n", "\r")):
        path.write_text("".join(valid_lines) + "\n")
    return {_identity(row) for row in rows if not row.get("error")}, rows


def _sanitize(value):
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _path_safe_error(exc: Exception, private_path=None, placeholder="<local-path>") -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message.replace(str(private_path), placeholder) if private_path else message


def _row(
    *, unit, name, metadata, data_fingerprint, source_tree_sha256, seed, impute,
    context_cap, test_cap, n_train=None, n_test=None, n_features=None,
    metrics=None, error=None,
) -> EvalResultRow:
    memory_policy = metadata.get("memory_policy") or {}
    return EvalResultRow(
        model=name,
        source=unit.source,
        dataset=unit.dataset,
        fold=unit.fold,
        n_folds=unit.n_folds,
        repeat=unit.meta.repeat,
        fold_in_repeat=unit.meta.fold_in_repeat,
        n_repeats=unit.meta.n_repeats,
        n_folds_per_repeat=unit.meta.n_folds_per_repeat,
        openml_task_id=unit.meta.openml_task_id,
        openml_dataset_id=unit.meta.openml_dataset_id,
        data_fingerprint=data_fingerprint,
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        r2=float(metrics["r2"]) if metrics else None,
        rmse=float(metrics["rmse"]) if metrics else None,
        mae=float(metrics["mae"]) if metrics else None,
        error=error,
        seed=seed,
        impute=impute,
        context_cap=context_cap,
        test_cap=test_cap,
        checkpoint_sha256=metadata.get("checkpoint_sha256"),
        reg_config_sha256=metadata.get("reg_config_sha256"),
        source_tree_sha256=source_tree_sha256,
        elements_budget=memory_policy.get("elements_budget"),
        allow_subsample=memory_policy.get("allow_subsample"),
        model_metadata=metadata,
    )


def _write_row(sink, rows, done, key, row: EvalResultRow) -> None:
    record = _sanitize(row.model_dump())
    rows.append(record)
    done.add(key)
    sink.write(json.dumps(record, allow_nan=False) + "\n")
    sink.flush()


def run_benchmark(
    loaders: Iterable,
    model_registry,
    *,
    out_jsonl: str,
    limit: Optional[int] = None,
    fold_stride: int = 1,
    resume: bool = True,
) -> pd.DataFrame:
    """Evaluate every registered model on the selected official loader units."""
    loaders = list(loaders)
    model_names = model_registry.list_models()
    expected_policy = {
        "elements_budget": OFFICIAL_ELEMENTS_BUDGET,
        "allow_subsample": OFFICIAL_ALLOW_SUBSAMPLE,
    }
    for name in model_names:
        policy = (getattr(model_registry.get(name), "metadata", {}) or {}).get("memory_policy")
        if policy != expected_policy:
            raise ValueError(f"{name!r} must use the official memory policy {expected_policy}")

    seed = OFFICIAL_SEED
    impute = OFFICIAL_IMPUTE
    output = Path(out_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output.unlink(missing_ok=True)
    done, rows = _read_existing(output)
    source_tree_sha256 = _source_tree_sha256()
    fingerprints = {
        id(loader): loader.fingerprint() if hasattr(loader, "fingerprint") else None
        for loader in loaders
    }

    try:
        with output.open("a") as sink:
            for loader, unit in iter_units(loaders, limit, fold_stride):
                context_cap = getattr(loader, "ctx_cap", None)
                test_cap = getattr(loader, "test_cap", None)
                data_fingerprint = fingerprints[id(loader)]
                pending = []
                for name in model_names:
                    entry = model_registry.get(name)
                    metadata = dict(getattr(entry, "metadata", {}) or {})
                    candidate = _row(
                        unit=unit,
                        name=name,
                        metadata=metadata,
                        data_fingerprint=data_fingerprint,
                        source_tree_sha256=source_tree_sha256,
                        seed=seed,
                        impute=impute,
                        context_cap=context_cap,
                        test_cap=test_cap,
                    )
                    key = _identity(candidate.model_dump())
                    if key not in done:
                        pending.append((name, entry, metadata, key))
                if not pending:
                    continue

                try:
                    split = loader.materialize(unit)
                    rng = np.random.default_rng(_unit_seed(unit.source, unit.dataset, unit.fold, seed))
                    X_train, y_train = _subsample(split.X_train, split.y_train, context_cap, rng)
                    X_test, y_test = _subsample(split.X_test, split.y_test, test_cap, rng)
                    X_train, X_test = _apply_impute(X_train, X_test, impute)
                except Exception as exc:
                    error = f"MaterializationError: {_path_safe_error(exc, unit.meta.source_path)}"
                    for name, _, metadata, key in pending:
                        row = _row(
                            unit=unit,
                            name=name,
                            metadata=metadata,
                            data_fingerprint=data_fingerprint,
                            source_tree_sha256=source_tree_sha256,
                            seed=seed,
                            impute=impute,
                            context_cap=context_cap,
                            test_cap=test_cap,
                            error=error,
                        )
                        _write_row(sink, rows, done, key, row)
                    continue

                for name, entry, metadata, key in pending:
                    metrics = None
                    error = None
                    try:
                        with strict_pipeline(SvdFallbackWarning):
                            prediction = entry.wrapper.predict_regression(X_train, y_train, X_test)
                        metrics = compute_reg_metrics(y_test, np.asarray(prediction).reshape(-1))
                    except Exception as exc:
                        error = _path_safe_error(
                            exc,
                            getattr(entry.wrapper, "model_path", None),
                            "<checkpoint>",
                        )
                    row = _row(
                        unit=unit,
                        name=name,
                        metadata=metadata,
                        data_fingerprint=data_fingerprint,
                        source_tree_sha256=source_tree_sha256,
                        seed=seed,
                        impute=impute,
                        context_cap=context_cap,
                        test_cap=test_cap,
                        n_train=len(y_train),
                        n_test=len(y_test),
                        n_features=split.n_features,
                        metrics=metrics,
                        error=error,
                    )
                    _write_row(sink, rows, done, key, row)
    finally:
        if hasattr(model_registry, "cleanup_all"):
            model_registry.cleanup_all()

    frame = pd.DataFrame(rows)
    if not frame.empty:
        for field in _IDENTITY_FIELDS:
            if field not in frame:
                frame[field] = None
        frame = frame.drop_duplicates(subset=list(_IDENTITY_FIELDS), keep="last").reset_index(drop=True)
    return frame
