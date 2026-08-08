"""Command-line runner for the three official public regression protocols."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from synthefy_nori import __version__
from synthefy_nori.api import config_path
from synthefy_nori.evaluation.harness import (
    OFFICIAL_ALLOW_SUBSAMPLE,
    OFFICIAL_ELEMENTS_BUDGET,
    OFFICIAL_IMPUTE,
    OFFICIAL_PROTOCOL,
    OFFICIAL_SEED,
    _source_tree_sha256,
    run_benchmark,
    validate_protocol_units,
)
from synthefy_nori.evaluation.loaders import OpenMLTaskLoader, TabArenaLoader, TalentNativeLoader
from synthefy_nori.evaluation.models import ModelRegistry
from synthefy_nori.hf import download_checkpoint

DEFAULT_CONFIG = config_path("reg_allordinal_poly10_adaptive_svd256.json")
EXPECTED_CONFIG_SHA256 = "134fe355510887086a0d55a419400916c82d713e8b780037d9779c280f0c25f6"
EXPECTED_CHECKPOINT_REVISION = {
    "nori-6m": "7d55529ac3cbf5e3ba1b8129605f391c874a70da",
    "nori-30m": "b9b2a734315225a8188afcfd997ad9d5a4fd9466",
}
EXPECTED_CHECKPOINT_SHA256 = {
    "nori-6m": "a13b2bc31d8db24d17bae6d04844e0adf669e446087b0b7a34c7b05045d61323",
    "nori-30m": "818433f8af12c1137b96d9ff47e109b4eef5818d4e52a9656b2e573dbf13b74d",
}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checkpoint(raw: str) -> tuple[str, str]:
    if ":" in raw:
        label, path = raw.split(":", 1)
        if not label or not path:
            raise ValueError("--checkpoint must be PATH or LABEL:PATH")
        return label, path
    return Path(raw).stem, raw


def _build_loaders(suites: list[str], *, talent_root: str, openml_cache_dir: str | None):
    loaders = []
    for suite in suites:
        if suite == "talent":
            loaders.append(TalentNativeLoader(root=talent_root))
        elif suite == "openml-ctr23":
            loaders.append(OpenMLTaskLoader.from_ctr23(cache_dir=openml_cache_dir))
        elif suite == "tabarena":
            loaders.append(TabArenaLoader(cache_dir=openml_cache_dir))
    return loaders


def _print_protocol_counts(loaders):
    print(f"Protocol: {OFFICIAL_PROTOCOL}")
    enumerated = []
    for loader in loaders:
        units = list(loader.units())
        validate_protocol_units(loader.name, units)
        enumerated.extend((loader, unit) for unit in units)
        datasets = {unit.dataset for unit in units}
        print(f"  {loader.name:14s} datasets={len(datasets):3d} units={len(units):4d}")
    return enumerated


def _selected_units(enumerated, *, limit: int | None, fold_stride: int):
    if fold_stride < 1:
        raise ValueError("--fold-stride must be >= 1")
    selected = []
    for loader, unit in enumerated:
        if unit.fold % fold_stride:
            continue
        if limit is not None and len(selected) >= limit:
            break
        selected.append((loader, unit))
    return selected


def _current_invocation_rows(
    frame,
    *,
    selected,
    registry,
    reg_config_sha256: str,
):
    """Return exactly this invocation's unit/model identities from a resume file."""
    model_names = registry.list_models()
    if frame.empty:
        expected = len(selected) * len(model_names)
        if expected:
            raise RuntimeError(f"incomplete result set for this invocation: found 0 of {expected} unit/model rows")
        return frame
    checkpoint_by_model = {
        name: registry.get(name).metadata["checkpoint_sha256"]
        for name in model_names
    }
    fingerprint_by_source = {
        loader.name: loader.fingerprint() if hasattr(loader, "fingerprint") else None
        for loader, _ in selected
    }
    unit_keys = {
        (
            unit.source,
            unit.dataset,
            unit.fold,
            unit.meta.openml_task_id if unit.meta.openml_task_id is not None else -1,
        )
        for _, unit in selected
    }
    frame_task_ids = pd.to_numeric(frame["openml_task_id"], errors="coerce").fillna(-1).astype(int)
    frame_unit_keys = zip(frame["source"], frame["dataset"], frame["fold"], frame_task_ids)
    selected_mask = pd.Series(
        [key in unit_keys for key in frame_unit_keys],
        index=frame.index,
        dtype=bool,
    )
    current = frame[
        selected_mask
        & (frame["protocol"] == OFFICIAL_PROTOCOL)
        & frame["model"].isin(model_names)
        & (frame["checkpoint_sha256"] == frame["model"].map(checkpoint_by_model))
        & (frame["reg_config_sha256"] == reg_config_sha256)
        & (frame["synthefy_nori_version"] == __version__)
        & (frame["source_tree_sha256"] == _source_tree_sha256())
        & (frame["data_fingerprint"] == frame["source"].map(fingerprint_by_source))
        & (frame["seed"] == OFFICIAL_SEED)
        & (frame["impute"] == OFFICIAL_IMPUTE)
        & (frame["elements_budget"] == OFFICIAL_ELEMENTS_BUDGET)
        & (frame["allow_subsample"] == OFFICIAL_ALLOW_SUBSAMPLE)
    ]
    identity = ["source", "dataset", "fold", "openml_task_id", "model"]
    if current.duplicated(subset=identity).any():
        raise RuntimeError("resume file contains duplicate rows for this invocation")
    expected = len(unit_keys) * len(model_names)
    if len(current) != expected:
        raise RuntimeError(
            f"incomplete result set for this invocation: found {len(current)} of {expected} unit/model rows"
        )
    return current


def _print_summary(frame) -> None:
    successful = frame[frame["error"].isna() & frame["r2"].notna()]
    if successful.empty:
        print("No successful scores.")
        return
    per_dataset = (
        successful.groupby(["model", "source", "dataset"], as_index=False)["r2"]
        .mean()
    )
    print("\nR2 by source (fold mean per dataset, then dataset mean/median):")
    for (model, source), group in per_dataset.groupby(["model", "source"]):
        units = len(successful[(successful["model"] == model) & (successful["source"] == source)])
        print(
            f"  {model:12s} {source:14s} datasets={len(group):3d} units={units:4d} "
            f"mean={group['r2'].mean():.4f} median={group['r2'].median():.4f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        choices=("talent", "openml-ctr23", "tabarena"),
        help="Official suite to run; repeatable. Default: all three.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=("nori-6m", "nori-30m"),
        help="Public model variant; repeatable. Default: nori-6m.",
    )
    parser.add_argument("--checkpoint", action="append", default=[], help="Local PATH or LABEL:PATH; repeatable.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/eval/official_results.jsonl")
    parser.add_argument("--talent-root", default="cache/talent/data")
    parser.add_argument("--openml-cache-dir", default="cache/openml")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test unit limit (non-canonical).")
    parser.add_argument("--fold-stride", type=int, default=1, help="Smoke-test fold stride (non-canonical).")
    parser.add_argument("--fresh", action="store_true", help="Overwrite the output instead of resuming it.")
    parser.add_argument("--dry-run", action="store_true", help="Enumerate protocol units without loading a model.")
    args = parser.parse_args(argv)

    suites = args.suite or ["talent", "openml-ctr23", "tabarena"]
    if len(suites) != len(set(suites)):
        raise ValueError("each --suite may be selected only once")
    loaders = _build_loaders(
        suites,
        talent_root=args.talent_root,
        openml_cache_dir=args.openml_cache_dir,
    )
    enumerated = _print_protocol_counts(loaders)
    selected = _selected_units(enumerated, limit=args.limit, fold_stride=args.fold_stride)
    if args.dry_run:
        return

    reg_config_sha256 = _file_sha256(DEFAULT_CONFIG)
    if reg_config_sha256 != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            f"bundled SVD-256 config hash changed: {reg_config_sha256}; expected {EXPECTED_CONFIG_SHA256}"
        )
    memory_policy = {
        "elements_budget": OFFICIAL_ELEMENTS_BUDGET,
        "allow_subsample": OFFICIAL_ALLOW_SUBSAMPLE,
    }
    print(f"Config SHA256: {reg_config_sha256}")
    print(f"Memory policy: {memory_policy}")

    registry = ModelRegistry(device=args.device)
    labels = set()
    for selector in args.model or ([] if args.checkpoint else ["nori-6m"]):
        if selector in labels:
            raise ValueError(f"duplicate model selector: {selector!r}")
        revision = EXPECTED_CHECKPOINT_REVISION[selector]
        checkpoint = download_checkpoint(model=selector, revision=revision)
        checkpoint_sha256 = _file_sha256(checkpoint)
        expected = EXPECTED_CHECKPOINT_SHA256[selector]
        if checkpoint_sha256 != expected:
            raise RuntimeError(
                f"{selector} checkpoint hash changed: {checkpoint_sha256}; expected {expected}"
            )
        labels.add(selector)
        registry.add_checkpoint(
            selector,
            checkpoint,
            device=args.device,
            reg_config=DEFAULT_CONFIG,
            memory_policy=memory_policy,
            metadata={
                "model_selector": selector,
                "hf_revision": revision,
                "checkpoint_sha256": checkpoint_sha256,
                "reg_config_sha256": reg_config_sha256,
                "synthefy_nori_version": __version__,
            },
        )

    for raw in args.checkpoint:
        label, checkpoint = _parse_checkpoint(raw)
        if label in labels:
            raise ValueError(f"duplicate model label: {label!r}")
        labels.add(label)
        registry.add_checkpoint(
            label,
            checkpoint,
            device=args.device,
            reg_config=DEFAULT_CONFIG,
            memory_policy=memory_policy,
            metadata={
                "model_selector": None,
                "checkpoint_sha256": _file_sha256(checkpoint),
                "reg_config_sha256": reg_config_sha256,
                "synthefy_nori_version": __version__,
            },
        )

    frame = run_benchmark(
        loaders,
        registry,
        out_jsonl=args.output,
        limit=args.limit,
        fold_stride=args.fold_stride,
        resume=not args.fresh,
    )
    current = _current_invocation_rows(
        frame,
        selected=selected,
        registry=registry,
        reg_config_sha256=reg_config_sha256,
    )
    failures = current[current["error"].notna()]
    if not failures.empty:
        print(f"\n{len(failures)} unit/model evaluations failed; see {args.output}.")
        raise SystemExit(1)
    _print_summary(current)
    print(f"\nResults JSONL: {args.output}")


if __name__ == "__main__":
    main()
