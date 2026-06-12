#!/usr/bin/env python
"""Benchmark SynthefyTabularRegressor across TabArena, TALENT, and OpenML.

This mirrors the regression eval loop in the outer SynthefyPFN repo
(``evaluation/datasets.py`` + ``evaluation/runner.py``), but driven through the
public ``synthefy_tabular`` package API (the same one used by ``test.py``):

    model = SynthefyTabularRegressor()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

Data sources (regression only):

  * TabArena / TALENT -- local CSV cache dirs in the outer repo, one folder per
    dataset containing ``{name}_train.csv`` and ``{name}_test.csv`` with the
    target in the last column.
  * OpenML            -- a fixed list of regression dataset IDs, fetched live via
    the ``openml`` package (optional; skipped with a hint if not installed).

Metrics match the outer repo's ``compute_reg_metrics``: R2, RMSE, MAE.

Run from the synthefy-tabular dir (``uv sync`` installs a cu128 torch build on
Linux, so ``uv run`` works as-is):

    uv run python tests/test_benchmark_performance.py
    uv run python tests/test_benchmark_performance.py --suites tabarena talent
    uv run python tests/test_benchmark_performance.py --device cuda:0 \
        --output benchmarks/benchmark_results.csv
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# NOTE: torch and synthefy_tabular are imported lazily inside main(), after
# SYNTHEFY_MAX_ELEMENTS_BUDGET is set, since the predictor reads it at import.

# OpenML regression dataset IDs. Fetched live via the openml package.
OPENML_REGRESSION_IDS = [287, 422, 507, 546, 541, 1030, 23515, 42225, 42571, 43071, 43093]

# Default root holding cache/ with the benchmark CSVs. Populate it with
# `synthefy-tabular-eval --download-benchmarks` (run from the repo root).
DEFAULT_BENCH_ROOT = Path(".")


# --------------------------------------------------------------------------- #
# Metrics (mirror evaluation/runner.py::compute_reg_metrics)
# --------------------------------------------------------------------------- #
def compute_reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Preprocessing (mirror evaluation/datasets.py: categorical encode + coerce)
# --------------------------------------------------------------------------- #
def _prepare_xy(X_train_df, X_test_df):
    """Label-encode categorical columns (fit on train+test), coerce to float32."""
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()
    cat_cols = X_train_df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_df[col], X_test_df[col]]).fillna("__MISSING__").astype(str)
        le.fit(combined)
        X_train_df[col] = le.transform(X_train_df[col].fillna("__MISSING__").astype(str))
        X_test_df[col] = le.transform(X_test_df[col].fillna("__MISSING__").astype(str))

    X_train = X_train_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32).to_numpy()
    X_test = X_test_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32).to_numpy()
    return X_train, X_test


def _subsample_train(X_train, y_train, max_samples):
    """Random train subsample (mirror runner.py regression path, RandomState(42))."""
    if max_samples is None or X_train.shape[0] <= max_samples:
        return X_train, y_train
    idx = np.random.RandomState(42).choice(X_train.shape[0], max_samples, replace=False)
    return X_train[idx], y_train[idx]


def _fit_predict(model, X_train, y_train, X_test):
    """fit + predict for one dataset, returning (pred, elapsed_seconds)."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pred, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Dataset loaders
# --------------------------------------------------------------------------- #
def _read_dataset_list(path: Path):
    """Read a benchmark_list CSV (one dataset name per line, optional header)."""
    if not path.exists():
        return []
    names = []
    for i, line in enumerate(path.read_text().splitlines()):
        name = line.strip().strip(",").strip()
        if not name:
            continue
        if i == 0 and name.lower() in ("name", "dataset", "datasets"):
            continue
        names.append(name)
    return names


def _load_csv_folder(folder: Path, name: str, source: str, test_size: float, random_state: int):
    """Load one cached dataset folder -> dict, or None if absent/unusable."""
    train_path = folder / f"{name}_train.csv"
    test_path = folder / f"{name}_test.csv"
    if not train_path.exists():
        return None
    train_df = pd.read_csv(train_path)
    if test_path.exists():
        test_df = pd.read_csv(test_path)
    else:
        train_df, test_df = train_test_split(train_df, test_size=test_size, random_state=random_state)

    X_train_df, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test_df, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
    return _finalize_entry(X_train_df, y_train, X_test_df, y_test, name, source)


def _finalize_entry(X_train_df, y_train, X_test_df, y_test, name, source):
    y_train = pd.to_numeric(pd.Series(y_train).reset_index(drop=True), errors="coerce")
    y_test = pd.to_numeric(pd.Series(y_test).reset_index(drop=True), errors="coerce")
    # Drop rows with non-numeric targets (regression).
    tr_mask = y_train.notna().to_numpy()
    te_mask = y_test.notna().to_numpy()
    X_train_df = X_train_df.reset_index(drop=True).loc[tr_mask]
    X_test_df = X_test_df.reset_index(drop=True).loc[te_mask]
    y_train = y_train[tr_mask].to_numpy(dtype=np.float64)
    y_test = y_test[te_mask].to_numpy(dtype=np.float64)
    if len(X_train_df) < 2 or len(X_test_df) < 1:
        return None

    X_train, X_test = _prepare_xy(X_train_df, X_test_df)
    return {
        "name": name,
        "source": source,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "n_features": X_train.shape[1],
    }


def _matches(name, name_filter):
    """True if name contains any filter substring (or no filter is set)."""
    return name_filter is None or any(s in name for s in name_filter)


def load_local_suite(bench_root: Path, cache_subdir: str, list_name: str, source: str,
                     test_size: float, random_state: int, name_filter=None):
    """Load all cached regression datasets for one suite (tabarena/talent)."""
    cache_dir = bench_root / "cache" / cache_subdir
    if not cache_dir.is_dir():
        print(f"  [skip] {source}: cache dir not found ({cache_dir})")
        return []

    listed = _read_dataset_list(bench_root / "benchmark_list" / list_name)
    names = listed if listed else sorted(p.name for p in cache_dir.iterdir() if p.is_dir())
    names = [n for n in names if _matches(n, name_filter)]

    entries, missing = [], 0
    for name in names:
        folder = cache_dir / name
        if not folder.is_dir():
            missing += 1
            continue
        try:
            entry = _load_csv_folder(folder, name, source, test_size, random_state)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {source}/{name}: load failed ({exc})")
            entry = None
        if entry is not None:
            entries.append(entry)
    print(f"  {source}: loaded {len(entries)} datasets"
          + (f" ({missing} listed but missing from cache)" if missing else ""))
    return entries


def load_openml_regression(test_size: float, random_state: int, max_datasets: int | None,
                           name_filter=None):
    """Fetch OpenML regression datasets live (optional; needs the openml package)."""
    try:
        import openml
    except ImportError:
        print("  [skip] openml: package not installed "
              "(`uv pip install openml` to enable OpenML datasets)")
        return []

    ids = OPENML_REGRESSION_IDS[: max_datasets] if max_datasets else OPENML_REGRESSION_IDS
    entries = []
    for did in ids:
        try:
            # Cheap metadata fetch first; only download the data if the name matches.
            ds = openml.datasets.get_dataset(did, download_data=False)
            if not _matches(ds.name, name_filter):
                continue
            ds = openml.datasets.get_dataset(did, download_data=True)
            target = ds.default_target_attribute
            X_df, y, _, _ = ds.get_data(target=target)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_df, y, test_size=test_size, random_state=random_state
            )
            entry = _finalize_entry(X_tr, y_tr, X_te, y_te, ds.name, "openml")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] openml/{did}: load failed ({exc})")
            entry = None
        if entry is not None:
            entries.append(entry)
    print(f"  openml: loaded {len(entries)} datasets")
    return entries


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT,
                        help="Outer repo holding cache/ and benchmark_list/ (default: ~/SynthefyPFN)")
    parser.add_argument("--suites", nargs="+", default=["tabarena", "talent", "openml"],
                        choices=["tabarena", "talent", "openml"],
                        help="Benchmark suites to evaluate")
    parser.add_argument("--device", default="cuda:0", help="Torch device (e.g. cuda:0, cpu)")
    parser.add_argument("--model-path", default=None,
                        help="Local checkpoint path (default: download from HF Hub)")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/benchmark_results.csv"),
                        help="Per-dataset results CSV output path")
    parser.add_argument("--max-train-samples", type=int, default=50000,
                        help="Cap on context rows per dataset (mirrors outer repo; 0 = no cap)")
    parser.add_argument("--max-openml", type=int, default=None,
                        help="Limit number of OpenML datasets (default: all)")
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Train/test split fraction when a dataset has no _test.csv")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-elements-budget", type=int, default=16_000_000,
                        help="SYNTHEFY_MAX_ELEMENTS_BUDGET for inference chunking (H200-friendly)")
    parser.add_argument("--retry-budget", type=int, default=8_000_000,
                        help="On a CUDA/inference error, retry the dataset once at this lower "
                             "element budget (forces finer test chunking). 0 disables retry.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N datasets (smoke test)")
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="Only evaluate datasets whose name contains one of these substrings")
    args = parser.parse_args()

    # The predictor re-reads this env var on every predict() call, so we can lower
    # it per-dataset for the retry path and restore it afterwards.
    os.environ.setdefault("SYNTHEFY_MAX_ELEMENTS_BUDGET", str(args.max_elements_budget))
    base_budget = int(os.environ["SYNTHEFY_MAX_ELEMENTS_BUDGET"])

    import torch
    from synthefy_tabular import SynthefyTabularRegressor

    bench_root = args.bench_root.expanduser()
    max_train = args.max_train_samples if args.max_train_samples and args.max_train_samples > 0 else None

    # ----- load datasets -------------------------------------------------- #
    print(f"Loading datasets from {bench_root} (suites: {', '.join(args.suites)})")
    datasets = []
    if "tabarena" in args.suites:
        datasets += load_local_suite(bench_root, "tabarena_reg", "tabarena_reg.csv",
                                      "tabarena", args.test_size, args.random_state, args.dataset)
    if "talent" in args.suites:
        datasets += load_local_suite(bench_root, "talent_reg", "talent_reg.csv",
                                      "talent", args.test_size, args.random_state, args.dataset)
    if "openml" in args.suites:
        datasets += load_openml_regression(args.test_size, args.random_state, args.max_openml,
                                           args.dataset)

    if args.limit:
        datasets = datasets[: args.limit]
    if not datasets:
        print("No datasets loaded -- nothing to evaluate.")
        return
    print(f"\nTotal datasets to evaluate: {len(datasets)}\n")

    # ----- build the model once (predictor is cached across fit calls) ---- #
    print(f"Loading SynthefyTabularRegressor on {args.device} ...")
    model = SynthefyTabularRegressor(model_path=args.model_path, device=args.device)

    # ----- evaluate ------------------------------------------------------- #
    rows = []
    for i, ds in enumerate(datasets, 1):
        X_train, y_train = _subsample_train(ds["X_train"], ds["y_train"], max_train)
        n_train, n_test = X_train.shape[0], ds["X_test"].shape[0]
        tag = f"[{i}/{len(datasets)}] {ds['source']}/{ds['name']}"

        pred, elapsed, err, note = None, float("nan"), "", ""
        try:
            pred, elapsed = _fit_predict(model, X_train, y_train, ds["X_test"])
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            # Big tables can trip CUDA kernel grid limits ("invalid configuration
            # argument") when a forward-pass chunk is too large. Retry once at a
            # lower element budget, which the predictor honors per-call and which
            # forces finer test chunking. Only retry if it actually shrinks.
            if args.retry_budget and args.retry_budget < base_budget:
                print(f"{tag:<60} error ({err.splitlines()[0][:50]}) -- retrying "
                      f"at budget={args.retry_budget}")
                torch.cuda.empty_cache()
                os.environ["SYNTHEFY_MAX_ELEMENTS_BUDGET"] = str(args.retry_budget)
                try:
                    pred, elapsed = _fit_predict(model, X_train, y_train, ds["X_test"])
                    err, note = "", f"retried@{args.retry_budget}"
                except Exception as exc2:  # noqa: BLE001
                    err = str(exc2)
                finally:
                    os.environ["SYNTHEFY_MAX_ELEMENTS_BUDGET"] = str(base_budget)

        if err == "" and pred is not None:
            metrics = compute_reg_metrics(ds["y_test"], pred)
            print(f"{tag:<60} R2={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}  "
                  f"MAE={metrics['mae']:.4f}  ({elapsed:.1f}s, n={n_train}/{n_test}, "
                  f"f={ds['n_features']})" + (f"  [{note}]" if note else ""))
            rows.append({
                "dataset": ds["name"], "source": ds["source"],
                "n_train": n_train, "n_test": n_test, "n_features": ds["n_features"],
                "latency_s": elapsed, **metrics, "note": note, "error": "",
            })
        else:
            print(f"{tag:<60} ERROR: {err}")
            rows.append({
                "dataset": ds["name"], "source": ds["source"],
                "n_train": n_train, "n_test": n_test, "n_features": ds["n_features"],
                "latency_s": float("nan"), "r2": float("nan"), "rmse": float("nan"),
                "mae": float("nan"), "note": note, "error": err,
            })

    # ----- save + summarize ----------------------------------------------- #
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nPer-dataset results written to {args.output}")

    ok = df[df["error"] == ""]
    if len(ok):
        print("\n=== Summary by source (mean over datasets) ===")
        by_src = ok.groupby("source").agg(
            n=("dataset", "count"),
            r2=("r2", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean"),
        )
        for src, r in by_src.iterrows():
            print(f"  {src:<12} n={int(r['n']):<4} "
                  f"R2={r['r2']:.4f}  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}")
        print(f"\n=== Overall (n={len(ok)}) ===")
        print(f"  mean R2={ok['r2'].mean():.4f}  median R2={ok['r2'].median():.4f}  "
              f"mean RMSE={ok['rmse'].mean():.4f}  mean MAE={ok['mae'].mean():.4f}")
    n_err = int((df["error"] != "").sum())
    if n_err:
        print(f"\n{n_err} dataset(s) errored -- see the 'error' column in {args.output}")


if __name__ == "__main__":
    main()
