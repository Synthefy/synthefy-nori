"""Evaluation runner with latency/speed benchmarking.

Runs all registered models on all loaded datasets, collecting:
  - Classification: AUC, accuracy, F1, log-loss, ECE
  - Regression: R2, RMSE, MAE
  - Latency: wall-clock time per dataset, throughput (samples/sec)
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, r2_score,
    roc_auc_score, mean_absolute_error,
)
from sklearn.model_selection import train_test_split
try:
    from sklearn.metrics import root_mean_squared_error as rmse_score
except ImportError:
    from sklearn.metrics import mean_squared_error
    import functools
    rmse_score = functools.partial(mean_squared_error, squared=False)

from synthefy_tabular.evaluation.datasets import DatasetEntry, DatasetRegistry
from synthefy_tabular.evaluation.models import ModelEntry, ModelRegistry


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_auc(y_true, y_prob, multi_class="ovo"):
    try:
        n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2
        if n_classes > 2:
            return roc_auc_score(y_true, y_prob, multi_class=multi_class)
        else:
            prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            return roc_auc_score(y_true, prob)
    except (ValueError, IndexError):
        return float("nan")


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    else:
        confidences = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        predictions = (confidences >= 0.5).astype(int)

    accuracies = (predictions == y_true)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin])) * prop
    return ece


def compute_cls_metrics(y_true, y_prob):
    """Compute all classification metrics."""
    y_pred = np.argmax(y_prob, axis=1)
    n_classes = y_prob.shape[1] if y_prob.ndim == 2 else len(np.unique(y_true))
    avg = "macro" if n_classes > 2 else "binary"

    return {
        "auc": compute_auc(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "log_loss": log_loss(y_true, y_prob, labels=list(range(n_classes))),
        "ece": compute_ece(y_true, y_prob),
    }


def compute_reg_metrics(y_true, y_pred):
    """Compute all regression metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(rmse_score(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of evaluating one model on one dataset."""
    model_name: str
    dataset_name: str
    dataset_source: str
    task_type: str
    metrics: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0
    n_classes: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self):
        d = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "source": self.dataset_source,
            "task_type": self.task_type,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "latency_ms": round(self.latency_ms, 1),
            "throughput_sps": round(self.throughput_samples_per_sec, 1),
            "error": self.error,
        }
        d.update(self.metrics)
        return d


# ---------------------------------------------------------------------------
# Eval Runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Runs evaluation across all models and datasets."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        dataset_registry: DatasetRegistry,
        output_dir: str = "./results/eval",
        warmup_runs: int = 0,
        skip_on_error: bool = True,
        verbose: bool = True,
        max_samples: int = 50_000,
        cache_dir: Optional[str] = None,
        no_cache: bool = False,
        no_cache_models: Optional[Set[str]] = None,
        parallel_models: int = 1,
        allow_single_gpu_parallel: bool = False,
        gpu_mem_gb: Optional[float] = None,
    ):
        self.models = model_registry
        self.datasets = dataset_registry
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.warmup_runs = warmup_runs
        self.skip_on_error = skip_on_error
        self.verbose = verbose
        self.max_samples = max_samples
        # None = uncapped (large-GPU protocol): train rows are bounded only by
        # max_samples. Set a GiB value to enable the memory-model train cap on
        # smaller GPUs (lowers results on large tables).
        self.gpu_mem_gb = gpu_mem_gb
        self.results: List[EvalResult] = []

        # Result cache
        self.no_cache = no_cache
        self.no_cache_models = no_cache_models or set()
        self.parallel_models = max(1, int(parallel_models))
        self.allow_single_gpu_parallel = bool(allow_single_gpu_parallel)
        self._cache_dir = Path(cache_dir) if cache_dir else Path("cache/eval_cache")
        if not self.no_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_hits = 0
        self._cache_misses = 0

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, model_name: str, dataset_name: str, source: str,
                   task_type: str) -> str:
        """Deterministic cache key from evaluation parameters."""
        raw = (f"{model_name}|{dataset_name}|{source}|{task_type}"
               f"|max={self.max_samples}|memcap={self.gpu_mem_gb}")
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _cache_get(self, model_name: str, dataset_name: str, source: str,
                   task_type: str) -> Optional[EvalResult]:
        """Return cached EvalResult or None."""
        if self.no_cache or model_name in self.no_cache_models:
            return None
        key = self._cache_key(model_name, dataset_name, source, task_type)
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            result = EvalResult(
                model_name=data["model"],
                dataset_name=data["dataset"],
                dataset_source=data["source"],
                task_type=data["task_type"],
                n_train=data.get("n_train", 0),
                n_test=data.get("n_test", 0),
                n_features=data.get("n_features", 0),
                n_classes=data.get("n_classes"),
                latency_ms=data.get("latency_ms", 0.0),
                throughput_samples_per_sec=data.get("throughput_sps", 0.0),
                error=data.get("error"),
            )
            for metric_key in ("auc", "accuracy", "f1", "log_loss", "ece",
                               "r2", "rmse", "mae"):
                if metric_key in data and data[metric_key] is not None:
                    result.metrics[metric_key] = data[metric_key]
            return result
        except (json.JSONDecodeError, KeyError):
            return None

    def _cache_put(self, result: EvalResult) -> None:
        """Write an EvalResult to cache."""
        if self.no_cache or result.model_name in self.no_cache_models:
            return
        if result.error is not None:
            return
        key = self._cache_key(result.model_name, result.dataset_name,
                              result.dataset_source, result.task_type)
        data = result.to_dict()
        self._cache_path(key).write_text(json.dumps(data, default=str))

    @classmethod
    def clear_cache(cls, cache_dir: str = "cache/eval_cache") -> int:
        """Delete all cached results. Returns number of files removed."""
        p = Path(cache_dir)
        if not p.exists():
            return 0
        n = 0
        for f in p.glob("*.json"):
            f.unlink()
            n += 1
        return n

    def _run_one_model(self, model_name: str, model_entry: ModelEntry,
                       all_datasets: List[DatasetEntry], start_idx: int, total: int):
        """Run one model across all datasets; returns results + cache stats."""
        local_results = []
        local_hits = 0
        local_misses = 0
        idx = start_idx

        print(f"\n--- Model: {model_name} ---")
        for ds in all_datasets:
            prefix = f"  [{idx}/{total}]"
            idx += 1

            # Check cache first
            cached = self._cache_get(model_name, ds.name, ds.source, ds.task_type)
            if cached is not None:
                local_results.append(cached)
                local_hits += 1
                if self.verbose:
                    metric_str = "  ".join(
                        f"{k}={v:.4f}" for k, v in cached.metrics.items()
                        if isinstance(v, float) and np.isfinite(v))
                    print(f"{prefix} {ds.source}/{ds.name} ({ds.task_type}): "
                          f"{metric_str}  [CACHED]")
                continue

            local_misses += 1
            result = self._eval_one(model_entry, ds, prefix)
            local_results.append(result)
            self._cache_put(result)

            if self.verbose and result.error is None:
                metric_str = "  ".join(f"{k}={v:.4f}" for k, v in result.metrics.items()
                                       if isinstance(v, float) and np.isfinite(v))
                print(f"{prefix} {ds.source}/{ds.name} ({ds.task_type}): "
                      f"{metric_str}  [{result.latency_ms:.0f}ms]")
            elif result.error:
                print(f"{prefix} {ds.source}/{ds.name}: ERROR - {result.error[:120]}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_entry.wrapper.cleanup()
        return local_results, local_hits, local_misses

    def run(
        self,
        model_names: Optional[List[str]] = None,
        dataset_names: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run evaluation and return results as a DataFrame.

        Args:
            model_names: Subset of models to run (None = all).
            dataset_names: Subset of datasets to run (None = all).
            sources: Filter datasets by source (e.g. ["tabarena", "ctu"]).
            task_types: Filter by task type (e.g. ["classification"]).
        """
        models = model_names or self.models.list_models()
        all_datasets = list(self.datasets.datasets.values())

        if dataset_names:
            all_datasets = [d for d in all_datasets if d.name in dataset_names
                           or f"{d.source}/{d.name}" in dataset_names]
        if sources:
            all_datasets = [d for d in all_datasets if d.source in sources]
        if task_types:
            all_datasets = [d for d in all_datasets if d.task_type in task_types]

        total = len(models) * len(all_datasets)
        cache_status = "OFF" if self.no_cache else f"ON ({self._cache_dir})"
        print(f"\n{'='*70}")
        print(f"  Synthefy Tabular Evaluation")
        print(f"  Models: {len(models)}  |  Datasets: {len(all_datasets)}  |  Total runs: {total}")
        print(f"  Cache: {cache_status}")
        print(f"  Parallel model workers: {self.parallel_models}")
        if self.allow_single_gpu_parallel:
            print("  Single-GPU parallel override: ON")
        if self.no_cache_models:
            print(f"  Cache bypass: {', '.join(sorted(self.no_cache_models))}")
        print(f"{'='*70}\n")

        model_entries = []
        for i, model_name in enumerate(models):
            model_entry = self.models.get(model_name)
            if model_entry is None:
                print(f"  [WARN] Model not found: {model_name}")
                continue
            start_idx = i * len(all_datasets) + 1
            model_entries.append((model_name, model_entry, start_idx))

        # Single-GPU safety: if multiple CUDA-bound models share one device,
        # concurrent execution is usually slower/unstable.
        effective_workers = min(self.parallel_models, max(1, len(model_entries)))
        cuda_devices = []
        for _model_name, model_entry, _start_idx in model_entries:
            dev = (model_entry.wrapper.device_str or "").lower()
            if dev.startswith("cuda"):
                cuda_devices.append(dev)
        if (
            not self.allow_single_gpu_parallel
            and effective_workers > 1
            and len(cuda_devices) > 1
            and len(set(cuda_devices)) <= 1
        ):
            print(
                "  [WARN] parallel-models>1 with multiple models on a single CUDA device. "
                "Falling back to 1 worker for stability."
            )
            effective_workers = 1

        if effective_workers <= 1 or len(model_entries) <= 1:
            for model_name, model_entry, start_idx in model_entries:
                local_results, local_hits, local_misses = self._run_one_model(
                    model_name, model_entry, all_datasets, start_idx, total
                )
                self.results.extend(local_results)
                self._cache_hits += local_hits
                self._cache_misses += local_misses
                # Hard cleanup between models to prevent CUDA state leaks
                model_entry.wrapper.cleanup()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as ex:
                futures = {
                    ex.submit(
                        self._run_one_model,
                        model_name,
                        model_entry,
                        all_datasets,
                        start_idx,
                        total,
                    ): model_name
                    for model_name, model_entry, start_idx in model_entries
                }
                for fut in as_completed(futures):
                    model_name = futures[fut]
                    try:
                        local_results, local_hits, local_misses = fut.result()
                        self.results.extend(local_results)
                        self._cache_hits += local_hits
                        self._cache_misses += local_misses
                    except Exception as e:
                        print(f"  [WARN] Parallel worker failed for {model_name}: {e}")

        if not self.no_cache:
            print(f"\n  Cache: {self._cache_hits} hits, {self._cache_misses} computed")

        # Save results
        df = self.results_dataframe()
        results_path = self.output_dir / "all_results.csv"
        df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        return df

    @staticmethod
    def _subsample_train(X_train, y_train, max_samples, task_type="classification"):
        """Subsample training data if it exceeds max_samples (stratified for cls)."""
        if X_train.shape[0] <= max_samples:
            return X_train, y_train
        rng = np.random.RandomState(42)
        if task_type == "classification":
            try:
                _, X_sub, _, y_sub = train_test_split(
                    X_train, y_train, test_size=max_samples,
                    random_state=42, stratify=y_train,
                )
                return X_sub, y_sub
            except ValueError:
                pass  # Fall through to random sampling
        idx = rng.choice(X_train.shape[0], max_samples, replace=False)
        return X_train[idx], y_train[idx]

    @staticmethod
    def _compute_max_train(n_test, n_features, gpu_mem_gb=70.0, features_per_group=2):
        """Estimate the maximum training samples that will fit in GPU memory.

        Model memory is dominated by:
        - Feature attention:  O(n_samples * n_groups^2) per layer
        - Sample attention:   O(n_samples^2 * n_heads) per layer
        - Embeddings:         O(n_samples * n_groups * embed_dim)

        We use an empirically-calibrated formula: the peak allocation
        in bytes is approximately  16 * n_total * n_groups * embed_dim * n_layers
        for the embedding/feature-attention path, plus 4 * n_total^2 * n_heads
        for sample attention.  We target 60% of GPU memory to leave room for
        weights, optimizer state, and fragmentation.
        """
        n_groups = max((n_features + features_per_group - 1) // features_per_group, 1)
        mem_bytes = gpu_mem_gb * (1024 ** 3) * 0.45

        # Empirical calibration from OOM failures on 80GB H100:
        #   Fashion-MNIST (784 feat, 10K samp) -> 16.8 GiB alloc => ~4600 bytes/sample/group
        #   CIFAR-10 (3072 feat, 10K samp) -> 55.4 GiB alloc => ~3870 bytes/sample/group
        # Use ~4000 bytes/sample/group as the per-group embedding + attention cost
        per_sample_bytes = 4000.0 * n_groups

        # Sample-attention quadratic term: ~32 bytes per sample-pair (6 heads * ~5 bytes)
        # This dominates for low-feature, high-sample datasets
        # We solve: per_sample_bytes * n_total + 32 * n_total^2 < mem_bytes
        # Approximate: n_total < mem_bytes / (per_sample_bytes + 32 * n_total)
        # Use quadratic formula: 32*n^2 + per_sample_bytes*n - mem_bytes = 0
        a = 32.0
        b = per_sample_bytes
        c = -mem_bytes
        discriminant = b * b - 4 * a * c
        max_total = int((-b + discriminant ** 0.5) / (2 * a))

        max_train = max(max_total - n_test, 200)
        return max_train

    def _eval_one(self, model_entry: ModelEntry, ds: DatasetEntry, prefix: str) -> EvalResult:
        """Evaluate one model on one dataset."""
        max_train = self.max_samples
        if self.gpu_mem_gb:
            # Memory-capped mode for smaller GPUs: bound train rows with the
            # memory model so high-dim / large-N datasets do not OOM.
            mem_max = self._compute_max_train(
                ds.n_test, ds.n_features, gpu_mem_gb=self.gpu_mem_gb)
            max_train = min(max_train, mem_max)
        X_train, y_train = self._subsample_train(
            ds.X_train, ds.y_train, max_train, ds.task_type,
        )

        result = EvalResult(
            model_name=model_entry.name,
            dataset_name=ds.name,
            dataset_source=ds.source,
            task_type=ds.task_type,
            n_train=X_train.shape[0],
            n_test=ds.n_test,
            n_features=ds.n_features,
            n_classes=ds.n_classes,
        )

        try:
            wrapper = model_entry.wrapper

            # Set CUDA device to match the model's device so that any
            # implicit cuda ops (torch.zeros, autocast, etc.) land on
            # the correct GPU instead of defaulting to cuda:0.
            model_device = getattr(wrapper, 'device', None) or getattr(wrapper, '_device', None)
            if model_device is not None:
                dev = torch.device(model_device) if isinstance(model_device, str) else model_device
                if dev.type == 'cuda':
                    torch.cuda.set_device(dev)

            # Warmup (for GPU timing stability)
            for _ in range(self.warmup_runs):
                if ds.task_type == "classification":
                    wrapper.predict_classification(
                        X_train[:min(100, len(X_train))],
                        y_train[:min(100, len(y_train))],
                        ds.X_test[:min(10, ds.n_test)],
                        ds.n_classes,
                    )
                else:
                    wrapper.predict_regression(
                        X_train[:min(100, len(X_train))],
                        y_train[:min(100, len(y_train))],
                        ds.X_test[:min(10, ds.n_test)],
                    )

            # Timed run — synchronize on the model's device, not the default device
            model_device = getattr(wrapper, 'device', None) or getattr(wrapper, '_device', None)
            if torch.cuda.is_available() and model_device is not None:
                torch.cuda.synchronize(model_device)
            t_start = time.perf_counter()

            if ds.task_type == "classification":
                y_prob = wrapper.predict_classification(
                    X_train, y_train, ds.X_test, ds.n_classes
                )
                if torch.cuda.is_available() and model_device is not None:
                    torch.cuda.synchronize(model_device)
                t_end = time.perf_counter()

                # Ensure correct shape
                y_prob = np.asarray(y_prob, dtype=np.float64)
                if y_prob.ndim == 1 or y_prob.shape[1] != ds.n_classes:
                    # Pad or trim probabilities
                    if y_prob.ndim == 1:
                        y_prob_full = np.zeros((len(y_prob), ds.n_classes))
                        y_prob_full[:, 1] = y_prob
                        y_prob_full[:, 0] = 1 - y_prob
                        y_prob = y_prob_full
                    elif y_prob.shape[1] < ds.n_classes:
                        padded = np.zeros((y_prob.shape[0], ds.n_classes))
                        padded[:, :y_prob.shape[1]] = y_prob
                        y_prob = padded

                result.metrics = compute_cls_metrics(ds.y_test, y_prob)

            else:  # regression
                y_pred = wrapper.predict_regression(
                    X_train, y_train, ds.X_test
                )
                if torch.cuda.is_available() and model_device is not None:
                    torch.cuda.synchronize(model_device)
                t_end = time.perf_counter()

                y_pred = np.asarray(y_pred, dtype=np.float64).squeeze()
                result.metrics = compute_reg_metrics(ds.y_test, y_pred)

            result.latency_ms = (t_end - t_start) * 1000
            total_samples = ds.n_train + ds.n_test
            if result.latency_ms > 0:
                result.throughput_samples_per_sec = total_samples / (result.latency_ms / 1000)

        except torch.cuda.OutOfMemoryError as e:
            result.error = str(e)
            # Aggressive cleanup on OOM so subsequent datasets can run
            gc.collect()
            torch.cuda.empty_cache()
            if not self.skip_on_error:
                raise
            traceback.print_exc()

        except Exception as e:
            result.error = str(e)
            if not self.skip_on_error:
                raise
            traceback.print_exc()

        return result

    def results_dataframe(self) -> pd.DataFrame:
        """Convert all results to a pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save_per_model_results(self):
        """Save individual CSVs per model."""
        df = self.results_dataframe()
        for model_name in df["model"].unique():
            model_df = df[df["model"] == model_name]
            safe_name = model_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            path = self.output_dir / f"results_{safe_name}.csv"
            model_df.to_csv(path, index=False)
        print(f"Per-model results saved to {self.output_dir}")
