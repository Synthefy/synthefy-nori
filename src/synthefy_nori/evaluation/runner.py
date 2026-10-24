"""Evaluation runner with latency/speed benchmarking.

Runs all registered models on all loaded datasets, collecting:
  - Regression: R2, RMSE, MAE
  - Predictive distribution (opt-in, ``distribution_metrics=True``): CRPS,
    central-interval coverage and width, PIT calibration
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
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from sklearn.metrics import root_mean_squared_error as rmse_score
except ImportError:
    from sklearn.metrics import mean_squared_error
    import functools

    rmse_score = functools.partial(mean_squared_error, squared=False)

from synthefy_nori.evaluation.datasets import DatasetEntry, DatasetRegistry
from synthefy_nori.evaluation.models import ModelEntry, ModelRegistry
from synthefy_nori.inference.degradation import SvdFallbackWarning, strict_pipeline


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


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


# Central-interval levels reported as cov50 / cov80 / cov90 / cov95.
DIST_COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
# Rows per chunk when computing CRPS, to bound memory on large test sets.
_DIST_ROW_CHUNK = 4096
DIST_METRIC_KEYS = (
    "crps",
    "crps_rel",
    *(f"cov{round(level * 100)}" for level in DIST_COVERAGE_LEVELS),
    "width90_rel",
    "pit_ks",
)


def _quantile_at(quantiles: np.ndarray, taus: np.ndarray, level: float) -> np.ndarray:
    """Per-row quantile at ``level``, linearly interpolated on the shared tau grid.

    Levels outside ``[taus[0], taus[-1]]`` clamp to the outermost quantile.
    """
    upper = int(np.clip(np.searchsorted(taus, level), 1, len(taus) - 1))
    lo_tau, hi_tau = taus[upper - 1], taus[upper]
    weight = float(np.clip((level - lo_tau) / (hi_tau - lo_tau), 0.0, 1.0))
    return quantiles[:, upper - 1] * (1.0 - weight) + quantiles[:, upper] * weight


def compute_dist_metrics(y_true, quantiles, taus):
    """Score a predictive distribution given as a per-row quantile bank.

    Args:
        y_true: ``(n,)`` observed targets.
        quantiles: ``(n, K)`` predicted quantiles, one row per target.
        taus: ``(K,)`` strictly increasing quantile levels in ``(0, 1)``.

    Returns a dict with:

    - ``crps``: mean CRPS, ``2 * integral of the pinball loss over tau``, integrated
      with the trapezoid rule over ``taus``. The tails outside ``[taus[0], taus[-1]]``
      are not covered, so this slightly under-counts CRPS (by about 1e-5 relative
      for the released 999-level grid on a Gaussian).
    - ``crps_rel``: ``crps / std(y_true)``, a scale-free value comparable across datasets.
    - ``cov50`` / ``cov80`` / ``cov90`` / ``cov95``: fraction of targets inside the
      central interval at that level (ideal: the level itself).
    - ``width90_rel``: mean width of the 90% interval divided by ``std(y_true)``, so
      coverage cannot be bought with uninformatively wide intervals.
    - ``pit_ks``: Kolmogorov-Smirnov distance of the PIT values (each target's level in
      its predicted CDF) from uniform; 0 is perfectly calibrated.

    Rows with a non-finite target or quantile are dropped. With fewer than two rows
    every value is NaN; the ``_rel`` values are NaN when ``y_true`` is constant.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64).reshape(-1)
    if quantiles.ndim != 2 or quantiles.shape != (len(y_true), len(taus)):
        raise ValueError(
            f"quantiles must have shape (n_targets, n_taus) = ({len(y_true)}, {len(taus)}); got {quantiles.shape}"
        )
    if len(taus) < 2 or not np.all(np.diff(taus) > 0) or taus[0] <= 0.0 or taus[-1] >= 1.0:
        raise ValueError("taus must contain at least two strictly increasing levels in (0, 1)")

    # The bank can be large (50k rows x 999 levels is 400 MB), so copy it only when needed.
    mask = np.isfinite(y_true) & np.isfinite(quantiles).all(axis=1)
    if not mask.all():
        y_true, quantiles = y_true[mask], quantiles[mask]
    if len(y_true) < 2:
        return {key: float("nan") for key in DIST_METRIC_KEYS}
    # Guard against quantile crossing: a predictive CDF must be non-decreasing.
    if not np.all(quantiles[:, 1:] >= quantiles[:, :-1]):
        quantiles = np.sort(quantiles, axis=1)
    scale = float(np.std(y_true))

    # Row chunks bound the (rows, K) temporaries: at the benchmark's 50k test rows and
    # K=999, a single pass would allocate several ~400 MB float64 arrays.
    tau_steps = np.diff(taus)
    crps_sum = 0.0
    for start in range(0, len(y_true), _DIST_ROW_CHUNK):
        diff = y_true[start : start + _DIST_ROW_CHUNK, None] - quantiles[start : start + _DIST_ROW_CHUNK]
        pinball = np.maximum(taus * diff, (taus - 1.0) * diff)
        # Trapezoid rule written out (np.trapz/np.trapezoid differ across NumPy versions).
        crps_sum += float(np.sum(pinball[:, 1:] + pinball[:, :-1], axis=0) @ tau_steps)
    crps = crps_sum / len(y_true)  # 2 * (1/2) trapezoid factors cancel

    metrics = {"crps": crps, "crps_rel": crps / scale if scale > 0 else float("nan")}
    for level in DIST_COVERAGE_LEVELS:
        lower = _quantile_at(quantiles, taus, (1.0 - level) / 2.0)
        upper = _quantile_at(quantiles, taus, (1.0 + level) / 2.0)
        metrics[f"cov{round(level * 100)}"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
    width90 = float(np.mean(_quantile_at(quantiles, taus, 0.95) - _quantile_at(quantiles, taus, 0.05)))
    metrics["width90_rel"] = width90 / scale if scale > 0 else float("nan")

    # PIT: invert each row's piecewise-linear quantile function at its target.
    pit = np.array([np.interp(y, row, taus, left=0.0, right=1.0) for y, row in zip(y_true, quantiles)])
    pit.sort()
    n = len(pit)
    ranks = np.arange(1, n + 1) / n
    metrics["pit_ks"] = float(max(np.max(ranks - pit), np.max(pit - (ranks - 1.0 / n))))
    # Keep the documented key order.
    return {key: metrics[key] for key in DIST_METRIC_KEYS}


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
        distribution_metrics: bool = False,
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
        # Opt-in: score the predictive distribution (CRPS, coverage, PIT) as well as
        # the point prediction. Off by default so published results are unchanged.
        self.distribution_metrics = bool(distribution_metrics)
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

    def _cache_key(self, model_name: str, dataset_name: str, source: str, task_type: str) -> str:
        """Deterministic cache key from evaluation parameters."""
        raw = f"{model_name}|{dataset_name}|{source}|{task_type}|max={self.max_samples}|memcap={self.gpu_mem_gb}"
        if self.distribution_metrics:
            # Separate entries, so a point-only cached result never answers a distribution
            # run. Appended only when on, so existing point-only cache keys stay valid.
            raw += "|dist=1"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _cache_get(self, model_name: str, dataset_name: str, source: str, task_type: str) -> Optional[EvalResult]:
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
                latency_ms=data.get("latency_ms", 0.0),
                throughput_samples_per_sec=data.get("throughput_sps", 0.0),
                error=data.get("error"),
            )
            for metric_key in ("r2", "rmse", "mae", *DIST_METRIC_KEYS):
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
        key = self._cache_key(result.model_name, result.dataset_name, result.dataset_source, result.task_type)
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

    def _run_one_model(
        self, model_name: str, model_entry: ModelEntry, all_datasets: List[DatasetEntry], start_idx: int, total: int
    ):
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
                        f"{k}={v:.4f}" for k, v in cached.metrics.items() if isinstance(v, float) and np.isfinite(v)
                    )
                    print(f"{prefix} {ds.source}/{ds.name} ({ds.task_type}): {metric_str}  [CACHED]")
                continue

            local_misses += 1
            result = self._eval_one(model_entry, ds, prefix)
            local_results.append(result)
            self._cache_put(result)

            if self.verbose and result.error is None:
                metric_str = "  ".join(
                    f"{k}={v:.4f}" for k, v in result.metrics.items() if isinstance(v, float) and np.isfinite(v)
                )
                print(f"{prefix} {ds.source}/{ds.name} ({ds.task_type}): {metric_str}  [{result.latency_ms:.0f}ms]")
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
    ) -> pd.DataFrame:
        """Run evaluation and return results as a DataFrame.

        Args:
            model_names: Subset of models to run (None = all).
            dataset_names: Subset of datasets to run (None = all).
            sources: Filter datasets by source (e.g. ["tabarena", "talent"]).
        """
        models = model_names or self.models.list_models()
        all_datasets = list(self.datasets.datasets.values())

        if dataset_names:
            all_datasets = [
                d for d in all_datasets if d.name in dataset_names or f"{d.source}/{d.name}" in dataset_names
            ]
        if sources:
            all_datasets = [d for d in all_datasets if d.source in sources]

        total = len(models) * len(all_datasets)
        cache_status = "OFF" if self.no_cache else f"ON ({self._cache_dir})"
        print(f"\n{'=' * 70}")
        print(f"  Nori Evaluation")
        print(f"  Models: {len(models)}  |  Datasets: {len(all_datasets)}  |  Total runs: {total}")
        print(f"  Cache: {cache_status}")
        print(f"  Parallel model workers: {self.parallel_models}")
        if self.allow_single_gpu_parallel:
            print("  Single-GPU parallel override: ON")
        if self.no_cache_models:
            print(f"  Cache bypass: {', '.join(sorted(self.no_cache_models))}")
        print(f"{'=' * 70}\n")

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
    def _subsample_train(X_train, y_train, max_samples):
        """Subsample training data if it exceeds max_samples (seeded)."""
        if X_train.shape[0] <= max_samples:
            return X_train, y_train
        rng = np.random.RandomState(42)
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
        mem_bytes = gpu_mem_gb * (1024**3) * 0.45

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
        max_total = int((-b + discriminant**0.5) / (2 * a))

        max_train = max(max_total - n_test, 200)
        return max_train

    def _predict(self, wrapper, X_train, y_train, X_test):
        """Return ``(point_predictions, (quantiles, taus) or None)`` for one dataset.

        Point-only unless ``distribution_metrics`` is on and the wrapper can return a
        distribution. Then a single distribution call supplies the quantiles and, under
        the default ``quantile_collapse="mean"``, the point predictions too (its mean is
        what ``predict_regression`` returns, to float precision). Any other collapse
        makes a different point prediction, so ``predict_regression`` still supplies it.
        A checkpoint without a quantile head is scored on its point prediction only.
        """
        predict_distribution = getattr(wrapper, "predict_distribution", None)
        if self.distribution_metrics and predict_distribution is not None:
            try:
                quantiles, taus, mean = predict_distribution(X_train, y_train, X_test)
            except NotImplementedError:
                pass  # e.g. a bar_distribution / MSE checkpoint: no quantile bank
            else:
                if getattr(wrapper, "quantile_collapse", "mean") == "mean":
                    return mean, (quantiles, taus)
                return wrapper.predict_regression(X_train, y_train, X_test), (quantiles, taus)
        return wrapper.predict_regression(X_train, y_train, X_test), None

    def _eval_one(self, model_entry: ModelEntry, ds: DatasetEntry, prefix: str) -> EvalResult:
        """Evaluate one model on one dataset."""
        max_train = self.max_samples
        if self.gpu_mem_gb:
            # Memory-capped mode for smaller GPUs: bound train rows with the
            # memory model so high-dim / large-N datasets do not OOM.
            mem_max = self._compute_max_train(ds.n_test, ds.n_features, gpu_mem_gb=self.gpu_mem_gb)
            max_train = min(max_train, mem_max)
        X_train, y_train = self._subsample_train(ds.X_train, ds.y_train, max_train)

        result = EvalResult(
            model_name=model_entry.name,
            dataset_name=ds.name,
            dataset_source=ds.source,
            task_type=ds.task_type,
            n_train=X_train.shape[0],
            n_test=ds.n_test,
            n_features=ds.n_features,
        )

        try:
            wrapper = model_entry.wrapper

            # Set CUDA device to match the model's device so that any
            # implicit cuda ops (torch.zeros, autocast, etc.) land on
            # the correct GPU instead of defaulting to cuda:0.
            model_device = getattr(wrapper, "device", None) or getattr(wrapper, "_device", None)
            if model_device is not None:
                dev = torch.device(model_device) if isinstance(model_device, str) else model_device
                if dev.type == "cuda":
                    torch.cuda.set_device(dev)

            # Warmup (for GPU timing stability)
            for _ in range(self.warmup_runs):
                with strict_pipeline(SvdFallbackWarning):
                    wrapper.predict_regression(
                        X_train[: min(100, len(X_train))],
                        y_train[: min(100, len(y_train))],
                        ds.X_test[: min(10, ds.n_test)],
                    )

            # Timed run — synchronize on the model's device, not the default device
            model_device = getattr(wrapper, "device", None) or getattr(wrapper, "_device", None)
            if torch.cuda.is_available() and model_device is not None:
                torch.cuda.synchronize(model_device)
            t_start = time.perf_counter()

            with strict_pipeline(SvdFallbackWarning):
                y_pred, distribution = self._predict(wrapper, X_train, y_train, ds.X_test)
            if torch.cuda.is_available() and model_device is not None:
                torch.cuda.synchronize(model_device)
            t_end = time.perf_counter()
            y_pred = np.asarray(y_pred, dtype=np.float64).squeeze()
            result.metrics = compute_reg_metrics(ds.y_test, y_pred)
            if distribution is not None:
                quantiles, taus = distribution
                result.metrics.update(compute_dist_metrics(ds.y_test, quantiles, taus))
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

        except SvdFallbackWarning as e:
            result.error = f"{type(e).__name__}: {e}"
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
