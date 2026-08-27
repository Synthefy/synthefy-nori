#!/usr/bin/env python3
"""Paired end-to-end benchmark for Nori inference-pipeline batching.

The workload is the measured narrow public-6M case: 512 context rows, 48 raw
features, and either 256 plain-query rows or 600 hot resident-cache query rows.
Batching ON/OFF calls alternate on one fitted predictor so model loading and
machine drift are shared by both arms. Results are emitted as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


SEED = 20260818
N_TRAIN = 512
N_PLAIN_TEST = 256
N_CACHED_TEST = 600
N_FEATURES = 48
ELEMENTS_BUDGET = 36_864
PARITY_ATOL = 5e-3
PARITY_RTOL = 5e-3
GIB = 1024**3


def at_least_two(raw: str) -> int:
    value = int(raw)
    if value < 2:
        raise argparse.ArgumentTypeError("value must be at least 2")
    return value


def nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return value


def percentile(samples: list[float], fraction: float) -> float:
    """Linearly interpolated percentile without an optional statistics package."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(SEED)
    x_train = generator.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float32)
    x_test = generator.normal(size=(N_CACHED_TEST, N_FEATURES)).astype(np.float32)
    weights = generator.normal(size=N_FEATURES).astype(np.float32)
    weights /= np.linalg.norm(weights)
    noise = generator.normal(scale=0.05, size=N_TRAIN).astype(np.float32)
    y_train = x_train @ weights + 0.2 * np.sin(2.0 * x_train[:, 0]) + noise
    return x_train, y_train.astype(np.float32), x_test


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def delta(name: str, candidate, baseline) -> dict:
    candidate_array = to_numpy(candidate)
    baseline_array = to_numpy(baseline)
    if candidate_array.shape != baseline_array.shape:
        return {
            "name": name,
            "passed": False,
            "candidate_shape": list(candidate_array.shape),
            "baseline_shape": list(baseline_array.shape),
        }
    absolute = np.abs(candidate_array - baseline_array)
    tolerance = PARITY_ATOL + PARITY_RTOL * np.abs(baseline_array)
    denominator = np.maximum(np.abs(baseline_array), PARITY_ATOL)
    return {
        "name": name,
        "passed": bool(np.all(absolute <= tolerance)),
        "shape": list(candidate_array.shape),
        "max_abs": float(absolute.max(initial=0.0)),
        "mean_abs": float(absolute.mean()) if absolute.size else 0.0,
        "max_rel": float((absolute / denominator).max(initial=0.0)),
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
    }


@contextmanager
def controlled_environment():
    keys = (
        "SYNTHEFY_CACHE_MAX_GB",
        "SYNTHEFY_DISABLE_CACHED_INFERENCE",
        "SYNTHEFY_ENABLE_CACHED_INFERENCE",
        "SYNTHEFY_DISABLE_PIPELINE_BATCHING",
        "SYNTHEFY_MAX_ELEMENTS_BUDGET",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.pop("SYNTHEFY_CACHE_MAX_GB", None)
        os.environ.pop("SYNTHEFY_DISABLE_CACHED_INFERENCE", None)
        os.environ.pop("SYNTHEFY_DISABLE_PIPELINE_BATCHING", None)
        os.environ.pop("SYNTHEFY_MAX_ELEMENTS_BUDGET", None)
        os.environ["SYNTHEFY_ENABLE_CACHED_INFERENCE"] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def set_pipeline_batching(enabled: bool) -> None:
    if enabled:
        os.environ.pop("SYNTHEFY_DISABLE_PIPELINE_BATCHING", None)
    else:
        os.environ["SYNTHEFY_DISABLE_PIPELINE_BATCHING"] = "1"


def clear_context_cache(predictor) -> None:
    cache = getattr(predictor, "_context_cache", None)
    if cache is not None:
        cache.clear()


def context_cache_signature(predictor) -> dict:
    """Bundle identities, used to prove a nominal hot call did not rebuild."""
    cache = getattr(predictor, "_context_cache", None) or {}
    return {key: id(entry[3]) for key, entry in cache.items()}


def assert_hot_cache_reused(predictor, expected: dict, *, label: str) -> None:
    actual = context_cache_signature(predictor)
    if not expected:
        raise RuntimeError(f"{label} warmup retained no context cache")
    if actual != expected:
        raise RuntimeError(
            f"{label} rebuilt or replaced its context cache during the timed call: before={expected}, after={actual}"
        )


def one_predict(
    regressor,
    predictor,
    x_test: np.ndarray,
    *,
    batching: bool,
    hot_reuse: bool,
):
    set_pipeline_batching(batching)
    clear_context_cache(predictor)
    warm_signature = {}
    if hot_reuse:
        warm_predictions = regressor.predict(x_test, output_type="mean")
        torch.cuda.synchronize()
        del warm_predictions
        warm_signature = context_cache_signature(predictor)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions = regressor.predict(x_test, output_type="mean")
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    peak_gib = torch.cuda.max_memory_allocated() / GIB
    report = getattr(predictor, "memory_report_", None)
    if hot_reuse:
        assert_hot_cache_reused(
            predictor,
            warm_signature,
            label="batching-on" if batching else "batching-off",
        )
    return elapsed_ms, peak_gib, predictions, report


def distribution_predict(
    regressor,
    predictor,
    x_test: np.ndarray,
    *,
    batching: bool,
    hot_reuse: bool,
) -> dict:
    set_pipeline_batching(batching)
    clear_context_cache(predictor)
    warm_signature = {}
    if hot_reuse:
        warm_distribution = regressor.predict(x_test, output_type="full")
        torch.cuda.synchronize()
        del warm_distribution
        warm_signature = context_cache_signature(predictor)
    torch.cuda.synchronize()
    result = regressor.predict(x_test, output_type="full")
    torch.cuda.synchronize()
    if hot_reuse:
        assert_hot_cache_reused(
            predictor,
            warm_signature,
            label="batching-on distribution" if batching else "batching-off distribution",
        )
    return result


def summarize_arm(durations_ms: list[float], peaks_gib: list[float], report: dict | None) -> dict:
    return {
        "durations_ms": durations_ms,
        "median_ms": statistics.median(durations_ms),
        "p95_ms": percentile(durations_ms, 0.95),
        "peak_vram_gib": max(peaks_gib),
        "peak_vram_gib_samples": peaks_gib,
        "memory_report": report,
    }


def benchmark_case(
    regressor,
    predictor,
    *,
    name: str,
    x_test: np.ndarray,
    memory_policy,
    repeats: int,
    warmup: int,
    expected_rung: str,
    expected_query_chunk: int | None = None,
    hot_reuse: bool = False,
) -> tuple[dict, list[dict]]:
    regressor.memory_policy = memory_policy
    arms = (("batching_on", True), ("batching_off", False))

    for index in range(warmup):
        order = arms if index % 2 == 0 else tuple(reversed(arms))
        for _, enabled in order:
            one_predict(
                regressor,
                predictor,
                x_test,
                batching=enabled,
                hot_reuse=hot_reuse,
            )

    durations = {label: [] for label, _ in arms}
    peaks = {label: [] for label, _ in arms}
    point_outputs = {}
    reports = {}
    for index in range(repeats):
        order = arms if index % 2 == 0 else tuple(reversed(arms))
        for label, enabled in order:
            elapsed_ms, peak_gib, predictions, report = one_predict(
                regressor,
                predictor,
                x_test,
                batching=enabled,
                hot_reuse=hot_reuse,
            )
            if report is None or report.get("rung") != expected_rung:
                actual = None if report is None else report.get("rung")
                raise RuntimeError(f"{name}/{label} reached memory rung {actual!r}; expected {expected_rung!r}")
            if expected_query_chunk is not None and report.get("query_chunk") != expected_query_chunk:
                raise RuntimeError(
                    f"{name}/{label} used query_chunk={report.get('query_chunk')!r}; expected {expected_query_chunk}"
                )
            durations[label].append(elapsed_ms)
            peaks[label].append(peak_gib)
            point_outputs[label] = predictions
            reports[label] = report

    distributions = {
        label: distribution_predict(
            regressor,
            predictor,
            x_test,
            batching=enabled,
            hot_reuse=hot_reuse,
        )
        for label, enabled in arms
    }
    parity = [
        delta(f"{name}-point", point_outputs["batching_on"], point_outputs["batching_off"]),
        delta(
            f"{name}-distribution-quantiles",
            distributions["batching_on"]["quantiles"],
            distributions["batching_off"]["quantiles"],
        ),
        delta(
            f"{name}-distribution-mean",
            distributions["batching_on"]["mean"],
            distributions["batching_off"]["mean"],
        ),
        delta(
            f"{name}-distribution-taus",
            distributions["batching_on"]["taus"],
            distributions["batching_off"]["taus"],
        ),
    ]
    arm_results = {label: summarize_arm(durations[label], peaks[label], reports[label]) for label, _ in arms}
    baseline_ms = arm_results["batching_off"]["median_ms"]
    candidate_ms = arm_results["batching_on"]["median_ms"]
    return (
        {
            "name": name,
            "n_test": len(x_test),
            "baseline_ms": baseline_ms,
            "candidate_ms": candidate_ms,
            "speedup": baseline_ms / candidate_ms,
            "hot_context_reuse": hot_reuse,
            "arms": arm_results,
            "deltas": {
                "point": parity[0],
                "distribution_quantiles": parity[1],
                "distribution_mean": parity[2],
                "distribution_taus": parity[3],
            },
        },
        parity,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--repeats",
        type=at_least_two,
        default=5,
        help="Timed repetitions per arm (minimum 2; default: 5).",
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON result.")
    parser.add_argument(
        "--include-resident-cache",
        action="store_true",
        help="Also benchmark the 600-query exact resident-bf16 hot-reuse case.",
    )
    parser.add_argument("--gpu-cache-cap-gb", type=nonnegative_float, default=40.0)
    parser.add_argument("--host-cache-cap-gb", type=nonnegative_float, default=80.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("this benchmark requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("--device must be cuda[:INDEX]")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)

    from synthefy_nori import NoriRegressor

    x_train, y_train, x_test = make_data()
    regressor = NoriRegressor(
        model="nori-6m",
        device=device,
        augmentations=(),
    ).fit(x_train, y_train)
    predictor = regressor._get_predictor()

    cases = []
    parity = []
    plain_policy = {
        "cache": False,
        "elements_budget": ELEMENTS_BUDGET,
    }
    cached_policy = {
        "elements_budget": ELEMENTS_BUDGET,
        "cache_dtype": "bf16",
        "allow_quantization": False,
        "reuse_context_cache": True,
        "gpu_budget_absolute_gb": args.gpu_cache_cap_gb,
        "host_budget_absolute_gb": args.host_cache_cap_gb,
    }

    with controlled_environment():
        case, checks = benchmark_case(
            regressor,
            predictor,
            name="public-6m-plain-n512-q256-f48",
            x_test=x_test[:N_PLAIN_TEST],
            memory_policy=plain_policy,
            repeats=args.repeats,
            warmup=args.warmup,
            expected_rung="no_cache",
        )
        cases.append(case)
        parity.extend(checks)
        if args.include_resident_cache:
            case, checks = benchmark_case(
                regressor,
                predictor,
                name="public-6m-resident-bf16-hot-reuse-n512-q600-f48",
                x_test=x_test,
                memory_policy=cached_policy,
                repeats=args.repeats,
                warmup=args.warmup,
                expected_rung="resident_bf16",
                expected_query_chunk=256,
                hot_reuse=True,
            )
            cases.append(case)
            parity.extend(checks)

    speedups = [case["speedup"] for case in cases]
    result = {
        "status": "complete",
        "benchmark": "nori-inference-pipeline-batching",
        "model": "nori-6m",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "seed": SEED,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "workload": {
            "n_train": N_TRAIN,
            "plain_n_test": N_PLAIN_TEST,
            "cached_n_test": N_CACHED_TEST if args.include_resident_cache else None,
            "raw_features": N_FEATURES,
            "elements_budget": ELEMENTS_BUDGET,
            "gpu_cache_cap_gb": args.gpu_cache_cap_gb,
            "host_cache_cap_gb": args.host_cache_cap_gb,
            "cached_measurement": "hot_context_reuse" if args.include_resident_cache else None,
        },
        "geomean_speedup": math.exp(sum(math.log(speedup) for speedup in speedups) / len(speedups)),
        "cases": cases,
        "parity": parity,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
