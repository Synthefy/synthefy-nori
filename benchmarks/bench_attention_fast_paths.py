#!/usr/bin/env python3
"""Paired CUDA benchmark for Nori's exact attention fast paths.

The baseline functions reproduce the implementations used before the fast
paths landed: generic einsum for cached projections and a conservative 32768
batch/head cutoff for SDPA. Candidate and baseline calls alternate in one
process so clock and host-load drift affect both arms similarly.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

import synthefy_nori.model.layer as layer_module
from synthefy_nori.model.layer import MultiheadAttention


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def timed_call(call: Callable[[], torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = call()
    end.record()
    end.synchronize()
    del output
    return float(start.elapsed_time(end))


def paired_measure(
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    for index in range(warmup):
        order = (baseline, candidate) if index % 2 == 0 else (candidate, baseline)
        for call in order:
            timed_call(call)

    durations = {"baseline": [], "candidate": []}
    for index in range(iterations):
        order = (
            (("baseline", baseline), ("candidate", candidate))
            if index % 2 == 0
            else (("candidate", candidate), ("baseline", baseline))
        )
        for name, call in order:
            durations[name].append(timed_call(call))
    return statistics.median(durations["baseline"]), statistics.median(durations["candidate"])


def projection_cases(
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[dict], list[dict]]:
    torch.manual_seed(10)
    attention = MultiheadAttention(
        embed_dim=96,
        num_heads=6,
        qkv_combined=False,
        device=device,
        dtype=torch.float32,
    ).eval()
    cases = []
    parity = []

    x_kv = torch.randn(1, 12, 8192, 96, device=device)
    x_kv_flat = x_kv.reshape(-1, *x_kv.shape[-2:])
    kv_mha_weights = attention.qkv_proj_weight[1:]

    def baseline_kv_mha() -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return torch.einsum(
                "... s, j h d s -> ... j h d",
                x_kv_flat,
                kv_mha_weights,
            ).contiguous()

    def candidate_kv_mha() -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return (
                torch.nn.functional.linear(
                    x_kv_flat,
                    kv_mha_weights.reshape(
                        2 * attention.num_heads * attention.head_dim,
                        attention.embed_dim,
                    ),
                )
                .view(
                    *x_kv_flat.shape[:-1],
                    2,
                    attention.num_heads,
                    attention.head_dim,
                )
                .contiguous()
            )

    with torch.no_grad():
        baseline_output = baseline_kv_mha()
        candidate_output = candidate_kv_mha()
    parity.append(
        {
            "name": "cached-kv-mha-output",
            "passed": torch.equal(baseline_output, candidate_output),
            "max_abs": float((baseline_output - candidate_output).abs().max().item()),
        }
    )
    baseline_ms, candidate_ms = paired_measure(
        baseline_kv_mha,
        candidate_kv_mha,
        warmup=warmup,
        iterations=iterations,
    )
    cases.append(
        {
            "name": "cached-kv-mha-g12-n8192",
            "baseline_ms": baseline_ms,
            "candidate_ms": candidate_ms,
        }
    )
    del baseline_output, candidate_output

    x_q = torch.randn(1, 192, 384, 96, device=device)
    x_q_flat = x_q.reshape(-1, *x_q.shape[-2:])
    q_weights = attention.qkv_proj_weight[0]

    def baseline_q() -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return torch.einsum("... s, h d s -> ... h d", x_q_flat, q_weights)

    def candidate_q() -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return torch.nn.functional.linear(
                x_q_flat,
                q_weights.reshape(attention.num_heads * attention.head_dim, attention.embed_dim),
            ).view(*x_q_flat.shape[:-1], attention.num_heads, attention.head_dim)

    with torch.no_grad():
        baseline_output = baseline_q()
        candidate_output = candidate_q()
    parity.append(
        {
            "name": "cached-q-output",
            "passed": torch.equal(baseline_output, candidate_output),
            "max_abs": float((baseline_output - candidate_output).abs().max().item()),
        }
    )
    baseline_ms, candidate_ms = paired_measure(
        baseline_q,
        candidate_q,
        warmup=warmup,
        iterations=iterations,
    )
    cases.append(
        {
            "name": "cached-q-g192-n384",
            "baseline_ms": baseline_ms,
            "candidate_ms": candidate_ms,
        }
    )
    return cases, parity


def attention_case(
    device: torch.device,
    *,
    batch: int,
    sequence: int,
    warmup: int,
    iterations: int,
) -> tuple[dict, list[dict]]:
    torch.manual_seed(batch + sequence)
    attention = MultiheadAttention(
        embed_dim=128,
        num_heads=2,
        qkv_combined=False,
        device=device,
        dtype=torch.bfloat16,
    ).eval()
    q = torch.randn(batch, sequence, 2, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(batch, sequence, 2, 2, 64, device=device, dtype=torch.bfloat16, requires_grad=True)

    def step(limit: int) -> torch.Tensor:
        q.grad = None
        kv.grad = None
        layer_module.SDPA_BATCH_HEAD_LIMIT = limit
        output = attention.compute_attention_by_torch(None, q, kv, None)
        output.backward(torch.ones_like(output))
        return output

    def baseline() -> torch.Tensor:
        return step(32_768)

    def candidate() -> torch.Tensor:
        return step(65_535)

    baseline_output = baseline().detach()
    baseline_q_grad = q.grad.detach().clone()
    baseline_kv_grad = kv.grad.detach().clone()
    candidate_output = candidate().detach()
    candidate_q_grad = q.grad.detach().clone()
    candidate_kv_grad = kv.grad.detach().clone()
    parity = []
    for name, expected, actual in (
        ("output", baseline_output, candidate_output),
        ("q-gradient", baseline_q_grad, candidate_q_grad),
        ("kv-gradient", baseline_kv_grad, candidate_kv_grad),
    ):
        parity.append(
            {
                "name": f"sdpa-b{batch}-s{sequence}-{name}",
                "passed": torch.equal(expected, actual),
                "max_abs": float((expected - actual).abs().max().item()),
            }
        )

    baseline_ms, candidate_ms = paired_measure(
        baseline,
        candidate,
        warmup=warmup,
        iterations=iterations,
    )
    layer_module.SDPA_BATCH_HEAD_LIMIT = 65_535
    return (
        {
            "name": f"sdpa-forward-backward-b{batch}-s{sequence}-h2-d64",
            "baseline_ms": baseline_ms,
            "candidate_ms": candidate_ms,
        },
        parity,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=nonnegative_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("this benchmark requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("--device must be cuda[:INDEX]")
    torch.cuda.set_device(device)

    cases, parity = projection_cases(device, warmup=args.warmup, iterations=args.iterations)
    for batch, sequence in ((24_000, 13), (28_000, 9)):
        case, checks = attention_case(
            device,
            batch=batch,
            sequence=sequence,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        cases.append(case)
        parity.extend(checks)

    speedups = [case["baseline_ms"] / case["candidate_ms"] for case in cases]
    result = {
        "status": "complete",
        "benchmark": "attention-fast-paths",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "geomean_speedup": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
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
