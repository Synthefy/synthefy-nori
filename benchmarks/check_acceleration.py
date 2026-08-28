#!/usr/bin/env python3
"""Check a held-out performance result against Nori's acceleration gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--min-geomean-speedup", type=float, default=1.05)
    parser.add_argument("--max-case-regression", type=float, default=0.01)
    args = parser.parse_args()

    result = json.loads(args.results.read_text())
    if result.get("status") != "complete":
        print(f"PENDING status={result.get('status', 'missing')}")
        return 1

    cases = result.get("cases", [])
    parity = result.get("parity", [])
    if not cases or not parity:
        print("FAIL missing held-out cases or parity checks")
        return 1

    speedups = []
    regressions = []
    for case in cases:
        baseline_ms = float(case["baseline_ms"])
        candidate_ms = float(case["candidate_ms"])
        if baseline_ms <= 0.0 or candidate_ms <= 0.0:
            print(f"FAIL invalid timing for {case.get('name', '<unnamed>')}")
            return 1
        speedups.append(baseline_ms / candidate_ms)
        regressions.append(candidate_ms / baseline_ms - 1.0)

    geomean_speedup = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    worst_regression = max(0.0, max(regressions))
    parity_ok = all(bool(check.get("passed")) for check in parity)
    passed = geomean_speedup >= args.min_geomean_speedup and worst_regression <= args.max_case_regression and parity_ok
    verdict = "PASS" if passed else "FAIL"
    print(
        f"{verdict} geomean_speedup={geomean_speedup:.4f} "
        f"worst_regression={worst_regression:.4%} parity={'ok' if parity_ok else 'failed'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
