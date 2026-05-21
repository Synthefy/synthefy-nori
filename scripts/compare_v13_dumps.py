#!/usr/bin/env python3
"""Compare two V13 tier-1 reproduction dumps.

Usage:
    python scripts/compare_v13_dumps.py <reference_dir> <candidate_dir>
                                        [--atol A] [--rtol R]

Both dirs are produced by scripts/repro_v13_tier1_dump.sh.

DATA arrays (X_batch, y_batch, x_input, y_input, eval_pos) are compared
BIT-EXACT (NaN-aware — X_batch legitimately contains NaNs from structural
missingness). Synthetic data generation is fully deterministic given the
seed, so ANY data difference is a real data-generator / config / seed bug.

GRADIENT / WEIGHT / LOSS arrays are compared with a tolerance (default
atol=1e-3, rtol=1e-3). GPU floating-point + bf16 autocast are not bit-
deterministic: two runs of identical code differ by ~1e-5 in gradients.
A real model / loss / optimizer code bug is orders of magnitude larger.

The FIRST real divergence localizes the problem:
    DATA differs               -> data generator / config / seed
    data same, GRADIENTS differ -> model or loss code
    grads same, WEIGHTS differ  -> optimizer / scheduler / init

Exit 0 if the dumps match, 1 if they diverge.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

FILES = ["init_weights.npz"] + [f"step{i}.npz" for i in range(5)]
DATA_PREFIXES = ("X_batch", "y_batch", "x_input", "y_input", "eval_pos")


def category(key: str) -> tuple[int, str]:
    """(priority, label) — lower priority = earlier in the pipeline."""
    if key.startswith(DATA_PREFIXES):
        return 0, "DATA — data generator / config / seed mismatch"
    if key.startswith(("loss", "grad_norm")):
        return 1, "LOSS — loss computation mismatch"
    if key.startswith("grad."):
        return 2, "GRADIENTS — model or loss code mismatch"
    if key.startswith("weight."):
        return 3, "WEIGHTS — optimizer / scheduler / init mismatch"
    return 4, "unknown"


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max abs difference, NaN-aware. NaN on both sides = 0; NaN on one = inf."""
    if a.shape != b.shape:
        return float("inf")
    af, bf = a.astype(np.float64), b.astype(np.float64)
    both_nan = np.isnan(af) & np.isnan(bf)
    d = np.where(both_nan, 0.0, np.abs(af - bf))
    d = np.where(np.isnan(d), np.inf, d)
    return float(np.max(d)) if d.size else 0.0


def is_real_diff(key: str, a: np.ndarray, b: np.ndarray,
                 atol: float, rtol: float) -> bool:
    """Data: bit-exact. Everything else: within (atol, rtol)."""
    if a.shape != b.shape:
        return True
    prio, _ = category(key)
    if prio == 0:  # DATA — bit-exact
        return not np.array_equal(a, b, equal_nan=np.issubdtype(a.dtype, np.floating))
    return not np.allclose(a.astype(np.float64), b.astype(np.float64),
                           atol=atol, rtol=rtol, equal_nan=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two V13 repro dumps.")
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    print(f"reference : {args.reference}")
    print(f"candidate : {args.candidate}")
    print(f"tolerance : atol={args.atol:g} rtol={args.rtol:g} "
          f"(data arrays compared bit-exact)")
    print("-" * 68)

    real_diffs: list[tuple[int, int, str, str, float]] = []
    noise_floor = 0.0  # largest within-tolerance diff (the GPU-FP noise)
    overall_ok = True

    for fidx, fn in enumerate(FILES):
        pa, pb = os.path.join(args.reference, fn), os.path.join(args.candidate, fn)
        if not os.path.exists(pa) or not os.path.exists(pb):
            miss = "reference" if not os.path.exists(pa) else "candidate"
            print(f"{fn:20s}  MISSING in {miss}")
            overall_ok = False
            continue
        za, zb = np.load(pa), np.load(pb)
        ka, kb = set(za.files), set(zb.files)
        if ka != kb:
            print(f"{fn:20s}  KEY SET DIFFERS  "
                  f"only-ref={sorted(ka - kb)[:3]} only-cand={sorted(kb - ka)[:3]}")
            overall_ok = False

        n_real, worst_mad, worst_key = 0, 0.0, None
        for k in sorted(ka & kb):
            a, b = za[k], zb[k]
            mad = max_abs_diff(a, b)
            if is_real_diff(k, a, b, args.atol, args.rtol):
                n_real += 1
                prio, _ = category(k)
                real_diffs.append((fidx, prio, fn, k, mad))
                if mad >= worst_mad:
                    worst_mad, worst_key = mad, k
            else:
                noise_floor = max(noise_floor, mad)

        if n_real == 0 and ka == kb:
            print(f"{fn:20s}  OK  ({len(ka)} arrays match)")
        elif n_real:
            overall_ok = False
            print(f"{fn:20s}  {n_real} array(s) DIVERGE  "
                  f"worst: {worst_key} max|d|={worst_mad:.3e}")

    print("-" * 68)
    print(f"GPU-FP noise floor (largest within-tolerance diff): {noise_floor:.3e}")
    if overall_ok:
        print("VERDICT: dumps MATCH — data bit-identical, compute within tolerance.")
        print("         Reproduction is correct.")
        sys.exit(0)

    real_diffs.sort(key=lambda t: (t[0], t[1]))
    if real_diffs:
        fidx, prio, fn, key, mad = real_diffs[0]
        _, label = category(key)
        print(f"VERDICT: first real divergence at {fn} :: {key}")
        print(f"         max|diff| = {mad:.3e}")
        print(f"         -> {label}")
    else:
        print("VERDICT: dumps differ (missing files or key sets) — see above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
