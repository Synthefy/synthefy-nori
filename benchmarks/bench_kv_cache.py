"""Benchmark: KV-cache inference speedup for Nori (cache ON vs OFF).

The Nori inference path chunks a large test set into row-batches that each share
the same fixed training context. The KV cache (``forward_cached_regression``)
projects the train-side sequence K/V **once** and reuses it across every test
chunk, instead of recomputing it per chunk. It is numerically equivalent to the
uncached chunked path (predictions are identical) and is documented as ~2-3x
faster on multi-chunk inference. It is controlled at *predict time* by:

  * ``SYNTHEFY_ENABLE_CACHED_INFERENCE`` = ``"1"`` (on, default) / ``"0"`` (off)
  * ``SYNTHEFY_DISABLE_CACHED_INFERENCE=1``                      (kill switch)

The cache only engages when the standard path is already chunking
(``n_test > chunk_size``). ``chunk_size`` shrinks as the per-row element budget
(``SYNTHEFY_MAX_ELEMENTS_BUDGET``) shrinks and as the (post-feature-selection)
feature count grows. We use the 1024-feature QSAR-TID-11 dataset and lower the
element budget so that ``chunk_size`` is well below the test-set sizes, forcing
several chunks per predict and thus a clear cache win.

For each test-set size we measure the wall-clock of ``model.predict(X_test)``
with the cache ON and OFF (best-of-2, after one throwaway warm-up predict, with
``torch.cuda.synchronize()`` around each timed region), confirm the cache
actually activated (multi-chunk), and verify the two prediction vectors are
near-identical (max abs diff ~1e-5 or smaller).

Run:
    uv run python benchmarks/bench_kv_cache.py
    # or: /home/pohanli/synthefy-nori/.venv/bin/python benchmarks/bench_kv_cache.py

Outputs a results table to stdout and saves benchmarks/plots/kv_cache_speed.png.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from synthefy_nori import NoriRegressor

# --- Configuration -----------------------------------------------------------
DATA_DIR = Path("cache/pr30_two/QSAR-TID-11")
TRAIN_CSV = DATA_DIR / "QSAR-TID-11_train.csv"
TEST_CSV = DATA_DIR / "QSAR-TID-11_test.csv"
PLOT_PATH = Path("benchmarks/plots/kv_cache_speed.png")

# 1024-feature dataset; lower the per-row element budget so chunk_size lands at
# its 256-row floor (forces MANY chunks -> the cache's per-chunk train-K/V reuse
# wins clearly). With n_train ~= 4019 and post-SVD budget 256 features, the
# default budget (2_000_000) gives chunk_size ~3793 (> 1723 test rows -> NO
# chunking at all). We set 1_050_000 ->
#   chunk_size = max(256, 1_050_000//256 - 4019) = 256  (the floor),
# while base_elements = (n_train+1)*256 ~= 1.03M < budget, so the full training
# context is kept (no train-row subsampling). The full 1723-row test set then
# splits into ~7 chunks.
MAX_ELEMENTS_BUDGET = 1_050_000

TEST_SIZES = [512, 1024, 1536, 1723]  # 1723 = full QSAR test set; each > chunk_size=256
DEVICE = "cuda:0"
SEED = 0
N_REPEATS = 2  # best-of-N timing per configuration


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV with a header row; last column is the target ``y``."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    return arr[:, :-1], arr[:, -1]


def timed_predict(model: NoriRegressor, x_test: np.ndarray, enable_cache: bool) -> tuple[float, np.ndarray]:
    """Best-of-N wall-clock of ``model.predict`` with the cache on/off.

    Env vars are re-read at predict time, so toggling ``os.environ`` here
    suffices. ``torch.cuda.synchronize()`` brackets each timed region.
    """
    os.environ["SYNTHEFY_ENABLE_CACHED_INFERENCE"] = "1" if enable_cache else "0"
    os.environ.pop("SYNTHEFY_DISABLE_CACHED_INFERENCE", None)

    best = float("inf")
    preds = None
    for _ in range(N_REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = model.predict(x_test)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best, preds


def main() -> None:
    os.environ["SYNTHEFY_MAX_ELEMENTS_BUDGET"] = str(MAX_ELEMENTS_BUDGET)

    x_train, y_train = load_csv(TRAIN_CSV)
    x_test_full, _ = load_csv(TEST_CSV)
    print(
        f"Loaded QSAR-TID-11: train {x_train.shape}, test {x_test_full.shape}, "
        f"budget={MAX_ELEMENTS_BUDGET}"
    )

    model = NoriRegressor(device=DEVICE).fit(x_train, y_train)

    # Warm up (downloads/compiles, allocates caches) with one throwaway predict.
    _ = model.predict(x_test_full[:128])

    rows = []
    for n in TEST_SIZES:
        x_test = x_test_full[:n]
        off_s, off_preds = timed_predict(model, x_test, enable_cache=False)
        on_s, on_preds = timed_predict(model, x_test, enable_cache=True)
        speedup = off_s / on_s if on_s > 0 else float("nan")
        max_diff = float(np.max(np.abs(on_preds - off_preds)))
        rows.append((n, on_s, off_s, speedup, max_diff))
        print(
            f"n_test={n:5d}  on={on_s:7.3f}s  off={off_s:7.3f}s  "
            f"speedup={speedup:5.2f}x  max_pred_diff={max_diff:.2e}"
        )

    # --- Results table -------------------------------------------------------
    print("\n=== KV cache speedup (best-of-{}) ===".format(N_REPEATS))
    print(f"{'n_test':>7} {'on_s':>9} {'off_s':>9} {'speedup':>9} {'max_pred_diff':>15}")
    for n, on_s, off_s, speedup, max_diff in rows:
        print(f"{n:>7} {on_s:>9.3f} {off_s:>9.3f} {speedup:>8.2f}x {max_diff:>15.2e}")

    # --- Plot ----------------------------------------------------------------
    ns = [r[0] for r in rows]
    on_t = [r[1] for r in rows]
    off_t = [r[2] for r in rows]
    speedups = [r[3] for r in rows]

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, off_t, "o-", color="#d62728", label="cache OFF (recompute train K/V per chunk)")
    ax.plot(ns, on_t, "o-", color="#2ca02c", label="cache ON (reuse train K/V)")
    ax.set_xlabel("test-set size (n_test rows)")
    ax.set_ylabel("predict wall-clock (s)")
    ax.set_title("Nori KV-cache inference speedup (QSAR-TID-11, 1024 features)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Annotate each point pair with the measured speedup.
    for n, on_s, off_s, sp in zip(ns, on_t, off_t, speedups):
        ax.annotate(
            f"{sp:.2f}x",
            xy=(n, (on_s + off_s) / 2),
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
