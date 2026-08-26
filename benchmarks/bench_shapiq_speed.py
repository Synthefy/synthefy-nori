"""Benchmark: SHAPIQ explanation wall-clock for Nori vs. coalition budget.

The cost of a SHAPIQ Shapley / Shapley-interaction explanation on Nori is
dominated by the number of sampled coalitions (the ``budget``): each coalition
is one masked ``model.predict`` call against the fixed training context. This
script measures how the mean wall-clock time *per explanation* scales with
``budget`` for two interaction orders:

  * ``max_order=1`` (index ``"SV"``)    -> plain Shapley values
  * ``max_order=2`` (index ``"k-SII"``) -> pairwise Shapley interactions

To keep coalition counts tractable on a ~81-feature dataset we (a) subsample the
training context to a few hundred background rows for the imputer, (b) restrict
the explanation to a small subset of the highest-variance features, and (c)
explain a small fixed set of test rows, reporting the mean per-explanation time.
The model is warmed up with one throwaway ``predict`` before any timing.

Data:
    Expects ``<name>_train.csv`` / ``<name>_test.csv`` (header row, last column is
    the target) for the ``superconductivity`` OpenML dataset. Point the script at
    your local copy with the ``NORI_BENCH_DATA_DIR`` environment variable; it
    defaults to ``cache/eval_datasets/superconductivity`` under the repo root.

Run:
    uv run python benchmarks/bench_shapiq_speed.py
    NORI_BENCH_DATA_DIR=/path/to/superconductivity uv run python benchmarks/bench_shapiq_speed.py

Outputs a results table to stdout and saves benchmarks/plots/shapiq_speed.png.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from synthefy_nori import NoriRegressor
from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

# --- Configuration -----------------------------------------------------------
DATA_DIR = Path(os.environ.get("NORI_BENCH_DATA_DIR", "cache/eval_datasets/superconductivity"))
TRAIN_CSV = DATA_DIR / "superconductivity_train.csv"
TEST_CSV = DATA_DIR / "superconductivity_test.csv"
PLOT_PATH = Path("benchmarks/plots/shapiq_speed.png")

N_CONTEXT = 1500  # training context (also imputer background) rows
N_FEATURES = 12  # top-variance feature subset for tractable coalitions
N_EXPLAIN = 5  # test rows explained (mean per-explanation time reported)
BUDGETS = [32, 64, 128, 256, 512]
DEVICE = "cuda:0"
SEED = 0


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV with a header row; last column is the target ``y``."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    return arr[:, :-1], arr[:, -1]


def main() -> None:
    rng = np.random.default_rng(SEED)

    X_train_full, y_train_full = load_csv(TRAIN_CSV)
    X_test_full, _ = load_csv(TEST_CSV)
    print(f"Loaded train {X_train_full.shape}, test {X_test_full.shape} ({X_train_full.shape[1]} features)")

    # Subsample the training context (and imputer background).
    ctx_idx = rng.choice(X_train_full.shape[0], size=min(N_CONTEXT, X_train_full.shape[0]), replace=False)
    X_ctx = X_train_full[ctx_idx]
    y_ctx = y_train_full[ctx_idx]

    # Select the top-variance features so coalition counts stay sane.
    variances = X_ctx.var(axis=0)
    feat_idx = np.argsort(variances)[::-1][:N_FEATURES]
    feat_idx = np.sort(feat_idx)
    print(f"Using {N_FEATURES} top-variance feature columns: {feat_idx.tolist()}")

    X_ctx_sel = X_ctx[:, feat_idx]
    X_test_sel = X_test_full[:, feat_idx]
    explain_rows = X_test_sel[:N_EXPLAIN]

    # Fit Nori on the reduced context and warm up the model with one predict.
    print(f"Fitting NoriRegressor on device={DEVICE} ...")
    model = NoriRegressor(device=DEVICE, model="nori-6m").fit(X_ctx_sel, y_ctx)
    t0 = time.perf_counter()
    _ = model.predict(explain_rows[:1])
    print(f"Warmup predict done in {time.perf_counter() - t0:.2f}s")

    # Reference: the cost of a single batched predict. shapiq's baseline imputer
    # stacks all coalitions of an explanation into ONE (n_coalitions, n_features)
    # array and issues a single predict, so this is the per-explanation floor.
    t0 = time.perf_counter()
    _ = model.predict(explain_rows)
    single_predict_s = time.perf_counter() - t0
    print(f"Single batched predict ({N_EXPLAIN} rows): {single_predict_s:.3f}s")

    # Two explanation settings: plain Shapley values and pairwise interactions.
    settings = [
        ("order 1 (SV)", {"index": "SV", "max_order": 1}),
        ("order 2 (k-SII)", {"index": "k-SII", "max_order": 2}),
    ]

    results: dict[str, list[float]] = {}
    for label, cfg in settings:
        explainer = get_nori_imputation_explainer(model, X_ctx_sel, imputer="baseline", random_state=SEED, **cfg)
        per_budget: list[float] = []
        for budget in BUDGETS:
            t0 = time.perf_counter()
            for i in range(N_EXPLAIN):
                explainer.explain(explain_rows[i : i + 1], budget=budget)
            elapsed = (time.perf_counter() - t0) / N_EXPLAIN
            per_budget.append(elapsed)
            print(f"[{label}] budget={budget:4d}  {elapsed:7.3f} s/explanation")
        results[label] = per_budget

    # --- Results table -------------------------------------------------------
    print("\n=== SHAPIQ explanation speed (seconds per explanation) ===")
    header = f"{'budget':>8} | " + " | ".join(f"{lbl:>16}" for lbl, _ in settings)
    print(header)
    print("-" * len(header))
    for j, budget in enumerate(BUDGETS):
        row = f"{budget:>8} | " + " | ".join(f"{results[lbl][j]:>16.3f}" for lbl, _ in settings)
        print(row)

    # --- Plot ----------------------------------------------------------------
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, _ in settings:
        ax.plot(BUDGETS, results[label], marker="o", label=label)
    ax.axhline(
        single_predict_s,
        color="gray",
        linestyle="--",
        label="single batched predict (floor)",
    )
    ax.set_xlabel("budget (number of sampled coalitions)")
    ax.set_ylabel("seconds per explanation")
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"SHAPIQ explanation speed on Nori\n"
        f"superconductivity, {N_FEATURES} features, {N_CONTEXT} context rows, "
        f"mean of {N_EXPLAIN} rows ({DEVICE})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
