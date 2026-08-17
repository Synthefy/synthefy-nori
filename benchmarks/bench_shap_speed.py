"""Benchmark SHAP explanation speed for Nori predictions.

What this measures
------------------
Nori is a model-agnostic, sklearn-style regressor, so the classic ``shap``
library explains it through a prediction-function explainer that wraps
``model.predict``. Every coalition SHAP evaluates is therefore one Nori forward
pass over the (fixed context + query) rows, and the wall-clock cost is dominated
by the number of model evaluations the explainer requests.

This script benchmarks two things on a small feature subset of the
``superconductivity`` dataset:

1. **SHAP wall-clock vs. evaluation budget.** ``shap.KernelExplainer`` and
   ``shap.PermutationExplainer`` per-row explanation time as ``nsamples`` /
   ``max_evals`` is swept over {32, 64, 128, 256, 512}. We report mean
   seconds-per-explanation.

2. **SHAP vs SHAPIQ at a matched budget (bonus).** On the same model and
   features, the ``shap`` KernelExplainer is compared against shapiq's
   imputation-based explainer (``get_nori_imputation_explainer`` with
   ``index="SV"`` for plain Shapley values) at equal coalition budgets.

Results are printed as a table and a labeled plot is saved to
``benchmarks/plots/shap_speed.png``.

Data
----
Expects ``<name>_train.csv`` / ``<name>_test.csv`` (header row, last column is the
target) for the ``superconductivity`` OpenML dataset. Point the script at your
local copy with the ``NORI_BENCH_DATA_DIR`` environment variable; it defaults to
``cache/eval_datasets/superconductivity`` under the repo root.

How to run
----------
    uv run python benchmarks/bench_shap_speed.py
    NORI_BENCH_DATA_DIR=/path/to/superconductivity uv run python benchmarks/bench_shap_speed.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap

from synthefy_nori import NoriRegressor
from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

# --- Benchmark configuration ------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.environ.get(
        "NORI_BENCH_DATA_DIR", REPO_ROOT / "cache" / "eval_datasets" / "superconductivity"
    )
)
TRAIN_CSV = DATA_DIR / "superconductivity_train.csv"
TEST_CSV = DATA_DIR / "superconductivity_test.csv"
PLOT_PATH = REPO_ROOT / "benchmarks" / "plots" / "shap_speed.png"

DEVICE = "cuda:0"
N_CONTEXT = 1500       # subsampled training context rows
N_BACKGROUND = 64      # background/reference rows handed to the explainers
N_FEATURES = 12        # small feature subset to keep Shapley tractable
N_EXPLAIN = 5          # test rows explained per setting
BUDGETS = [32, 64, 128, 256, 512]
SEED = 0


def load_subset():
    """Load a feature subset of superconductivity as float64 numpy arrays."""
    train = np.loadtxt(TRAIN_CSV, delimiter=",", skiprows=1, dtype=np.float64)
    test = np.loadtxt(TEST_CSV, delimiter=",", skiprows=1, dtype=np.float64)
    rng = np.random.default_rng(SEED)

    # Subsample context rows and pick the first N_FEATURES columns.
    ctx_idx = rng.choice(train.shape[0], size=min(N_CONTEXT, train.shape[0]), replace=False)
    feat_idx = np.arange(N_FEATURES)

    X_train = train[ctx_idx][:, feat_idx]
    y_train = train[ctx_idx][:, -1]

    test_idx = rng.choice(test.shape[0], size=N_EXPLAIN + N_BACKGROUND, replace=False)
    X_test_all = test[test_idx][:, feat_idx]
    X_background = X_test_all[:N_BACKGROUND]
    X_explain = X_test_all[N_BACKGROUND:N_BACKGROUND + N_EXPLAIN]
    return X_train, y_train, X_background, X_explain


def time_shap_kernel(predict_fn, X_background, X_explain, nsamples):
    """Mean seconds per row for shap.KernelExplainer at a given nsamples."""
    explainer = shap.KernelExplainer(predict_fn, X_background)
    t0 = time.perf_counter()
    explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    return (time.perf_counter() - t0) / len(X_explain)


def time_shap_permutation(predict_fn, X_background, X_explain, max_evals):
    """Mean seconds per row for shap.PermutationExplainer at a given max_evals."""
    explainer = shap.PermutationExplainer(predict_fn, X_background)
    t0 = time.perf_counter()
    explainer(X_explain, max_evals=max_evals, silent=True)
    return (time.perf_counter() - t0) / len(X_explain)


def time_shapiq(model, X_background, X_explain, budget):
    """Mean seconds per row for shapiq's imputation explainer (index='SV')."""
    explainer = get_nori_imputation_explainer(
        model, X_background, index="SV", max_order=1, imputer="baseline"
    )
    t0 = time.perf_counter()
    for row in X_explain:
        explainer.explain(row.reshape(1, -1), budget=budget)
    return (time.perf_counter() - t0) / len(X_explain)


def main():
    print(f"Loading superconductivity subset ({N_FEATURES} features)...")
    X_train, y_train, X_background, X_explain = load_subset()
    print(
        f"  context={X_train.shape}  background={X_background.shape}  "
        f"explain={X_explain.shape}"
    )

    print(f"Fitting NoriRegressor on {DEVICE} (first fit downloads checkpoint)...")
    model = NoriRegressor(device=DEVICE, model="nori-6m").fit(X_train, y_train)

    def predict_fn(x):
        return np.asarray(model.predict(np.asarray(x)), dtype=np.float64).reshape(-1)

    # Warm-up: one throwaway predict so timings exclude one-time setup.
    print("Warming up...")
    _ = predict_fn(X_explain)

    kernel_times = []
    perm_times = []
    shapiq_times = []
    for budget in BUDGETS:
        kt = time_shap_kernel(predict_fn, X_background, X_explain, budget)
        pt = time_shap_permutation(predict_fn, X_background, X_explain, budget)
        st = time_shapiq(model, X_background, X_explain, budget)
        kernel_times.append(kt)
        perm_times.append(pt)
        shapiq_times.append(st)
        print(
            f"budget={budget:4d} | shap.Kernel={kt:7.3f}s/row | "
            f"shap.Permutation={pt:7.3f}s/row | shapiq(SV)={st:7.3f}s/row"
        )

    # --- Results table ------------------------------------------------------
    print("\n=== Mean seconds per explanation (per test row) ===")
    print(f"{'budget':>8} {'shap.Kernel':>14} {'shap.Permutation':>18} {'shapiq(SV)':>14}")
    for i, budget in enumerate(BUDGETS):
        print(
            f"{budget:>8} {kernel_times[i]:>14.3f} {perm_times[i]:>18.3f} "
            f"{shapiq_times[i]:>14.3f}"
        )

    # --- Plot ---------------------------------------------------------------
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(BUDGETS, kernel_times, "o-", label="shap.KernelExplainer")
    ax.plot(BUDGETS, perm_times, "s-", label="shap.PermutationExplainer")
    ax.plot(BUDGETS, shapiq_times, "^-", label="shapiq imputation (SV)")
    ax.set_xlabel("Evaluation budget (nsamples / max_evals)")
    ax.set_ylabel("Mean seconds per explanation (per row)")
    ax.set_title(
        f"SHAP explanation speed on Nori\n"
        f"superconductivity, {N_FEATURES} features, context={N_CONTEXT}, "
        f"{N_EXPLAIN} rows, {DEVICE}"
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
