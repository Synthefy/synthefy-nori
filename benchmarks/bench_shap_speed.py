"""Benchmark SHAP explanation speed across Nori, TabICL, and TabPFN.

In-context tabular models are model-agnostic, sklearn-style regressors, so the
classic ``shap`` library explains them through a prediction-function explainer
that wraps ``model.predict``. Every coalition ``shap`` evaluates is one forward
pass over the (fixed context + query) rows, so the wall-clock cost is dominated
by how many model evaluations the explainer requests.

For each model this script measures, on a 12-feature subset of
``superconductivity`` (context = ``N_CONTEXT`` rows, ``N_EXPLAIN`` test rows):

1. **SHAP wall-clock vs. evaluation budget.** ``shap.KernelExplainer``
   (``nsamples``) and ``shap.PermutationExplainer`` (``max_evals``) per-row
   explanation time, swept over {32, 64, 128, 256, 512}. Mean seconds-per-row.

2. **shapiq imputation reference.** The same single-feature Shapley values via
   shapiq's baseline imputer (``index="SV"``), which batches all coalitions into
   one predict — the per-row floor — for comparison at matched budgets.

Results are printed as a per-model table and a labeled plot is saved to
``benchmarks/plots/shap_speed.png``.

Data:
    Expects ``<name>_train.csv`` / ``<name>_test.csv`` (header row, last column
    is the target) for ``superconductivity``. Generate with
    ``uv run python benchmarks/prep_data.py`` or set ``NORI_BENCH_DATA_DIR``.

Run:
    uv run python benchmarks/bench_shap_speed.py
    BENCH_MODELS=nori,tabpfn uv run python benchmarks/bench_shap_speed.py
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

from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

from _bench_models import MODEL_LABELS, MODELS, build_model, predict_fn

# --- Configuration -----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.environ.get(
        "NORI_BENCH_DATA_DIR",
        REPO_ROOT / "cache" / "eval_datasets" / "superconductivity",
    )
)
TRAIN_CSV = DATA_DIR / "superconductivity_train.csv"
TEST_CSV = DATA_DIR / "superconductivity_test.csv"
PLOT_PATH = REPO_ROOT / "benchmarks" / "plots" / "shap_speed.png"

DEVICE = os.environ.get("BENCH_DEVICE", "cuda:0")
N_CONTEXT = 1500  # subsampled training context rows
N_BACKGROUND = 64  # background/reference rows handed to the explainers
N_FEATURES = 12  # small feature subset to keep Shapley tractable
N_EXPLAIN = 5  # test rows explained per setting
BUDGETS = [32, 64, 128, 256, 512]
SEED = 0

MODELS_TO_RUN = [m.strip() for m in os.environ.get("BENCH_MODELS", ",".join(MODELS)).split(",") if m.strip()]
MODEL_COLOR = {"nori": "#1f77b4", "tabicl": "#ff7f0e", "tabpfn": "#2ca02c"}


def load_subset():
    """Load a feature subset of superconductivity as float64 numpy arrays."""
    train = np.loadtxt(TRAIN_CSV, delimiter=",", skiprows=1, dtype=np.float64)
    test = np.loadtxt(TEST_CSV, delimiter=",", skiprows=1, dtype=np.float64)
    rng = np.random.default_rng(SEED)

    ctx_idx = rng.choice(train.shape[0], size=min(N_CONTEXT, train.shape[0]), replace=False)
    feat_idx = np.arange(N_FEATURES)

    X_train = train[ctx_idx][:, feat_idx]
    y_train = train[ctx_idx][:, -1]

    test_idx = rng.choice(test.shape[0], size=N_EXPLAIN + N_BACKGROUND, replace=False)
    X_test_all = test[test_idx][:, feat_idx]
    X_background = X_test_all[:N_BACKGROUND]
    X_explain = X_test_all[N_BACKGROUND : N_BACKGROUND + N_EXPLAIN]
    return X_train, y_train, X_background, X_explain


def time_shap_kernel(pred, X_background, X_explain, nsamples):
    """Mean seconds per row for shap.KernelExplainer at a given nsamples."""
    explainer = shap.KernelExplainer(pred, X_background)
    t0 = time.perf_counter()
    explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    return (time.perf_counter() - t0) / len(X_explain)


def time_shap_permutation(pred, X_background, X_explain, max_evals):
    """Mean seconds per row for shap.PermutationExplainer at a given max_evals."""
    explainer = shap.PermutationExplainer(pred, X_background)
    t0 = time.perf_counter()
    explainer(X_explain, max_evals=max_evals, silent=True)
    return (time.perf_counter() - t0) / len(X_explain)


def time_shapiq(model, X_background, X_explain, budget):
    """Mean seconds per row for shapiq's imputation explainer (index='SV')."""
    explainer = get_nori_imputation_explainer(
        model, X_background, index="SV", max_order=1, imputer="baseline", random_state=SEED
    )
    t0 = time.perf_counter()
    for row in X_explain:
        explainer.explain(row.reshape(1, -1), budget=budget)
    return (time.perf_counter() - t0) / len(X_explain)


def main():
    print(f"Loading superconductivity subset ({N_FEATURES} features)...")
    X_train, y_train, X_background, X_explain = load_subset()
    print(f"  context={X_train.shape}  background={X_background.shape}  explain={X_explain.shape}")

    # results[model] -> {"kernel": [...], "perm": [...], "shapiq": [...]}
    results: dict[str, dict[str, list[float]]] = {}

    for model_name in MODELS_TO_RUN:
        print(f"\n=== {MODEL_LABELS.get(model_name, model_name)} on {DEVICE} ===")
        try:
            model = build_model(model_name, X_train, y_train, device=DEVICE)
        except ImportError as exc:
            print(f"  [skip] {model_name}: {exc} (install with .[internal-eval])")
            continue
        pred = predict_fn(model)

        print("  warming up...")
        _ = pred(X_explain)

        kernel_times, perm_times, shapiq_times = [], [], []
        for budget in BUDGETS:
            kt = time_shap_kernel(pred, X_background, X_explain, budget)
            pt = time_shap_permutation(pred, X_background, X_explain, budget)
            st = time_shapiq(model, X_background, X_explain, budget)
            kernel_times.append(kt)
            perm_times.append(pt)
            shapiq_times.append(st)
            print(
                f"  budget={budget:4d} | shap.Kernel={kt:7.3f}s/row | "
                f"shap.Permutation={pt:7.3f}s/row | shapiq(SV)={st:7.3f}s/row"
            )
        results[model_name] = {"kernel": kernel_times, "perm": perm_times, "shapiq": shapiq_times}

    # --- Results table ------------------------------------------------------
    print("\n=== Mean seconds per explanation (per test row) ===")
    for model_name, r in results.items():
        print(f"\n[{MODEL_LABELS.get(model_name, model_name)}]")
        print(f"{'budget':>8} {'shap.Kernel':>14} {'shap.Permutation':>18} {'shapiq(SV)':>14}")
        for i, budget in enumerate(BUDGETS):
            print(f"{budget:>8} {r['kernel'][i]:>14.3f} {r['perm'][i]:>18.3f} {r['shapiq'][i]:>14.3f}")

    # --- Plot ---------------------------------------------------------------
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, r in results.items():
        color = MODEL_COLOR.get(model_name)
        label = MODEL_LABELS.get(model_name, model_name)
        ax.plot(BUDGETS, r["kernel"], "o-", color=color, label=f"{label} · KernelExplainer")
        ax.plot(BUDGETS, r["perm"], "s--", color=color, label=f"{label} · PermutationExplainer")
        ax.plot(BUDGETS, r["shapiq"], "^:", color=color, label=f"{label} · shapiq(SV)")
    ax.set_xlabel("Evaluation budget (nsamples / max_evals)")
    ax.set_ylabel("Mean seconds per explanation (per row)")
    ax.set_title(
        "SHAP explanation speed: Nori vs TabICL vs TabPFN\n"
        f"superconductivity, {N_FEATURES} features, context={N_CONTEXT}, "
        f"{N_EXPLAIN} rows, {DEVICE}"
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
