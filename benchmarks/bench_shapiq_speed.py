"""Benchmark: SHAPIQ explanation wall-clock vs. coalition budget, across models.

The cost of a SHAPIQ Shapley / Shapley-interaction explanation on an in-context
tabular model is dominated by the number of sampled coalitions (the ``budget``):
shapiq's baseline imputer stacks every coalition of an explanation into ONE
``(n_coalitions, n_features)`` array and issues a single ``model.predict``, so
each explanation costs ≈ one batched forward pass against the fixed training
context — essentially **flat in the budget**. This script measures the mean
wall-clock *per explanation* as ``budget`` grows, for two interaction orders:

  * ``max_order=1`` (index ``"SV"``)    -> plain Shapley values
  * ``max_order=2`` (index ``"k-SII"``) -> pairwise Shapley interactions

and repeats the sweep for each tabular foundation model
(``nori`` / ``tabicl`` / ``tabpfn``) on the **same** model-agnostic explanation
path (``synthefy_nori.interpretability.shapiq.get_nori_imputation_explainer``,
which accepts any fitted estimator exposing ``.predict``), so the comparison is
apples-to-apples.

To keep coalition counts tractable on the ~81-feature ``superconductivity`` set
we (a) subsample the training context to ``N_CONTEXT`` background rows for the
imputer, (b) restrict the explanation to the top-variance ``N_FEATURES``
columns, and (c) explain a small fixed set of test rows, reporting the mean
per-explanation time. Each model is warmed up with one throwaway ``predict``.

Data:
    Expects ``<name>_train.csv`` / ``<name>_test.csv`` (header row, last column
    is the target) for the ``superconductivity`` OpenML dataset. Generate them
    with ``uv run python benchmarks/prep_data.py`` (writes to
    ``cache/eval_datasets/superconductivity``), or point ``NORI_BENCH_DATA_DIR``
    at your own copy.

Run:
    uv run python benchmarks/bench_shapiq_speed.py
    BENCH_MODELS=nori,tabicl uv run python benchmarks/bench_shapiq_speed.py

Outputs a per-model results table to stdout and saves
benchmarks/plots/shapiq_speed.png.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from synthefy_nori.interpretability.shapiq import get_nori_imputation_explainer

from _bench_models import MODEL_LABELS, MODELS, build_model

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
PLOT_PATH = REPO_ROOT / "benchmarks" / "plots" / "shapiq_speed.png"

N_CONTEXT = 1500  # training context (also imputer background) rows
N_FEATURES = 12  # top-variance feature subset for tractable coalitions
N_EXPLAIN = 5  # test rows explained (mean per-explanation time reported)
BUDGETS = [32, 64, 128, 256, 512]
DEVICE = os.environ.get("BENCH_DEVICE", "cuda:0")
SEED = 0

# Which models to run (comma-separated); defaults to all three.
MODELS_TO_RUN = [m.strip() for m in os.environ.get("BENCH_MODELS", ",".join(MODELS)).split(",") if m.strip()]

# Two explanation settings: plain Shapley values and pairwise interactions.
SETTINGS = [
    ("order 1 (SV)", {"index": "SV", "max_order": 1}),
    ("order 2 (k-SII)", {"index": "k-SII", "max_order": 2}),
]
# linestyle per setting, color per model -> a readable 6-line plot.
SETTING_STYLE = {"order 1 (SV)": "-", "order 2 (k-SII)": "--"}
MODEL_COLOR = {"nori": "#1f77b4", "tabicl": "#ff7f0e", "tabpfn": "#2ca02c"}


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV with a header row; last column is the target ``y``."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    return arr[:, :-1], arr[:, -1]


def main() -> None:
    rng = np.random.default_rng(SEED)

    X_train_full, y_train_full = load_csv(TRAIN_CSV)
    X_test_full, _ = load_csv(TEST_CSV)
    print(
        f"Loaded superconductivity: train {X_train_full.shape}, test "
        f"{X_test_full.shape} ({X_train_full.shape[1]} features)"
    )

    # Subsample the training context (and imputer background).
    ctx_idx = rng.choice(X_train_full.shape[0], size=min(N_CONTEXT, X_train_full.shape[0]), replace=False)
    X_ctx = X_train_full[ctx_idx]
    y_ctx = y_train_full[ctx_idx]

    # Select the top-variance features so coalition counts stay sane.
    variances = X_ctx.var(axis=0)
    feat_idx = np.sort(np.argsort(variances)[::-1][:N_FEATURES])
    print(f"Using {N_FEATURES} top-variance feature columns: {feat_idx.tolist()}")

    X_ctx_sel = X_ctx[:, feat_idx]
    X_test_sel = X_test_full[:, feat_idx]
    explain_rows = X_test_sel[:N_EXPLAIN]

    # results[model][setting] -> list of mean s/explanation per budget
    results: dict[str, dict[str, list[float]]] = {}
    floors: dict[str, float] = {}

    for model_name in MODELS_TO_RUN:
        print(f"\n=== {MODEL_LABELS.get(model_name, model_name)} ===")
        try:
            model = build_model(model_name, X_ctx_sel, y_ctx, device=DEVICE)
        except ImportError as exc:
            print(f"  [skip] {model_name}: {exc} (install with .[internal-eval])")
            continue

        # Warm up on BOTH the 1-row and N-row shapes before any timing — Nori
        # JIT-compiles its inference path per shape on first use, so timing the
        # floor on a cold shape would capture a one-time compile, not the floor.
        t0 = time.perf_counter()
        _ = model.predict(explain_rows[:1])
        _ = model.predict(explain_rows)
        print(f"  warmup predict (compile): {time.perf_counter() - t0:.2f}s")
        t0 = time.perf_counter()
        _ = model.predict(explain_rows)
        floors[model_name] = time.perf_counter() - t0
        print(f"  single batched predict ({N_EXPLAIN} rows): {floors[model_name]:.3f}s")

        results[model_name] = {}
        for label, cfg in SETTINGS:
            explainer = get_nori_imputation_explainer(model, X_ctx_sel, imputer="baseline", random_state=SEED, **cfg)
            per_budget: list[float] = []
            for budget in BUDGETS:
                t0 = time.perf_counter()
                for i in range(N_EXPLAIN):
                    explainer.explain(explain_rows[i : i + 1], budget=budget)
                elapsed = (time.perf_counter() - t0) / N_EXPLAIN
                per_budget.append(elapsed)
                print(f"  [{label}] budget={budget:4d}  {elapsed:7.3f} s/explanation")
            results[model_name][label] = per_budget

    # --- Results table -------------------------------------------------------
    print("\n=== SHAPIQ explanation speed (seconds per explanation) ===")
    cols = [(m, lbl) for m in results for lbl, _ in SETTINGS]
    header = f"{'budget':>8} | " + " | ".join(f"{MODEL_LABELS.get(m, m) + ' ' + lbl.split()[1]:>18}" for m, lbl in cols)
    print(header)
    print("-" * len(header))
    for j, budget in enumerate(BUDGETS):
        row = f"{budget:>8} | " + " | ".join(f"{results[m][lbl][j]:>18.3f}" for m, lbl in cols)
        print(row)
    print(
        "\nSingle-batched-predict floor (s): " + ", ".join(f"{MODEL_LABELS.get(m, m)}={floors[m]:.3f}" for m in results)
    )

    # --- Plot ----------------------------------------------------------------
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in results:
        for label, _ in SETTINGS:
            ax.plot(
                BUDGETS,
                results[model_name][label],
                marker="o",
                color=MODEL_COLOR.get(model_name),
                linestyle=SETTING_STYLE[label],
                label=f"{MODEL_LABELS.get(model_name, model_name)} · {label}",
            )
    ax.set_xlabel("budget (number of sampled coalitions)")
    ax.set_ylabel("seconds per explanation")
    ax.set_ylim(bottom=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_title(
        "SHAPIQ explanation speed: Nori vs TabICL vs TabPFN\n"
        f"superconductivity, {N_FEATURES} features, {N_CONTEXT} context rows, "
        f"mean of {N_EXPLAIN} rows ({DEVICE})"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
