"""Interpret Nori on real data with NoriInterpreter — UCI *Default of Credit Card Clients*.

Downloads the dataset fresh from the UCI ML Repository (via ``ucimlrepo`` — no local file
path), then a single ``NoriInterpreter().fit(X, y)`` runs the whole pipeline in sequence
(feature importance -> prune to the features that carry >=95% of Nori's skill -> distill a
glass-box EBM) and stores every artifact on the fitted estimator. Writes a JSON summary +
importances + EBM structure, the fitted EBM (joblib), and two figures.

Install:
  pip install "synthefy-nori[explainability]" ucimlrepo

Run:
  python examples/explainability_credit.py                 # full run + figures
  python examples/explainability_credit.py --nori-model nori-30m --out-dir /tmp/credit
"""
import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")          # headless: pick the non-interactive backend BEFORE pyplot,
                               # and before NoriInterpreter.fit renders its diagram

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from synthefy_nori.explainability import NoriInterpreter
from ucimlrepo import fetch_ucirepo


def load_credit():
    """Download the UCI Default of Credit Card Clients dataset FRESH (no local path).
    Returns a DataFrame X (descriptive column names) and a 0/1 target array."""
    ds = fetch_ucirepo(id=350)
    X = ds.data.features.copy()
    rename = {r["name"]: r["description"] for _, r in ds.variables.iterrows()
              if r["name"] in X.columns and isinstance(r["description"], str)}  # X1..X23 -> real names
    X = X.rename(columns=rename)
    y = ds.data.targets.iloc[:, 0].to_numpy().astype(int)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="credit_explainability_out")
    ap.add_argument("--nori-model", default="nori-6m")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    print("Downloading UCI Default of Credit Card Clients (fresh) ...", flush=True)
    X, y = load_credit()
    print(f"  rows={len(X)} features={X.shape[1]} default rate={y.mean():.3f}", flush=True)

    # one call: internal 70/30 split -> permutation importance -> prune -> glass-box EBM
    # (also renders the glass-box model diagram onto interp.model_figure_)
    interp = NoriInterpreter(model=a.nori_model, target_name="default").fit(X, y)

    print("\n=== summary ===")
    print(json.dumps(interp.summary(), indent=1))
    print(f"\nkept {interp.n_selected_}/{len(interp.feature_names_)} features  |  "
          f"Nori AUC {interp.nori_full_score_:.3f} -> glass-box EBM AUC {interp.ebm_score_:.3f}")
    print("top features:", [(e["feature"], round(e["importance"], 4)) for e in interp.importance_ranking_[:7]])

    # persist artifacts pulled straight off the fitted estimator
    joblib.dump({"model": interp.ebm_, "feature_names": interp.selected_features_,
                 "feature_indices": interp.selected_indices_, "task": interp.task_},
                os.path.join(a.out_dir, "credit.ebm.joblib"))
    json.dump({"summary": interp.summary(), "importance": interp.importance_ranking_,
               "selected_features": interp.selected_features_, "ebm_model": interp.ebm_model_},
              open(os.path.join(a.out_dir, "credit.json"), "w"), indent=1)

    # figure 1: feature importance (from interp.importance_ranking_)
    top = interp.importance_ranking_[:10]
    ramp = LinearSegmentedColormap.from_list("s", ["#7C2D12", "#F97316", "#FDBA74"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(range(len(top)), [e["importance"] for e in top],
            color=[ramp(i / max(len(top) - 1, 1) * 0.9) for i in range(len(top))])
    ax.set_yticks(range(len(top))); ax.set_yticklabels([e["feature"] for e in top]); ax.invert_yaxis()
    ax.set_xlabel("Nori-permutation importance  (drop in test AUC when the feature is shuffled)")
    ax.set_title("Credit default — Nori feature importance (top 10)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "fig_credit_importance.png"), dpi=150, bbox_inches="tight")

    # figure 2: the glass-box model diagram, rendered during fit() and kept on the estimator
    interp.model_figure_.savefig(os.path.join(a.out_dir, "fig_credit_glassbox_model.png"),
                                 dpi=150, bbox_inches="tight", pad_inches=0.35)

    print(f"\nwrote {a.out_dir}/: credit.json, credit.ebm.joblib, "
          f"fig_credit_importance.png, fig_credit_glassbox_model.png")


if __name__ == "__main__":
    main()
