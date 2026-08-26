"""Nori embeddings on a real TabArena dataset: extract, probe, and visualize.

The same workflow as ``embedding_synthetic.py``, but on a real regression dataset
from the TabArena suite (downloaded once from OpenML, pinned by dataset ID). It
extracts Nori row embeddings, runs a downstream Ridge probe, compares against
Nori's own native head, and draws a t-SNE of raw features vs embeddings.

    uv sync --extra eval                                    # openml + matplotlib
    uv run python examples/embedding_tabarena.py            # default dataset
    uv run python examples/embedding_tabarena.py --dataset wine_quality
    uv run python examples/embedding_tabarena.py --list     # available names

Two ways to get embeddings — both return a 3D array
``(n_estimators, n_samples, embed_dim)``; average over axis 0 (the
preprocessing-pipeline ensemble) for a 2D feature matrix:
  1. ``NoriRegressor.get_embeddings(X, data_source=...)`` — the low-level call.
  2. ``NoriEmbedding`` — an sklearn transformer; with ``n_fold >= 2`` it produces
     leakage-free out-of-fold embeddings for the training rows.

The first run downloads the dataset CSVs and the public ~47MB checkpoint; GPU if
available, else CPU.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from synthefy_nori import NoriEmbedding, NoriRegressor
from synthefy_nori.evaluation.datasets import DatasetRegistry

OUT_DIR = "results"
CACHE_DIR = "cache/tabarena_reg"
DEFAULT_DATASET = "wine_quality"
MAX_TRAIN, MAX_TEST = 3000, 1500  # keep the demo fast (embeddings scale w/ context)


def load_registry():
    """Download (once) and load the TabArena regression suite."""
    reg = DatasetRegistry(cache_dir="cache/eval_datasets", max_train_samples=MAX_TRAIN)
    if not os.path.isdir(CACHE_DIR) or not os.listdir(CACHE_DIR):
        reg.download_tabarena(reg_dir=CACHE_DIR)  # needs `openml`
    reg.load_tabarena(reg_dir=CACHE_DIR)
    return reg


def extract_embeddings(X_train, y_train, X_test):
    """Return OOF train embeddings, test embeddings (both 2D), and Nori's own
    native test prediction — using a single fitted model for all three.

    Demonstrates both embedding APIs: the ``NoriEmbedding`` transformer (for the
    leakage-free out-of-fold train embeddings) and the low-level
    ``get_embeddings`` call on its final full-data model.
    """
    embedder = NoriEmbedding(n_fold=5, shuffle=True, random_state=0, model=NoriRegressor(model="nori-6m"))
    Z_train = embedder.fit_transform(X_train, y_train).mean(axis=0)  # OOF, no leak
    Z_test = embedder.transform(X_test).mean(axis=0)  # full-data model

    raw = embedder.model_.get_embeddings(X_test, data_source="test")
    print(f"  get_embeddings -> (n_estimators, n_samples, embed_dim) = {raw.shape}; averaged to 2D {Z_test.shape}")

    native = np.asarray(embedder.model_.predict(X_test))  # Nori's native head
    return Z_train, Z_test, native


def ridge_probe(Z_train, y_train, Z_test, y_test):
    """A standardized Ridge regressor on the embeddings — a solid linear probe
    for the higher-dimensional embeddings of real data."""
    probe = make_pipeline(StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)))
    probe.fit(Z_train, y_train)
    return float(r2_score(y_test, probe.predict(Z_test)))


def tsne_plot(X_raw, Z_embed, color, title, fname, *, cbar_label):
    """Side-by-side t-SNE (raw features | Nori embeddings). Skipped if matplotlib
    is not installed, so the extract/probe path still runs without it.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: straight to file
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [skip t-SNE] matplotlib not installed (uv sync --extra eval)")
        return None

    def _tsne(M):
        M = StandardScaler().fit_transform(M)
        perp = float(min(30, max(5, (len(M) - 1) // 3)))  # valid for small n
        return TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(M)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    sc = None
    for ax, pts, sub in ((axes[0], _tsne(X_raw), "raw features"), (axes[1], _tsne(Z_embed), "Nori embeddings")):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=color, cmap="viridis", s=18, alpha=0.85, edgecolors="none")
        ax.set_title(sub, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.04).set_label(cbar_label)
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def subsample(X, y, n, seed=0):
    """Deterministically cap rows so the demo stays fast."""
    if len(X) <= n:
        return X, y
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    return X[idx], y[idx]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="TabArena regression dataset name (see --list)")
    p.add_argument("--list", action="store_true", help="print available dataset names and exit")
    args = p.parse_args()

    reg = load_registry()
    names = [k.split("/", 1)[1] for k in reg.list_datasets() if k.startswith("tabarena/")]
    if args.list:
        print("Available TabArena regression datasets:")
        for n in names:
            print(f"  {n}")
        return
    if not names:
        raise SystemExit("No TabArena datasets loaded (check network / `openml`).")

    entry = reg.get(f"tabarena/{args.dataset}")
    if entry is None:
        raise SystemExit(f"Dataset '{args.dataset}' not found. Try --list (e.g. {names[0]}).")

    X_train, y_train = entry.X_train, entry.y_train
    X_test, y_test = subsample(entry.X_test, entry.y_test, MAX_TEST)
    print(f"{entry.name}: train={len(X_train)} test={len(X_test)} features={X_train.shape[1]}")

    Z_train, Z_test, native = extract_embeddings(X_train, y_train, X_test)

    probe_r2 = ridge_probe(Z_train, y_train, Z_test, y_test)
    native_r2 = float(r2_score(y_test, native))
    print("\n=== R2 ===")
    print(f"  embedding dim              : {Z_train.shape[1]}")
    print(f"  Ridge probe on embeddings  : {probe_r2:.4f}")
    print(f"  Nori native head (ref.)    : {native_r2:.4f}")

    tsne_plot(
        X_test,
        Z_test,
        y_test,
        f"t-SNE — {entry.name} (color = target y)",
        f"embedding_tabarena_{entry.name}_tsne.png",
        cbar_label="target y",
    )


if __name__ == "__main__":
    main()
