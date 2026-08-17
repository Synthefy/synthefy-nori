"""Nori embeddings on a synthetic dataset: extract, probe, and visualize.

Nori's encoder turns every row into a learned vector. This example shows the two
ways to pull those vectors out, uses them for a downstream probe, and draws a
t-SNE that makes the target structure visible — all on a small synthetic
regression dataset. See ``embedding_tabarena.py`` for the same workflow on a real
dataset.

    uv run python examples/embedding_synthetic.py     # extract + probe
    uv sync --extra eval                              # adds matplotlib (t-SNE)
    uv run python examples/embedding_synthetic.py     # ... now also writes a PNG

Two ways to get embeddings — both return a 3D array
``(n_estimators, n_samples, embed_dim)``; average over axis 0 (the
preprocessing-pipeline ensemble) for a 2D feature matrix:
  1. ``NoriRegressor.get_embeddings(X, data_source=...)`` — the low-level call.
  2. ``NoriEmbedding`` — an sklearn transformer; with ``n_fold >= 2`` it produces
     leakage-free out-of-fold embeddings for the training rows.

The first run downloads the public ~47MB checkpoint; GPU if available, else CPU.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from synthefy_nori import NoriEmbedding, NoriRegressor

OUT_DIR = "results"


def extract_embeddings(X_train, y_train, X_test):
    """Return OOF train embeddings, test embeddings (both 2D), and Nori's own
    native test prediction — using a single fitted model for all three.

    Demonstrates both embedding APIs: the ``NoriEmbedding`` transformer (for the
    leakage-free out-of-fold train embeddings) and the low-level
    ``get_embeddings`` call on its final full-data model.
    """
    embedder = NoriEmbedding(n_fold=5, shuffle=True, random_state=0,
                             model=NoriRegressor(model="nori-6m"))
    Z_train = embedder.fit_transform(X_train, y_train).mean(axis=0)  # OOF, no leak
    Z_test = embedder.transform(X_test).mean(axis=0)                 # full-data model

    # Low-level call on the same fitted model (no extra load); shows the raw 3D
    # shape before we average over the ensemble axis.
    raw = embedder.model_.get_embeddings(X_test, data_source="test")
    print(f"  get_embeddings -> (n_estimators, n_samples, embed_dim) = {raw.shape}"
          f"; averaged to 2D {Z_test.shape}")

    native = np.asarray(embedder.model_.predict(X_test))  # Nori's native head
    return Z_train, Z_test, native


def knn_probe(Z_train, y_train, Z_test, y_test):
    """A plain kNN regressor on the embeddings — the simplest downstream head."""
    knn = KNeighborsRegressor(n_neighbors=10).fit(Z_train, y_train)
    return float(r2_score(y_test, knn.predict(Z_test)))


def tsne_plot(X_raw, Z_embed, color, title, fname, *, cbar_label):
    """Side-by-side t-SNE (raw features | Nori embeddings). Skipped if matplotlib
    is not installed, so the extract/probe path still runs without ``--extra eval``.
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
        return TSNE(n_components=2, perplexity=perp, init="pca",
                    random_state=0).fit_transform(M)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    sc = None
    for ax, pts, sub in ((axes[0], _tsne(X_raw), "raw features"),
                         (axes[1], _tsne(Z_embed), "Nori embeddings")):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=color, cmap="viridis", s=18,
                        alpha=0.85, edgecolors="none")
        ax.set_title(sub, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.04).set_label(cbar_label)
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def make_dataset(n=400, seed=0):
    """A smooth, nonlinear regression target with an interaction term."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 5)).astype(np.float32)
    y = (np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2] * X[:, 3]).astype(np.float64)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def main():
    X_train, X_test, y_train, y_test = make_dataset()
    print(f"synthetic regression: train={len(X_train)} test={len(X_test)} "
          f"features={X_train.shape[1]}")

    Z_train, Z_test, native = extract_embeddings(X_train, y_train, X_test)

    probe_r2 = knn_probe(Z_train, y_train, Z_test, y_test)
    native_r2 = float(r2_score(y_test, native))
    print("\n=== R2 ===")
    print(f"  embedding dim            : {Z_train.shape[1]}")
    print(f"  kNN probe on embeddings  : {probe_r2:.4f}")
    print(f"  Nori native head (ref.)  : {native_r2:.4f}")

    tsne_plot(X_test, Z_test, y_test,
              "t-SNE — synthetic regression (color = target y)",
              "embedding_synthetic_tsne.png", cbar_label="target y")


if __name__ == "__main__":
    main()
