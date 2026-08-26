"""Embeddings cluster by target — a representation-quality check.

Motivated by "A Closer Look at TabPFN v2: Strength, Limitation, and Extension"
(arXiv:2502.17361), which shows the model's learned embeddings are strong
representations whose geometry reflects the target: same-class points group
together, and downstream models on the embeddings are competitive. Nori is
regression-only, so we treat a binary target as a {0,1} regression target and
also bin a continuous target into quantile groups.

For each synthetic dataset we extract the test-row embeddings (averaged over the
inference-pipeline ensemble) and check they cluster by label markedly better
than the RAW features do — silhouette, KMeans agreement (adjusted Rand), and a
kNN probe. These are marked slow: they download the public checkpoint.
"""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from synthefy_nori import NoriRegressor

pytestmark = pytest.mark.slow


def _embed_test_rows(X_train, y_train, X_test):
    """Fit on the context, return mean-over-ensemble test embeddings [n, dim]."""
    model = NoriRegressor(model="nori-6m").fit(X_train, y_train)
    emb = model.get_embeddings(X_test, data_source="test")  # (n_est, n, dim)
    return emb.mean(axis=0)


def _kmeans_ari(Z, labels, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    return adjusted_rand_score(labels, km)


def test_binary_classification_embeddings_separate_by_class():
    """Two interleaving moons: not linearly separable, so KMeans on the raw
    features barely recovers the classes. Nori's embeddings — whose target token
    encodes the (predicted) class — should separate the two moons much better."""
    X, y = make_moons(n_samples=500, noise=0.18, random_state=0)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

    Z = _embed_test_rows(X_train, y_train.astype(np.float64), X_test)
    Xs = StandardScaler().fit_transform(X_test)  # fair raw baseline (scaled)

    sil_emb, sil_raw = silhouette_score(Z, y_test), silhouette_score(Xs, y_test)
    ari_emb, ari_raw = _kmeans_ari(Z, y_test, 2), _kmeans_ari(Xs, y_test, 2)
    # Unsupervised kNN probe on the embeddings (leave-one-out style via CV).
    knn_acc = accuracy_score(y_test, cross_val_predict(KNeighborsClassifier(15), Z, y_test, cv=5))

    print(f"\n[moons] silhouette  embed={sil_emb:+.3f} raw={sil_raw:+.3f}")
    print(f"[moons] KMeans ARI  embed={ari_emb:+.3f} raw={ari_raw:+.3f}")
    print(f"[moons] kNN acc on embeddings = {knn_acc:.3f}")

    assert sil_emb > sil_raw  # embeddings are more clusterable
    assert ari_emb > ari_raw + 0.10  # KMeans recovers classes far better
    assert ari_emb > 0.5  # and recovers them well in absolute terms
    assert knn_acc > 0.9  # near-perfectly class-separated


def test_regression_embeddings_organize_by_target():
    """Continuous nonlinear target. Embeddings should be organized by target
    magnitude: binning y into low/mid/high groups, the embeddings separate the
    groups better than raw features, and a kNN regressor on them scores high."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(500, 5)).astype(np.float32)
    y = (np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2] * X[:, 3]).astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    Z = _embed_test_rows(X_train, y_train, X_test)
    Xs = StandardScaler().fit_transform(X_test)

    # Quantile bins of the (held-out) target define the "clusters".
    bins = np.digitize(y_test, np.quantile(y_test, [1 / 3, 2 / 3]))
    sil_emb, sil_raw = silhouette_score(Z, bins), silhouette_score(Xs, bins)
    ari_emb, ari_raw = _kmeans_ari(Z, bins, 3), _kmeans_ari(Xs, bins, 3)
    knn_r2 = r2_score(y_test, cross_val_predict(KNeighborsRegressor(15), Z, y_test, cv=5))

    print(f"\n[reg] silhouette(target-bins)  embed={sil_emb:+.3f} raw={sil_raw:+.3f}")
    print(f"[reg] KMeans ARI(target-bins)  embed={ari_emb:+.3f} raw={ari_raw:+.3f}")
    print(f"[reg] kNN-on-embeddings R2 = {knn_r2:.3f}")

    assert sil_emb > sil_raw  # embeddings group same-target points
    assert ari_emb > ari_raw  # and KMeans recovers target groups better
    assert knn_r2 > 0.85  # embedding neighbors share the target
