"""NoriEmbedding: OOF/fold logic + sklearn compliance behind a stub model,
plus a slow end-to-end embedding extraction against the real checkpoint."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone

from synthefy_nori import NoriEmbedding, NoriRegressor


class _StubEmbModel(RegressorMixin, BaseEstimator):
    """A fitted-regressor stand-in whose embedding of a row is the row itself.

    Returns shape (1, n_samples, n_features) — the (n_estimators, n, d) contract
    with a single ensemble member — so OOF reassembly is checkable exactly.
    """

    def fit(self, X, y):
        self.X_ = np.asarray(X)
        return self

    def get_embeddings(self, X, *, data_source="test"):
        return np.asarray(X, dtype=float)[None, :, :]


class _StubPredictor:
    """Return preprocessed query features as embeddings without loading a checkpoint."""

    def get_embeddings(self, X_train, y_train, X_test, *, data_source="test"):
        del X_train, y_train
        assert data_source == "test"
        return np.asarray(X_test, dtype=np.float32)[None, :, :]


def test_embedding_is_sklearn_estimator_and_clones():
    e = NoriEmbedding(n_fold=5, shuffle=True, random_state=7)
    params = e.get_params()
    assert params["n_fold"] == 5 and params["random_state"] == 7
    c = clone(e)
    assert c.get_params()["n_fold"] == 5

    # model is required -- there is no default template
    with pytest.raises(ValueError, match="requires model"):
        NoriEmbedding()._resolve_template()
    # with an explicit model, the template is a clone of it (no weights touched)
    assert isinstance(
        NoriEmbedding(model=NoriRegressor(model="nori-6m"))._resolve_template(),
        NoriRegressor,
    )


def test_n_fold_one_is_rejected():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)
    with pytest.raises(ValueError, match="n_fold must be 0"):
        NoriEmbedding(n_fold=1, model=_StubEmbModel()).fit(X, y)


def test_vanilla_transform_uses_full_data_model():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)
    e = NoriEmbedding(n_fold=0, model=_StubEmbModel())
    train_emb = e.fit_transform(X, y)
    assert train_emb.shape == (1, 10, 2)
    np.testing.assert_allclose(train_emb[0], X)

    X_new = np.full((3, 2), 99.0)
    test_emb = e.transform(X_new)
    np.testing.assert_allclose(test_emb[0], X_new)


def test_oof_embeddings_are_aligned_to_original_order():
    """Each row is embedded as itself, so OOF output must reconstruct X exactly
    in the original row order despite the K-fold shuffle."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 4))
    y = rng.normal(size=25)
    e = NoriEmbedding(n_fold=5, shuffle=True, random_state=0, model=_StubEmbModel())
    oof = e.fit_transform(X, y)
    assert oof.shape == (1, 25, 4)
    np.testing.assert_allclose(oof[0], X)
    # fit also refits the full-data model for transform()
    np.testing.assert_allclose(e.transform(X[:2])[0], X[:2])


@pytest.mark.parametrize("n_fold", [0, 2])
@pytest.mark.parametrize("categorical_columns", ["auto", ["plan"]], ids=["auto", "explicit"])
def test_dataframe_categorical_columns_survive_folds(monkeypatch, n_fold, categorical_columns):
    monkeypatch.setattr(NoriRegressor, "_get_predictor", lambda self: _StubPredictor())
    X = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0, 40.0],
            "plan": ["free", "pro", "free", "pro"],
        },
        index=[10, 20, 30, 40],
    )
    y = np.arange(len(X), dtype=float)
    embedding = NoriEmbedding(
        n_fold=n_fold,
        model=NoriRegressor(model_path="unused.pt", categorical_columns=categorical_columns),
    )

    train_embeddings = embedding.fit_transform(X, y)

    np.testing.assert_equal(
        train_embeddings[0],
        np.asarray([[10.0, 0.0], [20.0, 1.0], [30.0, 0.0], [40.0, 1.0]], dtype=np.float32),
    )
    assert embedding.model_._feature_preprocessor.categorical_columns_ == ["plan"]


def test_dataframe_transform_replays_fitted_schema_after_oof(monkeypatch):
    monkeypatch.setattr(NoriRegressor, "_get_predictor", lambda self: _StubPredictor())
    X = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0, 40.0],
            "plan": ["free", "pro", "free", "pro"],
        }
    )
    embedding = NoriEmbedding(
        n_fold=2,
        model=NoriRegressor(model_path="unused.pt", categorical_columns=["plan"]),
    ).fit(X, np.arange(len(X), dtype=float))

    query = pd.DataFrame({"plan": ["enterprise"], "amount": [50.0]})
    np.testing.assert_equal(
        embedding.transform(query)[0],
        np.asarray([[50.0, 2.0]], dtype=np.float32),
    )

    mismatched = pd.DataFrame({"amount": [50.0], "region": ["us"]})
    with pytest.raises(ValueError) as caught:
        embedding.transform(mismatched)
    assert "missing columns=['plan']" in str(caught.value)
    assert "extra columns=['region']" in str(caught.value)


@pytest.mark.slow
def test_end_to_end_embeddings_real_checkpoint():
    """Downloads the public checkpoint and extracts real embeddings."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(60, 5)).astype(np.float32)
    y_train = (X_train[:, 0] * 2 - X_train[:, 1]).astype(np.float64)
    X_test = rng.normal(size=(8, 5)).astype(np.float32)

    model = NoriRegressor(model="nori-6m").fit(X_train, y_train)

    test_emb = model.get_embeddings(X_test, data_source="test")
    assert test_emb.ndim == 3
    n_estimators, n_samples, embed_dim = test_emb.shape
    assert n_samples == 8 and n_estimators >= 1 and embed_dim > 0
    assert np.isfinite(test_emb).all()

    train_emb = model.get_embeddings(X_test, data_source="train")
    assert train_emb.shape == (n_estimators, 60, embed_dim)

    # Same end-to-end through the sklearn transformer.
    emb = NoriEmbedding(n_fold=0, model=NoriRegressor(model="nori-6m")).fit_transform(X_train, y_train)
    assert emb.shape[1] == 60 and emb.shape[2] == embed_dim
