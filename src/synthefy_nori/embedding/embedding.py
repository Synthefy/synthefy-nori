"""scikit-learn transformer that extracts Nori embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from synthefy_nori.api import NoriRegressor


class NoriEmbedding(TransformerMixin, BaseEstimator):
    """scikit-learn style transformer that extracts Nori embeddings.

    When ``n_fold >= 2``, ``fit`` produces out-of-fold (OOF) embeddings for the
    training data — each training row is embedded by a model that did *not*
    have it in its context, avoiding leakage — and then refits a single model
    on the full training set for use on unseen data. The OOF embeddings are
    stored on ``train_embeddings_`` and returned by ``fit_transform``.

    ``transform(X)`` ALWAYS uses the final, full-data model — it does NOT
    return cached OOF embeddings, even when ``X`` happens to equal the training
    set. For OOF embeddings call ``fit_transform`` (or read
    ``train_embeddings_``).

    Note on output shape: ``transform`` returns a 3D array of shape
    ``(n_estimators, n_samples, embed_dim)`` (``n_estimators`` is the number of
    preprocessing pipelines in the inference config). It is not a drop-in input
    for ``sklearn.pipeline.Pipeline`` / ``ColumnTransformer``, which expect 2D
    output. Pick an ensemble member (``embeds[0]``) or aggregate across
    ``axis=0`` before passing to a downstream 2D estimator.

    Parameters
    ----------
    n_fold : int, default=0
        Number of folds for cross-validation. ``0`` disables CV — the model is
        fit once on the entire training set and used for both train and unseen
        data. Must be ``0`` or ``>= 2``.
    model : NoriRegressor
        Pre-configured estimator to embed with. **Required** — pass a ``NoriRegressor``
        with an explicit size (e.g. ``NoriRegressor(model="nori-30m")``); there is no
        default, and ``None`` raises at ``fit``.
    shuffle : bool, default=False
        Whether to shuffle the K-fold split. Independent of ``random_state``.
    random_state : int, optional
        Seed used by the K-fold split when ``shuffle=True``.

    Attributes
    ----------
    model_ : NoriRegressor
        The fitted model (cloned from ``model`` or auto-constructed). After
        ``fit`` with ``n_fold >= 2`` this is the model fit on the full
        training set.
    train_embeddings_ : np.ndarray
        Embeddings for the training set. For ``n_fold >= 2`` these are OOF
        embeddings aligned to the original sample order; for ``n_fold == 0``
        they come from the single full-data model.

    Examples
    --------
    >>> from synthefy_nori import NoriRegressor
    >>> from synthefy_nori.embedding import NoriEmbedding
    >>> embedding = NoriEmbedding(n_fold=5, model=NoriRegressor(model="nori-30m"))
    >>> train_embeds = embedding.fit_transform(X_train, y_train)  # OOF
    >>> test_embeds = embedding.transform(X_test)                 # final model
    """

    def __init__(
        self,
        n_fold: int = 0,
        *,
        model: NoriRegressor | None = None,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_fold = n_fold
        self.model = model
        self.shuffle = shuffle
        self.random_state = random_state

    def _resolve_template(self) -> NoriRegressor:
        """Return a fresh model to use: a clone of ``model``. ``model`` is required -- there is
        no default; pass a NoriRegressor with an explicit size."""
        if self.model is None:
            raise ValueError(
                "NoriEmbedding requires model=<a NoriRegressor with an explicit size>, e.g. "
                "NoriEmbedding(model=NoriRegressor(model='nori-30m')). There is no default."
            )
        return clone(self.model)

    def _compute_oof(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Run K-fold and return OOF embeddings aligned to original order."""
        X = np.asarray(X)
        y = np.asarray(y)

        rs = self.random_state if self.shuffle else None
        cv = KFold(n_splits=self.n_fold, shuffle=self.shuffle, random_state=rs)

        chunks: list[np.ndarray] = []
        val_indices: list[np.ndarray] = []
        for train_idx, val_idx in cv.split(X):
            # Reuse the single self.model_ instance across folds instead of
            # clone()-ing per fold. NoriRegressor.fit only overwrites the stored
            # context arrays and leaves the cached _predictor (checkpoint +
            # torch.compile) intact, and get_embeddings takes the context
            # explicitly, so nothing is fold-specific — one compiled predictor
            # serves every fold. clone() resets _predictor (it is set in the
            # __init__ body, not a get_params() param), which would pay the cold
            # torch.compile (~minutes on CUDA) again on every fold.
            self.model_.fit(X[train_idx], y[train_idx])
            chunks.append(self.model_.get_embeddings(X[val_idx], data_source="test"))
            val_indices.append(val_idx)

        oof = np.concatenate(chunks, axis=1)
        order = np.argsort(np.concatenate(val_indices))
        return oof[:, order, ...]

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoriEmbedding:
        if self.n_fold < 0 or self.n_fold == 1:
            raise ValueError("n_fold must be 0 (vanilla) or >= 2.")

        self.model_ = self._resolve_template()

        if self.n_fold == 0:
            self.model_.fit(X, y)
            self.train_embeddings_ = self.model_.get_embeddings(X, data_source="test")
            return self

        self.train_embeddings_ = self._compute_oof(X, y)
        self.model_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embed unseen data ``X`` using the full-data model.

        Always runs inference through ``model_`` (fit on the full training set)
        and never returns cached embeddings. For *training*-data embeddings,
        prefer ``fit_transform`` (OOF for ``n_fold >= 2``) or read
        ``train_embeddings_``.
        """
        check_is_fitted(self, "model_")
        return self.model_.get_embeddings(X, data_source="test")

    def fit_transform(self, X: np.ndarray, y: np.ndarray, **fit_params) -> np.ndarray:
        """Fit and return embeddings for the training data.

        For ``n_fold >= 2`` these are out-of-fold embeddings; for ``n_fold == 0``
        they come from the single full-data model.
        """
        self.fit(X, y)
        return self.train_embeddings_


__all__ = ["NoriEmbedding"]
