import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from synthefy_nori import NoriRegressor
from synthefy_nori.entity_ids import EntityIDPreprocessor


def test_multiple_ids_context_only_large_integer_categories_and_query():
    x = pd.DataFrame(
        {"sku": ["a", "b"] * 3, "asset": pd.Categorical([2**63 + 1, 2**63 + 2] * 3), "price": np.arange(6.0)}
    )
    original = x.copy(deep=True)
    m = clone(NoriRegressor(model="nori-30m", device="cpu", encode_id_cols=["sku", "asset"]))
    m.fit(x, np.arange(6.0))
    np.testing.assert_array_equal(m.X_train_[:, :2], [[0, 0], [1, 1]] * 3)
    q = pd.DataFrame(
        {
            "price": [1.0, 2.0, 3.0],
            "asset": pd.Series([2**63 + 2, 2**63 + 3, None], dtype="UInt64"),
            "sku": ["b", "new", None],
        }
    )
    out = m._prepare_query_features(q)
    np.testing.assert_array_equal(out[:, :2], [[1, 1], [-1, -1], [-1, -1]])
    pd.testing.assert_frame_equal(x, original)
    encoded = pickle.loads(pickle.dumps(m._entity_id_preprocessor)).transform(q)
    assert encoded.columns.tolist() == q.columns.tolist()
    assert encoded.index.equals(q.index)


@pytest.mark.parametrize("cols", [None, []])
def test_disabled_preserves_default_arrays(cols):
    x = pd.DataFrame({"sku": ["a", "b"] * 3})
    a = NoriRegressor(model="nori-30m", device="cpu", encode_id_cols=cols).fit(x, np.arange(6.0))
    b = NoriRegressor(model="nori-30m", device="cpu").fit(x, np.arange(6.0))
    np.testing.assert_array_equal(a.X_train_, b.X_train_)


@pytest.mark.parametrize("cols", ["sku", ["sku", "sku"], [1], [""]])
def test_invalid_column_declarations(cols):
    with pytest.raises(ValueError):
        EntityIDPreprocessor(cols)


@pytest.mark.parametrize("values", [[True, False], [1, "a"], [True, 2], [1.5, 2.5]])
def test_invalid_id_values(values):
    with pytest.raises(ValueError, match="integers or strings"):
        EntityIDPreprocessor(["sku"]).fit(pd.DataFrame({"sku": pd.Series(values, dtype=object)}))


def test_missing_and_conflicting_columns():
    x = pd.DataFrame({"sku": ["a", "b"] * 3})
    p = EntityIDPreprocessor(["sku"]).fit(x)
    with pytest.raises(ValueError, match="not found"):
        p.transform(x.rename(columns={"sku": "other"}))
    with pytest.raises(ValueError, match="DataFrame"):
        p.transform(np.ones((2, 1)))
    with pytest.raises(ValueError, match="overlap"):
        NoriRegressor(model="nori-30m", device="cpu", encode_id_cols=["sku"], categorical_columns=["sku"]).fit(
            x, np.arange(6.0)
        )


def test_unused_integer_categories_missing_context_and_refit():
    p = EntityIDPreprocessor(["sku"]).fit(pd.DataFrame({"sku": pd.Categorical([1, 2], categories=[1, 2, 3])}))
    assert p.transform(pd.DataFrame({"sku": pd.Categorical([2, 3, None])})).sku.tolist() == [1, -1, -1]
    p.fit(pd.DataFrame({"sku": [None, None]}))
    assert p.transform(pd.DataFrame({"sku": ["new", None]})).sku.tolist() == [-1, -1]


def test_marginal_and_refit_do_not_reencode(monkeypatch):
    x = pd.DataFrame({"sku": ["a", "b"] * 3})
    m = NoriRegressor(model="nori-30m", device="cpu", encode_id_cols=["sku"]).fit(x, np.arange(6.0))
    monkeypatch.setattr(m, "_get_predictor", lambda: object())
    marginal = m._make_marginal_estimator(m.X_train_, np.arange(6.0))
    np.testing.assert_array_equal(marginal._prepare_query_features(m.X_train_), m.X_train_)
    m.set_params(encode_id_cols=None).fit(pd.DataFrame({"value": np.arange(6.0)}), np.arange(6.0))
    assert m._entity_id_preprocessor is None
