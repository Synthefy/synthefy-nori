"""Permanent tests for the canonical fitted DataFrame feature contract."""

import builtins
import pickle

import numpy as np
import pandas as pd
import pytest

from synthefy.featurize import DataFramePreprocessor, align_and_featurize


def _rows(frame):
    return frame.to_numpy(dtype=float).tolist()


def test_canonical_ordinal_encoding_is_train_fitted_and_bounded():
    X_train, X_test = align_and_featurize(
        pd.DataFrame({"cat": ["y", "x"], "value": [1.0, 2.0]}),
        pd.DataFrame({"value": [3.0, 4.0, 5.0], "cat": ["x", "new", None]}),
    )

    assert list(X_train.columns) == list(X_test.columns) == ["cat", "value"]
    assert X_train["cat"].dtype == np.dtype("float32")
    assert _rows(X_train) == [[1.0, 1.0], [0.0, 2.0]]
    assert _rows(X_test[:2]) == [[0.0, 3.0], [2.0, 4.0]]  # unseen -> bounded other=K
    assert np.isnan(X_test.iloc[2, 0])  # missing remains missing, not "other"


def test_canonical_onehot_compatibility_uses_train_layout():
    X_train, X_test = align_and_featurize(
        pd.DataFrame({"value": [1.0, 2.0], "cat": ["x", "y"]}),
        pd.DataFrame({"value": [3.0], "cat": ["new"]}),
        categorical_encoding="onehot",
    )

    assert list(X_train.columns) == list(X_test.columns) == ["value", "cat_x", "cat_y"]
    assert _rows(X_train) == [[1.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
    assert _rows(X_test) == [[3.0, 0.0, 0.0]]


@pytest.mark.parametrize("categorical_columns", [None, ["declared"]])
def test_strict_modes_name_every_undeclared_non_numeric_column(categorical_columns):
    frame = pd.DataFrame(
        {
            "declared": pd.Series(["a", "b"], dtype="string"),
            "ambiguous": pd.Series(["x", "y"], dtype="category"),
        }
    )
    with pytest.raises(ValueError) as caught:
        DataFramePreprocessor(categorical_columns=categorical_columns).fit(frame)
    message = str(caught.value)
    if categorical_columns is None:
        assert "'declared' (string)" in message
    assert "'ambiguous' (category)" in message
    assert "categorical_columns" in message and "text_columns" in message
    assert "categorical_levels" in message


def test_explicit_categorical_uses_top_k_and_other_without_query_leakage():
    preprocessor = DataFramePreprocessor(categorical_columns=["plan"], max_categorical_cardinality=2)
    train = pd.DataFrame({"plan": ["pro", "free", "pro", "rare"], "amount": [1, 2, 3, 4]})
    query = pd.DataFrame({"amount": [5, 6, 7], "plan": ["free", "new", None]})

    X_train = preprocessor.fit_transform(train)
    mapping_before = dict(preprocessor.category_maps_["plan"])
    X_query = preprocessor.transform(query)

    assert mapping_before == {"free": 0, "pro": 1}
    assert preprocessor.category_maps_["plan"] == mapping_before
    assert X_train["plan"].tolist() == [1.0, 0.0, 1.0, 2.0]
    assert X_query["plan"].tolist()[:2] == [0.0, 2.0]
    assert np.isnan(X_query["plan"].iloc[2])


def test_auto_high_cardinality_requires_an_explicit_choice():
    frame = pd.DataFrame({"email": [f"user-{index}@example.com" for index in range(4)]})
    with pytest.raises(ValueError, match="Automatic handling is ambiguous") as caught:
        DataFramePreprocessor(max_categorical_cardinality=3).fit(frame)
    assert "categorical_columns=['email']" in str(caught.value)
    assert "text_columns" in str(caught.value)


def test_categorical_dtype_is_an_explicit_high_cardinality_role():
    train = pd.DataFrame({"category": pd.Series([f"value-{index}" for index in range(4)], dtype="category")})
    query = pd.DataFrame({"category": pd.Series(["value-0", "unseen"], dtype="category")})

    X_train, X_query = align_and_featurize(
        train,
        query,
        max_categorical_cardinality=2,
    )

    assert X_train["category"].tolist() == [0.0, 1.0, 2.0, 2.0]
    assert X_query["category"].tolist() == [0.0, 2.0]


def test_text_and_categorical_declarations_are_validated_before_text_import(monkeypatch):
    real_import = builtins.__import__

    def fail_text_import(name, *args, **kwargs):
        if name == "synthefy.text_features":
            raise AssertionError("text dependency must not be imported before schema validation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_text_import)
    frame = pd.DataFrame({"body": ["a", "b"]})
    with pytest.raises(ValueError, match="must not overlap"):
        DataFramePreprocessor(categorical_columns=["body"], text_columns=["body"]).fit(frame)


def test_categorical_only_never_imports_text_dependencies(monkeypatch):
    real_import = builtins.__import__

    def fail_text_import(name, *args, **kwargs):
        if name == "synthefy.text_features":
            raise AssertionError("categorical-only preprocessing imported text support")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_text_import)
    result = DataFramePreprocessor(categorical_columns=["plan"]).fit_transform(pd.DataFrame({"plan": ["free", "pro"]}))
    assert _rows(result) == [[0.0], [1.0]]


def test_transform_aligns_order_and_reports_missing_and_extra_columns():
    preprocessor = DataFramePreprocessor().fit(pd.DataFrame({"a": [1.0], "cat": ["x"]}))
    reordered = preprocessor.transform(pd.DataFrame({"cat": ["x"], "a": [2.0]}))
    assert list(reordered.columns) == ["a", "cat"]

    with pytest.raises(ValueError) as caught:
        preprocessor.transform(pd.DataFrame({"a": [2.0], "extra": [3.0]}))
    assert "missing columns=['cat']" in str(caught.value)
    assert "extra columns=['extra']" in str(caught.value)


def test_categorical_dtype_is_respected_even_when_levels_are_numeric():
    frame = pd.DataFrame({"rating": pd.Categorical([3, 1, 3], categories=[1, 2, 3])})
    result = DataFramePreprocessor().fit_transform(frame)
    assert result["rating"].tolist() == [1.0, 0.0, 1.0]


def test_fitted_preprocessor_pickles_with_schema_and_mappings():
    original = DataFramePreprocessor(categorical_columns=["cat"])
    original.fit(pd.DataFrame({"cat": ["x", "y"], "n": [1.0, 2.0]}))
    restored = pickle.loads(pickle.dumps(original))
    result = restored.transform(pd.DataFrame({"n": [3.0], "cat": ["new"]}))
    assert _rows(result) == [[2.0, 3.0]]
    assert list(restored.feature_names_in_) == ["cat", "n"]


def test_validation_and_positional_compatibility():
    with pytest.raises(ValueError, match="same feature columns"):
        align_and_featurize(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"b": [1.0]}))
    with pytest.raises(ValueError, match="categorical_encoding"):
        align_and_featurize(
            pd.DataFrame({"cat": ["x"]}),
            pd.DataFrame({"cat": ["x"]}),
            categorical_encoding="hashing",
        )

    X_train = [[1.0, 2.0]]
    X_test = [[3.0, 4.0]]
    actual_train, actual_test = align_and_featurize(X_train, X_test)
    assert actual_train is X_train and actual_test is X_test


def test_column_declaration_iterators_are_materialized_once():
    train = pd.DataFrame({"cat": ["x", "y"], "value": [1.0, 2.0]})
    query = pd.DataFrame({"value": [3.0], "cat": ["x"]})

    X_train, X_test = align_and_featurize(
        train,
        query,
        categorical_columns=iter(["cat"]),
        text_columns=iter(()),
    )

    assert _rows(X_train) == [[0.0, 1.0], [1.0, 2.0]]
    assert _rows(X_test) == [[0.0, 3.0]]
