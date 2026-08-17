"""Featurization of non-numeric DataFrame inputs (fit on X_train).

These exercise :func:`synthefy_nori.featurize.align_and_featurize`, the pure
DataFrame-in / DataFrame-out transform that the public ``infer`` / ``predict``
helpers apply before handing a fully numeric matrix to the model — so they run
without loading any checkpoint. The default encoding is ordinal; the one-hot
path is exercised with ``categorical_encoding="onehot"``.
"""

import builtins
import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

from synthefy.featurize import (
    CATEGORICAL_ENCODINGS as CANONICAL_CATEGORICAL_ENCODINGS,
    DEFAULT_CATEGORICAL_ENCODING as CANONICAL_DEFAULT_CATEGORICAL_ENCODING,
    DEFAULT_MAX_CARDINALITY as CANONICAL_DEFAULT_MAX_CARDINALITY,
    align_and_featurize as canonical_align_and_featurize,
)
import synthefy_nori.featurize as legacy_featurize_module
from synthefy_nori.featurize import (
    CATEGORICAL_ENCODINGS,
    DEFAULT_CATEGORICAL_ENCODING,
    DEFAULT_MAX_CARDINALITY,
    align_and_featurize,
)


def _rows(frame):
    return frame.to_numpy(dtype=float).tolist()


def test_synthefy_owns_the_legacy_v7_entry_points():
    assert align_and_featurize is canonical_align_and_featurize
    assert CATEGORICAL_ENCODINGS is CANONICAL_CATEGORICAL_ENCODINGS
    assert DEFAULT_CATEGORICAL_ENCODING == CANONICAL_DEFAULT_CATEGORICAL_ENCODING
    assert DEFAULT_MAX_CARDINALITY == CANONICAL_DEFAULT_MAX_CARDINALITY


def test_legacy_module_falls_back_only_when_the_canonical_submodule_is_missing(
    monkeypatch,
):
    real_import = builtins.__import__

    def missing_canonical(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "synthefy.featurize":
            raise ModuleNotFoundError(
                "No module named 'synthefy.featurize'",
                name="synthefy.featurize",
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_canonical)
    try:
        fallback = importlib.reload(legacy_featurize_module)
        assert fallback.align_and_featurize.__module__ == (
            "synthefy_nori._legacy_featurize"
        )
        Xtr, Xte = fallback.align_and_featurize(
            pd.DataFrame({"a": [1.0, 2.0], "cat": ["y", "x"]}),
            pd.DataFrame({"cat": ["z"], "a": [3.0]}),
        )
        assert list(Xtr.columns) == list(Xte.columns) == ["a", "cat"]
        assert _rows(Xtr) == [[1.0, 1.0], [2.0, 0.0]]
        assert _rows(Xte) == [[3.0, -1.0]]
    finally:
        monkeypatch.undo()
        importlib.reload(legacy_featurize_module)

    assert legacy_featurize_module.align_and_featurize is canonical_align_and_featurize


def test_legacy_module_does_not_mask_a_transitive_import_failure(monkeypatch):
    real_import = builtins.__import__

    def broken_canonical_dependency(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "synthefy.featurize":
            raise ModuleNotFoundError(
                "No module named 'sentinel_tabular_dependency'",
                name="sentinel_tabular_dependency",
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_canonical_dependency)
    try:
        with pytest.raises(ModuleNotFoundError) as caught:
            importlib.reload(legacy_featurize_module)
        assert caught.value.name == "sentinel_tabular_dependency"
    finally:
        monkeypatch.undo()
        importlib.reload(legacy_featurize_module)


def test_non_numeric_columns_are_ordinal_encoded_by_default():
    Xtr, Xte = align_and_featurize(
        pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["y", "x", "y"]}),
        # 'z' is unseen in training -> bounded other code K=2.
        pd.DataFrame({"a": [3.0, 4.0], "cat": ["x", "z"]}),
    )
    # one column per categorical, codes in sorted-category order: x=0, y=1
    assert list(Xtr.columns) == ["a", "cat"]
    assert _rows(Xtr) == [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]
    assert _rows(Xte) == [[3.0, 0.0], [4.0, 2.0]]


def test_ordinal_missing_categorical_is_nan():
    Xtr, _ = align_and_featurize(
        pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["x", None, "y"]}),
        pd.DataFrame({"a": [5.0], "cat": ["x"]}),
    )
    # x=0, y=1; the missing row stays NaN for server-side imputation.
    assert Xtr["cat"].tolist()[0] == 0.0 and Xtr["cat"].tolist()[2] == 1.0
    assert np.isnan(Xtr["cat"].tolist()[1])


def test_ordinal_column_order_preserved():
    # categorical stays in place (not moved after numerics like one-hot does)
    Xtr, _ = align_and_featurize(
        pd.DataFrame({"cat": ["b", "a"], "n": [1.0, 2.0]}),
        pd.DataFrame({"cat": ["a"], "n": [3.0]}),
    )
    assert list(Xtr.columns) == ["cat", "n"]


def test_invalid_categorical_encoding_raises():
    with pytest.raises(ValueError, match="categorical_encoding"):
        align_and_featurize(
            pd.DataFrame({"cat": ["x", "y"]}),
            pd.DataFrame({"cat": ["x"]}),
            categorical_encoding="hashing",
        )


def test_non_numeric_columns_are_one_hot_encoded():
    Xtr, Xte = align_and_featurize(
        pd.DataFrame({"a": [0.0, 1.0], "cat": ["x", "y"]}),
        # 'z' is unseen in training -> its indicator group is all zeros.
        pd.DataFrame({"a": [2.0], "cat": ["z"]}),
        categorical_encoding="onehot",
    )
    # columns: a, cat_x, cat_y  (numerics first, then sorted one-hot groups)
    assert list(Xtr.columns) == ["a", "cat_x", "cat_y"]
    assert _rows(Xtr) == [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    assert _rows(Xte) == [[2.0, 0.0, 0.0]]


def test_one_hot_train_category_absent_in_test_is_kept_as_zero_column():
    _, Xte = align_and_featurize(
        pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["x", "y", "z"]}),
        pd.DataFrame({"a": [5.0], "cat": ["x"]}),
        categorical_encoding="onehot",
    )
    # train has 3 categories -> cat_x, cat_y, cat_z; test row 'x' -> [1,0,0]
    assert _rows(Xte) == [[5.0, 1.0, 0.0, 0.0]]


def test_column_order_in_test_need_not_match_train():
    Xtr, Xte = align_and_featurize(
        pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}),
        pd.DataFrame({"b": [2.0], "a": [3.0]}),  # reversed order
    )
    assert list(Xtr.columns) == list(Xte.columns) == ["a", "b"]
    assert _rows(Xte) == [[3.0, 2.0]]


def test_auto_high_cardinality_column_requires_explicit_role():
    with pytest.raises(ValueError, match="Automatic handling is ambiguous"):
        align_and_featurize(
            pd.DataFrame({"a": [0.0, 1.0, 2.0], "hc": ["p", "q", "r"]}),
            pd.DataFrame({"a": [3.0], "hc": ["p"]}),
            max_categorical_cardinality=2,
        )


def test_datetime_column_requires_explicit_conversion():
    with pytest.raises(ValueError, match="unsupported dtype"):
        align_and_featurize(
            pd.DataFrame(
                {"a": [0.0, 1.0], "d": pd.to_datetime(["2024-01-01", "2024-01-02"])}
            ),
            pd.DataFrame({"a": [2.0], "d": pd.to_datetime(["2024-01-03"])}),
        )


def test_bool_columns_pass_through_as_numeric():
    # bool is numeric (is_numeric_dtype) -> not one-hot; True/False -> 1.0/0.0
    Xtr, Xte = align_and_featurize(
        pd.DataFrame({"a": [0.0, 1.0], "flag": [True, False]}),
        pd.DataFrame({"a": [2.0], "flag": [True]}),
    )
    assert _rows(Xtr) == [[0.0, 1.0], [1.0, 0.0]]
    assert _rows(Xte) == [[2.0, 1.0]]


def test_all_numeric_dataframe_is_not_featurized():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any featurization warning would fail
        Xtr, Xte = align_and_featurize(
            pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}),
            pd.DataFrame({"a": [2.0], "b": [2.0]}),
        )
    assert _rows(Xtr) == [[0.0, 1.0], [1.0, 0.0]]


def test_non_dataframe_test_with_categorical_train_raises():
    # A list/numpy X_test has no column names, so a non-numeric X_train column
    # can't be aligned and one-hot encoded — we raise and point at DataFrames.
    with pytest.raises(ValueError, match="not a DataFrame"):
        align_and_featurize(
            pd.DataFrame({"a": [1.0, 2.0], "cat": ["x", "y"]}),
            [[3.0, 9.0]],
        )


def test_both_non_dataframe_numeric_pass_through_unchanged():
    Xtr = [[1.0, 2.0]]
    Xte = [[3.0, 4.0]]
    out_tr, out_te = align_and_featurize(Xtr, Xte)
    assert out_tr is Xtr and out_te is Xte


def test_column_numeric_in_train_but_not_test_raises_clearly():
    # A column numeric in X_train but object-dtype in X_test is caught with a
    # clear type-mismatch error (not a later cryptic float-cast failure).
    with pytest.raises(ValueError, match="matching column types"):
        align_and_featurize(
            pd.DataFrame({"b": [1.0, 2.0]}),
            pd.DataFrame({"b": ["x"]}),  # object dtype, not numeric
        )


def test_numeric_category_dtype_is_respected_as_categorical():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must NOT warn / drop / explode
        Xtr, Xte = align_and_featurize(
            pd.DataFrame(
                {"a": [0.0, 1.0, 2.0], "r": pd.Categorical([1, 2, 3], categories=[1, 2, 3])}
            ),
            pd.DataFrame({"a": [5.0], "r": pd.Categorical([2], categories=[1, 2, 3])}),
        )
    # Pandas categorical dtype is a semantic declaration even when levels are numbers.
    assert _rows(Xtr) == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    assert _rows(Xte) == [[5.0, 1.0]]


def test_all_missing_categorical_column_preserves_schema_and_nan():
    Xtr, Xte = align_and_featurize(
        pd.DataFrame({"a": [0.0, 1.0], "cat": [None, None]}),
        pd.DataFrame({"a": [2.0], "cat": [None]}),
    )
    assert list(Xtr.columns) == list(Xte.columns) == ["a", "cat"]
    assert Xtr["cat"].isna().all() and Xte["cat"].isna().all()


def test_timedelta_column_raises_unsupported():
    with pytest.raises(ValueError, match="timedelta"):
        align_and_featurize(
            pd.DataFrame({"a": [0.0, 1.0], "d": pd.to_timedelta(["1 days", "2 days"])}),
            pd.DataFrame({"a": [2.0], "d": pd.to_timedelta(["3 days"])}),
        )


def test_nan_in_categorical_gets_its_own_indicator_column():
    Xtr, Xte = align_and_featurize(
        pd.DataFrame({"a": [0.0, 1.0, 2.0], "cat": ["x", None, "y"]}),
        pd.DataFrame({"a": [5.0], "cat": ["x"]}),
        categorical_encoding="onehot",
    )
    # columns: a, cat_x, cat_y, cat_nan (the missing row -> its own indicator)
    assert list(Xtr.columns) == ["a", "cat_x", "cat_y", "cat_nan"]
    assert _rows(Xtr) == [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0, 0.0],
    ]
    assert _rows(Xte) == [[5.0, 1.0, 0.0, 0.0]]


def test_column_set_mismatch_raises():
    with pytest.raises(ValueError, match="same feature columns"):
        align_and_featurize(
            pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}),
            pd.DataFrame({"a": [2.0], "c": [3.0]}),
        )


def test_duplicate_column_names_raise():
    dup = pd.DataFrame(np.zeros((2, 2)))
    dup.columns = ["a", "a"]
    with pytest.raises(ValueError, match="duplicate column name"):
        align_and_featurize(dup, dup.iloc[:1].copy())


def test_non_positive_cardinality_cap_raises():
    with pytest.raises(ValueError, match="positive integer"):
        align_and_featurize(
            pd.DataFrame({"a": [0.0], "cat": ["x"]}),
            pd.DataFrame({"a": [1.0], "cat": ["x"]}),
            max_categorical_cardinality=0,
        )


def test_only_temporal_columns_raise_actionable_error():
    with pytest.raises(ValueError, match="unsupported dtype"):
        align_and_featurize(
            pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-01-02"])}),
            pd.DataFrame({"d": pd.to_datetime(["2024-01-03"])}),
        )
