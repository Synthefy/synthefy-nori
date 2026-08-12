"""Permanent client-only tests for canonical tabular preparation."""

import numpy as np
import pandas as pd
import pytest

from synthefy.featurize import align_and_featurize


def _rows(frame):
    return frame.to_numpy(dtype=float).tolist()


def test_canonical_ordinal_encoding_preserves_order_dtype_and_unseen_semantics():
    X_train, X_test = align_and_featurize(
        pd.DataFrame({"cat": ["y", "x"], "value": [1.0, 2.0]}),
        pd.DataFrame({"value": [3.0], "cat": ["new"]}),
    )

    assert list(X_train.columns) == list(X_test.columns) == ["cat", "value"]
    assert X_train["cat"].dtype == np.dtype("float64")
    assert _rows(X_train) == [[1.0, 1.0], [0.0, 2.0]]
    assert _rows(X_test) == [[-1.0, 3.0]]


def test_canonical_onehot_encoding_uses_train_layout():
    X_train, X_test = align_and_featurize(
        pd.DataFrame({"value": [1.0, 2.0], "cat": ["x", "y"]}),
        pd.DataFrame({"value": [3.0], "cat": ["new"]}),
        categorical_encoding="onehot",
    )

    assert list(X_train.columns) == list(X_test.columns) == [
        "value",
        "cat_x",
        "cat_y",
    ]
    assert _rows(X_train) == [[1.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
    assert _rows(X_test) == [[3.0, 0.0, 0.0]]


def test_canonical_validation_errors_remain_specific():
    with pytest.raises(ValueError, match="same feature columns"):
        align_and_featurize(
            pd.DataFrame({"a": [1.0]}),
            pd.DataFrame({"b": [1.0]}),
        )
    with pytest.raises(ValueError, match="categorical_encoding"):
        align_and_featurize(
            pd.DataFrame({"cat": ["x"]}),
            pd.DataFrame({"cat": ["x"]}),
            categorical_encoding="hashing",
        )


def test_canonical_non_dataframe_inputs_are_returned_unchanged():
    X_train = [[1.0, 2.0]]
    X_test = [[3.0, 4.0]]

    actual_train, actual_test = align_and_featurize(X_train, X_test)

    assert actual_train is X_train
    assert actual_test is X_test
