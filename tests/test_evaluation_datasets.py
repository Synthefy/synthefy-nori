"""Fast unit tests for DatasetRegistry preprocessing (imputation semantics)."""

import numpy as np
import pandas as pd
import pytest

from synthefy_nori.evaluation.datasets import (
    DatasetRegistry,
    encode_categorical_column,
)


@pytest.fixture
def registry(tmp_path):
    return DatasetRegistry(cache_dir=str(tmp_path / "cache"))


def _make_frames():
    # Column "a": train median is 4.0 (NaN excluded); test has a NaN and a
    # deliberately different value distribution so train/test medians differ.
    X_train = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 4.0, 4.0, 4.0, 4.0, np.nan],
            "b": [np.nan] * 12,
            "c": np.arange(12, dtype=float),
        }
    )
    y_train = pd.Series(np.arange(12, dtype=float))
    X_test = pd.DataFrame(
        {
            "a": [100.0, np.nan, 100.0],
            "b": [np.nan, np.nan, np.nan],
            "c": [1.0, 2.0, 3.0],
        }
    )
    y_test = pd.Series([1.0, 2.0, 3.0])
    return X_train, y_train, X_test, y_test


def test_numeric_nan_filled_with_train_median(registry):
    X_train, y_train, X_test, y_test = _make_frames()
    entry = registry._make_entry_from_df(X_train, y_train, "toy", "unit", X_test=X_test, y_test=y_test)
    assert entry is not None
    # Train NaN in "a" (row 11) -> train median 4.0, not 0.
    assert entry.X_train[11, 0] == pytest.approx(4.0)


def test_test_nan_filled_with_train_median_not_test_median(registry):
    X_train, y_train, X_test, y_test = _make_frames()
    entry = registry._make_entry_from_df(X_train, y_train, "toy", "unit", X_test=X_test, y_test=y_test)
    # Test NaN in "a" (row 1) -> TRAIN median 4.0; the test median (100.0)
    # must not leak in, and the legacy zero-fill must not return.
    assert entry.X_test[1, 0] == pytest.approx(4.0)


def test_all_missing_column_filled_with_zero(registry):
    X_train, y_train, X_test, y_test = _make_frames()
    entry = registry._make_entry_from_df(X_train, y_train, "toy", "unit", X_test=X_test, y_test=y_test)
    # "b" has no observed train values -> median is NaN -> falls back to 0
    # in both frames.
    assert np.all(entry.X_train[:, 1] == 0.0)
    assert np.all(entry.X_test[:, 1] == 0.0)


def test_no_nan_survives_preprocessing(registry):
    X_train, y_train, X_test, y_test = _make_frames()
    entry = registry._make_entry_from_df(X_train, y_train, "toy", "unit", X_test=X_test, y_test=y_test)
    assert np.isfinite(entry.X_train).all()
    assert np.isfinite(entry.X_test).all()


def test_categorical_vocab_is_train_only_and_query_unknown_is_minus_one(registry):
    entry = registry._make_entry_from_df(
        pd.DataFrame({"cat": ["a", "c"] * 6}),
        pd.Series(np.arange(12, dtype=float)),
        "toy",
        "unit",
        X_test=pd.DataFrame({"cat": ["b", "c"]}),
        y_test=pd.Series([4.0, 5.0]),
    )

    np.testing.assert_array_equal(entry.X_train[:, 0], [0.0, 1.0] * 6)
    np.testing.assert_array_equal(entry.X_test[:, 0], [-1.0, 1.0])


def test_categorical_encoder_rejects_unknown_without_explicit_policy():
    _, classes = encode_categorical_column(pd.Series(["a", "c"]))

    with pytest.raises(ValueError, match="absent"):
        encode_categorical_column(pd.Series(["b"]), classes)
