"""Compatibility tests for the released fingerprint preprocessing contract."""

from __future__ import annotations

import numpy as np

from synthefy_nori.inference.preprocess import FingerprintFeatureEncoder, float_hash_arr


def _fit_encoder(x: np.ndarray) -> FingerprintFeatureEncoder:
    encoder = FingerprintFeatureEncoder()
    encoder.fit(x, categorical_features=[1], seed=42)
    return encoder


def test_context_and_query_use_legacy_single_and_double_salt_namespaces():
    x = np.array([[1.0, 2.0], [3.5, -4.0]], dtype=np.float32)
    encoder = _fit_encoder(x)

    context, context_categorical = encoder.transform(x, is_test=False)
    query, query_categorical = encoder.transform(x.copy(), is_test=True)
    expected_context = np.asarray(
        [float_hash_arr(row + encoder.salt_value) for row in x],
        dtype=x.dtype,
    )
    expected_query = np.asarray(
        [float_hash_arr(row + 2 * encoder.salt_value) for row in x],
        dtype=x.dtype,
    )

    np.testing.assert_array_equal(context[:, :-1], x)
    np.testing.assert_array_equal(query[:, :-1], x)
    np.testing.assert_array_equal(context[:, -1], expected_context)
    np.testing.assert_array_equal(query[:, -1], expected_query)
    assert np.all(context[:, -1] != query[:, -1])
    assert context_categorical == query_categorical == [1]


def test_duplicate_context_rows_are_rehashed_but_query_duplicates_are_not():
    row = np.array([[1.0, 2.0]], dtype=np.float32)
    rows = np.repeat(row, repeats=2, axis=0)
    encoder = _fit_encoder(rows)

    context, _ = encoder.transform(rows, is_test=False)
    query, _ = encoder.transform(rows, is_test=True)
    expected_first_context = float_hash_arr(row[0] + encoder.salt_value)
    expected_second_context = float_hash_arr(row[0] + encoder.salt_value + 1)
    expected_query = float_hash_arr(row[0] + 2 * encoder.salt_value)

    np.testing.assert_array_equal(
        context[:, -1],
        np.asarray([expected_first_context, expected_second_context], dtype=row.dtype),
    )
    np.testing.assert_array_equal(
        query[:, -1],
        np.asarray([expected_query, expected_query], dtype=row.dtype),
    )


def test_unseen_query_uses_legacy_query_namespace():
    context_rows = np.array([[1.0, 2.0]], dtype=np.float32)
    query_rows = np.array([[3.0, 4.0]], dtype=np.float32)
    encoder = _fit_encoder(context_rows)

    query, _ = encoder.transform(query_rows, is_test=True)
    expected = float_hash_arr(query_rows[0] + 2 * encoder.salt_value)

    np.testing.assert_array_equal(query[:, -1], np.asarray([expected], dtype=query_rows.dtype))
