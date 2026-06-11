"""Tests for baseten/token_accounting.py::compute_tokens.

``baseten/`` is a deployment directory, not part of the installed
``synthefy_tabular`` package, so make the repo root importable here regardless
of how pytest is invoked. CI runs ``uv run pytest tests``, which only puts
``tests/`` on ``sys.path``; without this, ``import baseten`` fails.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baseten.token_accounting import compute_tokens


def test_dense_request_counts_every_known_value():
    # input = X_train cells (3x2) + y_train (3) + X_test cells (2x2)
    X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    y_train = [0.1, 0.2, 0.3]
    X_test = [[7.0, 8.0], [9.0, 10.0]]
    assert compute_tokens(X_train, y_train, X_test) == (6 + 3 + 4, 2)


def test_lists_and_numpy_inputs_agree():
    # compute_tokens is typed np.ndarray but must accept array-likes (lists)
    # too, exactly like SynthefyTabularRegressor.fit/predict; both must agree.
    X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    y_train = [0.1, 0.2, 0.3]
    X_test = [[7.0, 8.0], [9.0, 10.0]]
    from_lists = compute_tokens(X_train, y_train, X_test)
    from_numpy = compute_tokens(
        np.asarray(X_train), np.asarray(y_train), np.asarray(X_test)
    )
    assert from_lists == from_numpy == (13, 2)


def test_missing_test_cells_excluded_from_input_none_and_nan():
    # A null/NaN feature cell is imputed server-side, never sent and never
    # predicted, so it is billed neither as input nor output. None (from JSON
    # null) and np.nan must behave identically.
    X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # 6 known
    y_train = [0.1, 0.2, 0.3]  # 3 known
    X_test_none = [[7.0, None], [None, 8.0]]  # 2 known, 2 missing
    X_test_nan = np.array([[7.0, np.nan], [np.nan, 8.0]])  # 2 known, 2 missing
    assert compute_tokens(X_train, y_train, X_test_none) == (11, 2)
    assert compute_tokens(X_train, y_train, X_test_nan) == (11, 2)


def test_missing_values_in_every_block_are_excluded():
    X_train = [[1.0, np.nan], [3.0, 4.0]]  # 3 known, 1 missing
    y_train = [np.nan, 0.2]  # 1 known, 1 missing
    X_test = [[5.0, 6.0]]  # 2 known
    inp, out = compute_tokens(X_train, y_train, X_test)
    assert inp == 3 + 1 + 2
    assert out == 1


def test_output_tokens_count_rows_not_cells():
    # One predicted target per X_test row, independent of width.
    X_train = [[1.0, 2.0, 3.0]]
    y_train = [0.5]
    X_test = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]  # 3 rows, 9 cells
    inp, out = compute_tokens(X_train, y_train, X_test)
    assert out == 3
    assert inp == 3 + 1 + 9


def test_fully_missing_test_row_still_counts_as_one_output():
    # A query row with no known features still gets one prediction.
    X_train = [[1.0, 2.0]]
    y_train = [0.5]
    X_test = [[None, None]]
    inp, out = compute_tokens(X_train, y_train, X_test)
    assert out == 1
    assert inp == 2 + 1 + 0


def test_integer_inputs_are_counted():
    # Integer array-likes carry no NaN and are counted in full after coercion.
    inp, out = compute_tokens([[1, 2], [3, 4]], [0, 1], [[5, 6]])
    assert inp == 4 + 2 + 2
    assert out == 1


def test_returns_plain_python_ints():
    # Counts feed JSON/arithmetic downstream, so they must be builtin ints,
    # not numpy integer scalars.
    inp, out = compute_tokens(np.ones((2, 2)), np.ones(2), np.ones((1, 2)))
    assert type(inp) is int
    assert type(out) is int


@pytest.mark.parametrize(
    "n_train, n_features, n_test",
    [(1, 1, 1), (10, 5, 3), (100, 5, 8), (50, 20, 1)],
)
def test_dense_shape_formula(n_train, n_features, n_test):
    X_train = np.ones((n_train, n_features))
    y_train = np.ones(n_train)
    X_test = np.ones((n_test, n_features))
    inp, out = compute_tokens(X_train, y_train, X_test)
    assert inp == n_train * n_features + n_train + n_test * n_features
    assert out == n_test
