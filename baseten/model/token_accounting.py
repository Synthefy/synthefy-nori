"""Token accounting for Synthefy Tabular inference requests.

The deployed model is an in-context regressor: each request supplies context
rows (``X_train``, ``y_train``) and query rows (``X_test``) and predicts one
target per query row (see ``baseten/model/model.py``). A ``null``/``NaN``
feature cell is imputed away during preprocessing and never emitted, so the
only values the model *fills* are the test targets — one per ``X_test`` row.

``input_tokens``  — every real (non-missing) value the request carries: the
    known cells of ``X_train``, the known entries of ``y_train``, and the known
    cells of ``X_test``. ``null``/``NaN`` cells are not values you sent, so they
    are not counted.
``output_tokens`` — one predicted target per ``X_test`` row (``len(X_test)``).
"""

from __future__ import annotations

import numpy as np


def compute_tokens(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> tuple[int, int]:
    """Compute ``(input_tokens, output_tokens)`` for one inference request.

    ``X_train``, ``y_train`` and ``X_test`` are typed ``np.ndarray`` to match
    ``SynthefyTabularPredictor.predict``; array-likes (lists) are also accepted
    and coerced with the same dtypes ``fit``/``predict`` use.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float32)

    # Input tokens: real (non-NaN) values across the context and query blocks.
    input_tokens = int(
        np.count_nonzero(~np.isnan(X_train))
        + np.count_nonzero(~np.isnan(y_train))
        + np.count_nonzero(~np.isnan(X_test))
    )
    # Output tokens: one predicted target per X_test row.
    output_tokens = int(X_test.shape[0])

    return input_tokens, output_tokens


def usage(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> dict:
    """Return an OpenAI-compatible ``usage`` block for one inference request.

    Wraps :func:`compute_tokens` into the shape clients expect on the response::

        {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...}

    ``total_tokens`` is exactly ``input_tokens + output_tokens``. Values are
    builtin ``int`` (not numpy scalars) so the dict is JSON-serializable as-is.
    """
    input_tokens, output_tokens = compute_tokens(X_train, y_train, X_test)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
