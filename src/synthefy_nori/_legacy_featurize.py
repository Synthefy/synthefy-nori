"""Private compatibility featurizer for clients without synthefy.featurize.

The public ``infer`` / ``predict`` helpers (see :mod:`synthefy_nori.api`) accept
Python lists, numpy arrays, or pandas DataFrames. When **both** ``X_train`` and
``X_test`` are DataFrames, this module turns any non-numeric columns into a fully
numeric, model-ready matrix client-side — fitting the encoding on ``X_train`` and
applying the same column layout to ``X_test`` — so raw categorical frames "just
work" without relying on the model to detect categories itself. The default
encoding is **ordinal** (one integer-code column per categorical, matching the
model's own server-side ``OrdinalEncoder`` path); pass
``categorical_encoding="onehot"`` for indicator columns instead. Numeric inputs
(lists / numpy / all-numeric DataFrames) are byte-for-byte unchanged apart from a
by-name column reorder of ``X_test`` to match ``X_train``.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

# Default cap on a categorical column's distinct values before encoding.
# Columns above this are dropped (with a warning): they are almost always
# identifiers, and under one-hot they also explode the feature matrix — matches
# the offline evaluator, which drops string columns with more than 100 unique
# values.
DEFAULT_MAX_CARDINALITY = 100

# How non-numeric columns are converted for the model. "ordinal" mirrors the
# model's own server-side OrdinalEncoder path and benchmarked at least as well
# as one-hot across 35 categorical datasets while never widening the matrix (and
# never OOM-ing on wide tables); "onehot" preserves the original behavior.
DEFAULT_CATEGORICAL_ENCODING = "ordinal"
CATEGORICAL_ENCODINGS = ("ordinal", "onehot")


def _has_encodable_columns(frame: pd.DataFrame) -> bool:
    """``True`` if any column is non-numeric (so featurization is needed)."""
    return any(not pd.api.types.is_numeric_dtype(frame[col]) for col in frame.columns)


def _numeric_categories_to_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert ``category`` columns whose categories are numeric back to a plain
    numeric dtype, so they are kept as magnitudes rather than one-hot exploded
    (``is_numeric_dtype`` is ``False`` for any ``category`` dtype, even integer
    ones). Returns ``frame`` unchanged — no copy — when there is nothing to
    convert.
    """
    out = frame
    for col in frame.columns:
        s = frame[col]
        if isinstance(s.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(
            s.cat.categories
        ):
            if out is frame:
                out = frame.copy()
            # cast to float (not the categories' dtype) so a missing value in an
            # *integer*-category column promotes to NaN instead of raising
            # "Cannot convert NaN to integer".
            out[col] = s.astype("float64")
    return out


def _featurize_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_cardinality: int,
    encoding: str = DEFAULT_CATEGORICAL_ENCODING,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode non-numeric columns of two aligned frames (fit on train).

    Numeric columns (including ``bool``, and ``category`` columns whose
    categories are numeric) pass through unchanged. Datetime columns, columns
    with no non-missing values, and categorical columns with more than
    ``max_cardinality`` distinct *training* values are dropped with a
    ``UserWarning``; ``timedelta`` columns are unsupported and raise.

    With ``encoding="ordinal"`` (default) each categorical column stays a single
    column of integer codes assigned in sorted-category order — the same
    convention as the model's own server-side ``OrdinalEncoder`` path.
    Categories come from ``X_train`` (compared as strings): a value seen only in
    ``X_test`` maps to ``-1`` and a missing value (in either frame) maps to
    ``NaN``, mirroring the server's ``unknown_value``/``encoded_missing_value``.
    Column names and order are preserved.

    With ``encoding="onehot"`` categories come from ``X_train``: a value seen
    only in ``X_test`` maps to an all-zeros indicator group, a category absent
    from ``X_test`` is still emitted as a zero column, and a missing value (in
    either frame) gets its own indicator column (``dummy_na=True``, but the
    indicator is dropped when no row is missing) — so both frames come out with
    identical numeric columns.

    Either way the model receives a fully model-ready matrix (no reliance on
    model-side category detection). A column that is numeric in one frame but
    not the other raises ``ValueError`` (rather than failing later with a
    confusing message). ``X_train`` and ``X_test`` must already share the same
    columns (callers align them by name first). Row order and count are
    preserved.
    """
    # category-of-numeric -> plain numeric, so it is kept as a magnitude (not
    # one-hot exploded). Applied to both frames before any dtype inspection.
    X_train = _numeric_categories_to_values(X_train)
    X_test = _numeric_categories_to_values(X_test)

    # A column must be the same kind (numeric vs not) in both frames; otherwise
    # featurization would silently mis-handle it (or crash later in the float
    # cast with a misleading message). Fail loud and specific instead.
    mismatched = [
        col
        for col in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[col])
        != pd.api.types.is_numeric_dtype(X_test[col])
    ]
    if mismatched:
        raise ValueError(
            f"Column(s) {mismatched} are numeric in one of X_train/X_test but "
            "not the other; X_train and X_test must have matching column types "
            "(a common cause is object-dtype numbers, e.g. from read_csv — cast "
            "them with pd.to_numeric first)."
        )

    numeric_cols: List[Any] = []
    cat_cols: List[Any] = []
    dropped: List[str] = []
    for col in X_train.columns:
        s = X_train[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            dropped.append(f"{col!r} (datetime)")
        elif pd.api.types.is_timedelta64_dtype(s) or isinstance(s.dtype, pd.PeriodDtype):
            raise ValueError(
                f"Column {col!r} has unsupported dtype {s.dtype}; convert it to a "
                "number (e.g. .dt.total_seconds()) or a string before calling "
                "predict()."
            )
        else:
            n_unique = s.nunique(dropna=True)
            if n_unique == 0:
                dropped.append(f"{col!r} (no non-missing values)")
            elif n_unique > max_cardinality:
                dropped.append(f"{col!r} (>{max_cardinality} unique values)")
            else:
                cat_cols.append(col)

    if dropped:
        warnings.warn(
            "Nori featurization dropped non-encodable column(s): "
            + ", ".join(dropped)
            + ". Encode them yourself (e.g. target/hash encoding) if you need them.",
            stacklevel=4,
        )

    if cat_cols and encoding == "ordinal":
        # Single numeric column per categorical, codes in sorted-category order
        # (the server's own OrdinalEncoder convention): train categories only,
        # unseen test value -> -1, missing -> NaN. Original column order kept.
        keep = set(numeric_cols) | set(cat_cols)
        kept_cols = [c for c in X_train.columns if c in keep]
        X_train_feat = X_train[kept_cols].reset_index(drop=True).copy()
        X_test_feat = X_test[kept_cols].reset_index(drop=True).copy()
        for col in cat_cols:
            cats = np.unique(X_train[col].dropna().astype(str).to_numpy())
            mapping = {c: i for i, c in enumerate(cats)}
            for feat, src in ((X_train_feat, X_train), (X_test_feat, X_test)):
                s = src[col].reset_index(drop=True)
                codes = np.full(len(s), np.nan, dtype=np.float64)
                notna = s.notna()
                present = s[notna].astype(str).map(mapping)
                codes[notna.to_numpy()] = present.fillna(-1).to_numpy(dtype=np.float64)
                feat[col] = codes
    elif cat_cols:
        # dummy_na=True gives missing values their own indicator column, so NaN is
        # a distinct category rather than silently all-zeros. get_dummies always
        # emits a NaN column per categorical; drop columns that are all-zero in
        # TRAIN — that removes the dead NaN-indicator when a column has no missing
        # rows (so a legitimate literal "nan" value column doesn't collide with
        # it). Done positionally so a transient duplicate label can't break it.
        train_d = pd.get_dummies(
            X_train[cat_cols].astype(object),
            columns=cat_cols,
            dummy_na=True,
            dtype=np.uint8,
        )
        train_d = train_d.loc[:, (train_d.to_numpy() != 0).any(axis=0)]
        if train_d.columns.has_duplicates:
            raise ValueError(
                "One-hot encoding produced duplicate column names — a column name "
                "and value collide under '<column>_<value>' naming. Rename the "
                "offending column(s) before calling predict()."
            )
        test_d = pd.get_dummies(
            X_test[cat_cols].astype(object),
            columns=cat_cols,
            dummy_na=True,
            dtype=np.uint8,
        )
        # Drop test all-zero columns too (e.g. the dummy_na column when X_test has
        # no missing rows) so test_d has no duplicate label before reindex.
        test_d = test_d.loc[:, (test_d.to_numpy() != 0).any(axis=0)]
        test_d = test_d.reindex(columns=train_d.columns, fill_value=0)
        X_train_feat = pd.concat(
            [X_train[numeric_cols].reset_index(drop=True), train_d.reset_index(drop=True)],
            axis=1,
        )
        X_test_feat = pd.concat(
            [X_test[numeric_cols].reset_index(drop=True), test_d.reset_index(drop=True)],
            axis=1,
        )
        if X_train_feat.columns.has_duplicates:
            raise ValueError(
                "Featurized columns are not unique — a numeric column name "
                "collides with a generated one-hot column name. Rename the "
                "offending column(s) before calling predict()."
            )
    else:
        X_train_feat = X_train[numeric_cols].reset_index(drop=True)
        X_test_feat = X_test[numeric_cols].reset_index(drop=True)

    if X_train_feat.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after featurization (every "
            "column was dropped — temporal, all-missing, or above the "
            f"max_categorical_cardinality={max_cardinality} cap)."
        )
    return X_train_feat, X_test_feat


def align_and_featurize(
    X_train: Any,
    X_test: Any,
    max_categorical_cardinality: int = DEFAULT_MAX_CARDINALITY,
    categorical_encoding: str = DEFAULT_CATEGORICAL_ENCODING,
) -> Tuple[Any, Any]:
    """Align two inputs by column name and encode non-numeric columns.

    When both ``X_train`` and ``X_test`` are pandas DataFrames, ``X_test`` is
    aligned to ``X_train``'s columns *by name* (so column order is irrelevant; a
    mismatch in the column sets raises ``ValueError``), then any non-numeric
    columns are encoded — fit on ``X_train`` and applied to ``X_test`` — so the
    resulting matrices are fully numeric. ``categorical_encoding`` selects
    ordinal codes (default) or one-hot indicators; see :func:`_featurize_frames`.
    Otherwise the inputs are returned unchanged (positional matching, as before)
    — but a DataFrame with non-numeric columns paired with a non-DataFrame
    raises, since alignment needs column names on both sides.

    Returns possibly-transformed ``(X_train, X_test)`` ready to be fed to the
    model. NaN/missing values in numeric columns are preserved for server-side
    imputation.
    """
    if max_categorical_cardinality < 1:
        raise ValueError(
            "max_categorical_cardinality must be a positive integer; got "
            f"{max_categorical_cardinality}."
        )
    if categorical_encoding not in CATEGORICAL_ENCODINGS:
        raise ValueError(
            f"categorical_encoding must be one of {CATEGORICAL_ENCODINGS}; "
            f"got {categorical_encoding!r}."
        )

    train_is_df = isinstance(X_train, pd.DataFrame)
    test_is_df = isinstance(X_test, pd.DataFrame)

    if train_is_df != test_is_df:
        # one side is a DataFrame, the other isn't: we can't one-hot/align by
        # name. Give a targeted error if the DataFrame side has columns to encode.
        df, df_name, other = (
            (X_train, "X_train", "X_test")
            if train_is_df
            else (X_test, "X_test", "X_train")
        )
        if _has_encodable_columns(df):
            raise ValueError(
                f"{df_name} has non-numeric column(s) to encode, but "
                f"{other} is not a DataFrame; pass both X_train and X_test as "
                "DataFrames with the same columns so they can be aligned and "
                "encoded (or pre-encode to numeric)."
            )
        return X_train, X_test

    if not (train_is_df and test_is_df):
        return X_train, X_test

    train_cols = list(X_train.columns)
    test_cols = list(X_test.columns)
    for cols, nm in ((train_cols, "X_train"), (test_cols, "X_test")):
        idx = pd.Index(cols)
        if idx.has_duplicates:
            dups = sorted({str(c) for c in idx[idx.duplicated()]})
            raise ValueError(
                f"{nm} has duplicate column name(s) {dups}; column names must be "
                "unique (duplicates break by-name alignment and encoding)."
            )
    if set(train_cols) != set(test_cols):
        raise ValueError(
            "X_train and X_test must have the same feature columns; "
            f"X_train has {train_cols} but X_test has {test_cols}."
        )
    if train_cols != test_cols:
        # Same columns, different order: reorder X_test to match X_train so the
        # model sees features in a consistent position.
        X_test = X_test[train_cols]

    # Encode any non-numeric columns into a fully numeric matrix, fitting on
    # X_train and applying the same layout to X_test. Check both frames so a
    # column non-numeric in only one of them is caught with a clear error rather
    # than a later cryptic float-cast.
    if (
        len(X_train)
        and len(X_test)
        and (_has_encodable_columns(X_train) or _has_encodable_columns(X_test))
    ):
        X_train, X_test = _featurize_frames(
            X_train, X_test, max_categorical_cardinality, encoding=categorical_encoding
        )

    return X_train, X_test
