"""Stateful DataFrame preparation shared by every Nori public API.

Nori consumes numeric matrices.  This module owns the model-free contract that
turns a pandas DataFrame into one: columns are resolved as numeric, categorical,
or text during :meth:`DataFramePreprocessor.fit`, and :meth:`~DataFramePreprocessor.transform`
replays the fitted schema without learning anything from query rows.

The lightweight :mod:`synthefy` package deliberately keeps text dependencies
optional.  Categorical-only use imports neither scikit-learn nor a sentence
encoder; the text implementation is loaded only when ``text_columns`` is not
empty.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_MAX_CARDINALITY = 100
DEFAULT_CATEGORICAL_ENCODING = "ordinal"
CATEGORICAL_ENCODINGS = ("ordinal", "onehot")
CATEGORICAL_AUTO = "auto"


def _has_encodable_columns(frame: pd.DataFrame) -> bool:
    """Return whether a frame contains a column that is not plain numeric."""
    return any(not _is_numeric_series(frame[column]) for column in frame.columns)


def _is_numeric_series(series: pd.Series) -> bool:
    """Numeric magnitudes, excluding pandas' explicitly categorical dtype."""
    return not isinstance(series.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(series)


def _numeric_categories_to_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Legacy helper retained for import compatibility.

    The unified contract respects pandas ``CategoricalDtype`` as an explicit
    categorical signal, even when its levels happen to be numbers.  New code
    should use :class:`DataFramePreprocessor`; this helper preserves the older
    conversion behavior for callers that imported it privately.
    """
    out = frame
    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(
            series.cat.categories
        ):
            if out is frame:
                out = frame.copy()
            out[column] = series.astype("float64")
    return out


def _canonical_category(value: Any) -> str:
    """Return the stable category key used across pandas storage dtypes."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_columns(value: Any, *, name: str, allow_auto: bool = False) -> tuple[str | None, list[Any]]:
    """Normalize a column declaration and reject ambiguous/duplicate forms."""
    if value is None:
        return None, []
    if allow_auto and isinstance(value, str) and value == CATEGORICAL_AUTO:
        return CATEGORICAL_AUTO, []
    if isinstance(value, str):
        if allow_auto:
            raise ValueError(
                f"{name} must be 'auto', None, or a sequence of column names; "
                f"use {name}=[{value!r}] to name one categorical column."
            )
        columns = [value]
    else:
        try:
            columns = list(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be None or a sequence of column names; got {value!r}.") from exc
    duplicates = [column for column, count in Counter(columns).items() if count > 1]
    if duplicates:
        raise ValueError(f"{name} contains duplicate column name(s): {duplicates}.")
    return "explicit", columns


def _dtype_items(frame: pd.DataFrame, columns: list[Any]) -> str:
    return ", ".join(f"{column!r} ({frame[column].dtype})" for column in columns)


def _unsupported_temporal(series: pd.Series) -> bool:
    return (
        pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
        or isinstance(series.dtype, pd.PeriodDtype)
    )


class DataFramePreprocessor:
    """Fit-once DataFrame-to-numeric transform used by estimator, helper, and client.

    Parameters are stored verbatim so the object remains friendly to sklearn
    cloning when it is owned by :class:`synthefy_nori.NoriRegressor`.

    ``categorical_columns`` has three modes:

    * ``"auto"``: every remaining non-numeric, non-text column is categorical;
    * a sequence: exactly those columns are categorical;
    * ``None``: categorical inference is disabled.

    Automatically inferred columns above ``max_categorical_cardinality`` raise
    because they may be identifiers or free text.  An explicitly declared
    categorical keeps its most frequent K training levels and maps rarer and
    unseen values to one reserved ``other`` code.  Missing values remain NaN
    under ordinal encoding.
    """

    def __init__(
        self,
        *,
        categorical_columns: Any = CATEGORICAL_AUTO,
        text_columns: Any = None,
        max_categorical_cardinality: int = DEFAULT_MAX_CARDINALITY,
        categorical_encoding: str = DEFAULT_CATEGORICAL_ENCODING,
        svd_dim: int | None = 128,
        embedder: Any = "minilm",
        text_device: Any = None,
        text_normalize: bool | None = None,
    ) -> None:
        self.categorical_columns = categorical_columns
        self.text_columns = text_columns
        self.max_categorical_cardinality = max_categorical_cardinality
        self.categorical_encoding = categorical_encoding
        self.svd_dim = svd_dim
        self.embedder = embedder
        self.text_device = text_device
        self.text_normalize = text_normalize

    def _validate_parameters(self) -> None:
        if not isinstance(self.max_categorical_cardinality, int) or self.max_categorical_cardinality < 1:
            raise ValueError(
                "max_categorical_cardinality must be a positive integer; got "
                f"{self.max_categorical_cardinality!r}."
            )
        if self.categorical_encoding not in CATEGORICAL_ENCODINGS:
            raise ValueError(
                f"categorical_encoding must be one of {CATEGORICAL_ENCODINGS}; "
                f"got {self.categorical_encoding!r}."
            )

    @staticmethod
    def _validate_frame(frame: Any, *, name: str) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame; got {type(frame).__name__}.")
        columns = list(frame.columns)
        duplicates = list(pd.Index(columns)[pd.Index(columns).duplicated()].unique())
        if duplicates:
            raise ValueError(f"{name} has duplicate column name(s) {duplicates}; column names must be unique.")
        if not columns:
            raise ValueError(f"{name} must contain at least one feature column.")
        return frame

    def _fit_schema(self, frame: pd.DataFrame) -> None:
        self._validate_parameters()
        categorical_mode, declared_categorical = _normalize_columns(
            self.categorical_columns, name="categorical_columns", allow_auto=True
        )
        _, declared_text = _normalize_columns(self.text_columns, name="text_columns")

        columns = list(frame.columns)
        missing_categorical = [column for column in declared_categorical if column not in frame.columns]
        missing_text = [column for column in declared_text if column not in frame.columns]
        if missing_categorical:
            raise ValueError(f"categorical_columns not found in the DataFrame: {missing_categorical}.")
        if missing_text:
            raise ValueError(f"text_columns not found in the DataFrame: {missing_text}.")
        overlap = [column for column in declared_text if column in set(declared_categorical)]
        if overlap:
            raise ValueError(
                "categorical_columns and text_columns must not overlap; "
                f"both contain {overlap}."
            )

        text_set = set(declared_text)
        categorical_set = set(declared_categorical)
        numeric_columns: list[Any] = []
        categorical_columns: list[Any] = []
        ambiguous_columns: list[Any] = []
        temporal_columns: list[Any] = []

        for column in columns:
            if column in text_set:
                continue
            series = frame[column]
            if column in categorical_set:
                if _unsupported_temporal(series):
                    temporal_columns.append(column)
                else:
                    categorical_columns.append(column)
                continue
            if _is_numeric_series(series):
                numeric_columns.append(column)
            elif _unsupported_temporal(series):
                temporal_columns.append(column)
            elif categorical_mode == CATEGORICAL_AUTO:
                categorical_columns.append(column)
            else:
                ambiguous_columns.append(column)

        if temporal_columns:
            raise ValueError(
                "Feature column(s) have unsupported dtype (temporal values are not inferred): "
                + _dtype_items(frame, temporal_columns)
                + ". Convert datetimes/timedeltas/periods to numeric features or strings, "
                "then declare strings as categorical_columns=[...] or text_columns=[...]."
            )
        if ambiguous_columns:
            raise ValueError(
                "Non-numeric feature column(s) have no declared role: "
                + _dtype_items(frame, ambiguous_columns)
                + ". Choose categorical_columns=[...], text_columns=[...], convert them "
                "to numeric values, or remove them. categorical_levels= describes the target, not feature columns."
            )

        self.columns_ = columns
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.n_features_in_ = len(columns)
        self.categorical_mode_ = categorical_mode
        self.text_columns_ = declared_text
        self.numeric_columns_ = numeric_columns
        self.categorical_columns_ = categorical_columns
        self.category_maps_: dict[Any, dict[str, int]] = {}
        self.category_other_codes_: dict[Any, int] = {}
        self.category_has_missing_: dict[Any, bool] = {}

        explicitly_declared = set(declared_categorical)
        for column in categorical_columns:
            series = frame[column]
            keys = [_canonical_category(value) for value in series[series.notna()].tolist()]
            counts = Counter(keys)
            if len(counts) > self.max_categorical_cardinality and column not in explicitly_declared:
                raise ValueError(
                    f"Column {column!r} ({series.dtype}) has {len(counts)} distinct training values, "
                    f"above max_categorical_cardinality={self.max_categorical_cardinality}. "
                    f"Automatic handling is ambiguous: declare categorical_columns=[{column!r}] "
                    "to use top-K plus an 'other' value, declare it in text_columns=[...], "
                    "encode it explicitly, or remove it."
                )
            if len(counts) > self.max_categorical_cardinality:
                selected = sorted(counts, key=lambda key: (-counts[key], key))[
                    : self.max_categorical_cardinality
                ]
            else:
                selected = list(counts)
            selected = sorted(selected)
            mapping = {key: index for index, key in enumerate(selected)}
            self.category_maps_[column] = mapping
            self.category_other_codes_[column] = len(mapping)
            self.category_has_missing_[column] = bool(series.isna().any())

        self._text_preprocessor = None

    def _validate_transform_schema(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "columns_"):
            raise RuntimeError("Call fit()/fit_transform() before transform().")
        frame = self._validate_frame(frame, name="X")
        missing = [column for column in self.columns_ if column not in frame.columns]
        extra = [column for column in frame.columns if column not in set(self.columns_)]
        if missing or extra:
            raise ValueError(
                "X_train and X_test must have the same feature columns; query schema does not match "
                "the fitted training schema: "
                f"missing columns={missing}, extra columns={extra}."
            )
        frame = frame.loc[:, self.columns_]
        numeric_mismatches = [
            column for column in self.numeric_columns_ if not _is_numeric_series(frame[column])
        ]
        if numeric_mismatches:
            raise ValueError(
                "X_train and X_test must have matching column types. Query type mismatch for "
                "fitted numeric column(s): "
                + _dtype_items(frame, numeric_mismatches)
                + ". Convert them to numeric values before prediction."
            )
        temporal_categorical = [
            column for column in self.categorical_columns_ if _unsupported_temporal(frame[column])
        ]
        if temporal_categorical:
            raise ValueError(
                "Query categorical column(s) have unsupported temporal dtype: "
                + _dtype_items(frame, temporal_categorical)
                + ". Convert them to strings or numeric features."
            )
        return frame

    def _transform_tabular(self, frame: pd.DataFrame) -> tuple[list[np.ndarray], list[Any]]:
        blocks: list[np.ndarray] = []
        names: list[Any] = []
        numeric_set = set(self.numeric_columns_)
        categorical_set = set(self.categorical_columns_)

        for column in self.columns_:
            if column in numeric_set:
                values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float32).reshape(-1, 1)
                blocks.append(values)
                names.append(column)
            elif column in categorical_set:
                series = frame[column]
                mapping = self.category_maps_[column]
                other = self.category_other_codes_[column]
                missing = series.isna().to_numpy()
                codes = np.full(len(series), np.nan, dtype=np.float32)
                present_indices = np.flatnonzero(~missing)
                if len(present_indices):
                    present_keys = [_canonical_category(value) for value in series.iloc[present_indices].tolist()]
                    codes[present_indices] = np.asarray(
                        [mapping.get(key, other) for key in present_keys], dtype=np.float32
                    )
                if self.categorical_encoding == "ordinal":
                    blocks.append(codes.reshape(-1, 1))
                    names.append(column)
                else:
                    # Keep the released one-hot compatibility contract: categories
                    # are fitted on training rows, unseen/rare query values are an
                    # all-zero group, and a missing indicator exists only when
                    # missing was observed during fit.
                    has_missing = self.category_has_missing_[column]
                    width = len(mapping) + int(has_missing)
                    onehot = np.zeros((len(series), width), dtype=np.float32)
                    for row, code in enumerate(codes):
                        if np.isnan(code):
                            if has_missing:
                                onehot[row, len(mapping)] = 1.0
                        elif int(code) < len(mapping):
                            onehot[row, int(code)] = 1.0
                    ordered_keys = [key for key, _ in sorted(mapping.items(), key=lambda item: item[1])]
                    generated = [f"{column}_{key}" for key in ordered_keys]
                    if has_missing:
                        generated.append(f"{column}_nan")
                    blocks.append(onehot)
                    names.extend(generated)
        return blocks, names

    def _fit_transform_text(self, frame: pd.DataFrame) -> np.ndarray | None:
        if not self.text_columns_:
            return None
        from synthefy.text_features import MultimodalPreprocessor

        self._text_preprocessor = MultimodalPreprocessor(
            self.text_columns_,
            svd_dim=self.svd_dim,
            embedder=self.embedder,
            device=self.text_device,
            max_cardinality=self.max_categorical_cardinality,
            normalize=self.text_normalize,
        )
        return self._text_preprocessor.fit_transform(frame[self.text_columns_])

    def _transform_text(self, frame: pd.DataFrame) -> np.ndarray | None:
        if not self.text_columns_:
            return None
        return self._text_preprocessor.transform(frame[self.text_columns_])

    def _assemble(
        self,
        frame: pd.DataFrame,
        *,
        text_values: np.ndarray | None,
        fitting: bool,
    ) -> pd.DataFrame:
        blocks, names = self._transform_tabular(frame)
        if text_values is not None:
            blocks.append(np.asarray(text_values, dtype=np.float32))
            names.extend([f"text__{index}" for index in range(text_values.shape[1])])
        if not blocks:
            raise ValueError("No usable feature columns remain after DataFrame preprocessing.")
        duplicate_names = [name for name, count in Counter(names).items() if count > 1]
        if duplicate_names:
            raise ValueError(
                "Generated duplicate column names are not unique; rename the colliding input column(s): "
                f"{duplicate_names}."
            )
        values = np.hstack(blocks).astype(np.float32, copy=False)
        if fitting:
            self.feature_names_out_ = np.asarray(names, dtype=object)
            self.n_features_out_ = values.shape[1]
        elif names != list(self.feature_names_out_):
            raise RuntimeError("DataFrame preprocessing produced a different feature layout at transform time.")
        return pd.DataFrame(values, index=frame.index, columns=names)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate_frame(X, name="X")
        self._fit_schema(frame)
        text_values = self._fit_transform_text(frame)
        return self._assemble(frame, text_values=text_values, fitting=True)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DataFramePreprocessor":
        self.fit_transform(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate_transform_schema(X)
        text_values = self._transform_text(frame)
        return self._assemble(frame, text_values=text_values, fitting=False)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("Call fit()/fit_transform() before get_feature_names_out().")
        if input_features is not None and list(input_features) != self.columns_:
            raise ValueError(f"input_features must match fitted columns {self.columns_}; got {list(input_features)}.")
        return self.feature_names_out_.copy()


def align_and_featurize(
    X_train: Any,
    X_test: Any,
    max_categorical_cardinality: int = DEFAULT_MAX_CARDINALITY,
    categorical_encoding: str = DEFAULT_CATEGORICAL_ENCODING,
    *,
    categorical_columns: Any = CATEGORICAL_AUTO,
    text_columns: Any = None,
    svd_dim: int | None = 128,
    embedder: Any = "minilm",
    text_device: Any = None,
    text_normalize: bool | None = None,
    _warning_stacklevel: int = 4,
    _allow_empty: bool = False,
) -> Tuple[Any, Any]:
    """Fit on context rows and prepare context/query features identically.

    DataFrame inputs are aligned by the schema learned from ``X_train``.  Other
    inputs remain positional and are returned unchanged; if either side is a
    DataFrame, both must be DataFrames so column identity cannot be lost.
    """
    del _warning_stacklevel, _allow_empty
    parameter_validator = DataFramePreprocessor(
        max_categorical_cardinality=max_categorical_cardinality,
        categorical_encoding=categorical_encoding,
    )
    parameter_validator._validate_parameters()
    _, declared_text = _normalize_columns(text_columns, name="text_columns")
    categorical_mode, declared_categorical = _normalize_columns(
        categorical_columns, name="categorical_columns", allow_auto=True
    )
    normalized_categorical_columns: Any
    if categorical_mode == CATEGORICAL_AUTO:
        normalized_categorical_columns = CATEGORICAL_AUTO
    elif categorical_mode is None:
        normalized_categorical_columns = None
    else:
        normalized_categorical_columns = declared_categorical
    train_is_df = isinstance(X_train, pd.DataFrame)
    test_is_df = isinstance(X_test, pd.DataFrame)
    if train_is_df != test_is_df:
        frame = X_train if train_is_df else X_test
        if declared_text or declared_categorical or _has_encodable_columns(frame):
            raise ValueError(
                "One input is not a DataFrame: named categorical/text features require both "
                "X_train and X_test to be pandas DataFrames with the same columns."
            )
        return X_train, X_test
    if not train_is_df:
        if declared_text:
            raise ValueError("text_columns requires X_train and X_test to be pandas DataFrames.")
        if declared_categorical:
            raise ValueError("Explicit categorical_columns requires pandas DataFrame inputs with named columns.")
        return X_train, X_test

    preprocessor = DataFramePreprocessor(
        categorical_columns=normalized_categorical_columns,
        text_columns=declared_text,
        max_categorical_cardinality=max_categorical_cardinality,
        categorical_encoding=categorical_encoding,
        svd_dim=svd_dim,
        embedder=embedder,
        text_device=text_device,
        text_normalize=text_normalize,
    )
    return preprocessor.fit_transform(X_train), preprocessor.transform(X_test)


__all__ = [
    "CATEGORICAL_AUTO",
    "CATEGORICAL_ENCODINGS",
    "DEFAULT_CATEGORICAL_ENCODING",
    "DEFAULT_MAX_CARDINALITY",
    "DataFramePreprocessor",
    "align_and_featurize",
]
