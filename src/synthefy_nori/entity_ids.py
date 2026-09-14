"""Context-fitted encoding of explicitly named identifier columns."""

import numpy as np
import pandas as pd


class EntityIDPreprocessor:
    """Encode each declared ID in place; missing/unseen IDs become -1."""

    def __init__(self, encode_id_cols):
        if not isinstance(encode_id_cols, list) or any(not isinstance(c, str) or not c for c in encode_id_cols):
            raise ValueError("encode_id_cols must be a list of nonempty column names")
        if len(set(encode_id_cols)) != len(encode_id_cols):
            raise ValueError("encode_id_cols must not contain duplicate names")
        self.encode_id_cols = list(encode_id_cols)

    @staticmethod
    def _validate(X, columns):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("encode_id_cols requires a DataFrame with unique column names")
        for column in columns:
            if column not in X:
                raise ValueError(f"ID column not found: {column}")
            values = X[column].dropna()
            valid = (
                values.empty
                or all(isinstance(v, str) for v in values)
                or all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values)
            )
            if not valid:
                raise ValueError(f"ID column {column!r} must contain integers or strings, plus missing values")

    def fit(self, X):
        self._validate(X, self.encode_id_cols)
        self.values_ = {column: pd.Index(X[column].dropna().unique()) for column in self.encode_id_cols}
        return self

    def transform(self, X):
        self._validate(X, self.encode_id_cols)
        result = X.copy()
        for column, values in self.values_.items():
            codes = values.get_indexer(X[column])
            codes[X[column].isna().to_numpy()] = -1
            result[column] = codes
        return result
