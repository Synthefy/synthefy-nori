"""Cross-surface contract tests for DataFrame feature preparation."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import synthefy.text_features as text_features_module
from synthefy.featurize import DataFramePreprocessor
from synthefy.nori_client import _build_nori_request
from synthefy_nori import NoriRegressor
from synthefy_nori.api import predict as one_shot_predict


FIXTURE = Path(__file__).parent / "compat" / "categorical_features_v1.json"


def _fixture():
    return json.loads(FIXTURE.read_text())


def _expected(rows):
    return np.asarray([[np.nan if value is None else value for value in row] for row in rows], dtype=np.float32)


def test_frozen_fixture_matches_preprocessor_estimator_helper_and_client(monkeypatch):
    fixture = _fixture()
    train = pd.DataFrame(fixture["train"])
    query = pd.DataFrame(fixture["query"])
    kwargs = {
        "categorical_columns": fixture["categorical_columns"],
        "max_categorical_cardinality": fixture["max_categorical_cardinality"],
    }
    expected_train = _expected(fixture["expected_train"])
    expected_query = _expected(fixture["expected_query"])

    preprocessor = DataFramePreprocessor(**kwargs)
    np.testing.assert_equal(preprocessor.fit_transform(train).to_numpy(), expected_train)
    np.testing.assert_equal(preprocessor.transform(query).to_numpy(), expected_query)

    estimator = NoriRegressor(model_path="unused.pt", **kwargs).fit(train, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_equal(estimator.X_train_, expected_train)
    np.testing.assert_equal(estimator._prepare_query_features(query), expected_query)

    captured = {}

    def capture_predict(self, X, **predict_kwargs):
        captured["train"] = self.X_train_.copy()
        captured["query"] = self._prepare_query_features(X)
        return np.zeros(len(X), dtype=np.float32)

    monkeypatch.setattr(NoriRegressor, "predict", capture_predict)
    one_shot_predict(train, [1.0, 2.0, 3.0, 4.0], query, model_path="unused.pt", **kwargs)
    np.testing.assert_equal(captured["train"], expected_train)
    np.testing.assert_equal(captured["query"], expected_query)

    request = _build_nori_request(
        train,
        [1.0, 2.0, 3.0, 4.0],
        query,
        **kwargs,
    )
    np.testing.assert_equal(_expected(request.X_train), expected_train)
    np.testing.assert_equal(_expected(request.X_test), expected_query)


def test_documented_mixed_dataframe_example_runs_without_a_checkpoint(monkeypatch):
    def make_test_encoder(*args, **kwargs):
        del args, kwargs

        def encode(texts):
            return np.asarray(
                [
                    [
                        len(value),
                        value.count(" "),
                        sum(map(ord, value)) % 97,
                        sum((index + 1) * ord(char) for index, char in enumerate(value))
                        % 101,
                    ]
                    for value in texts
                ],
                dtype=np.float32,
            )

        return encode

    class PredictorStub:
        quantile_collapse = "mean"
        bar_point_estimator = "mean"

        def predict(self, X_train, y_train, X_test):
            assert X_train.shape[1] == X_test.shape[1]
            assert y_train.shape == (len(X_train),)
            return np.zeros(len(X_test), dtype=np.float32)

    monkeypatch.setattr(text_features_module, "_make_encoder", make_test_encoder)
    monkeypatch.setattr(
        NoriRegressor, "_get_predictor", lambda self: PredictorStub()
    )

    X_train = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0, 40.0],
            "plan": ["free", "pro", "free", "team"],
            "region": ["us", "eu", "us", "apac"],
            "ticket_description": [
                "cannot sign in",
                "invoice question",
                "password reset",
                "add another user",
            ],
        }
    )
    y_train = [0.0, 1.0, 0.0, 1.0]
    X_test = pd.DataFrame(
        {
            "amount": [15.0, 50.0],
            "plan": ["free", "enterprise"],
            "region": ["eu", "us"],
            "ticket_description": ["login help", "security review"],
        }
    )

    reg = NoriRegressor(
        model="nori-30m",
        categorical_columns=["plan", "region"],
        text_columns=["ticket_description"],
    )
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    assert y_pred.tolist() == [0.5, 0.5]
    assert reg._feature_preprocessor.categorical_columns_ == ["plan", "region"]
    assert reg._feature_preprocessor.text_columns_ == ["ticket_description"]


def test_direct_estimator_auto_encodes_raw_strings_without_text_runtime(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_text_import(name, *args, **kwargs):
        if name == "synthefy.text_features":
            raise AssertionError("ordinary categoricals must not import the text runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_text_import)
    frame = pd.DataFrame({"amount": [1.0, 2.0], "plan": ["free", "pro"]})
    estimator = NoriRegressor(model_path="unused.pt").fit(frame, [1.0, 2.0])
    assert estimator.X_train_.tolist() == [[1.0, 0.0], [2.0, 1.0]]


def test_estimator_strict_modes_and_query_schema_errors_are_actionable():
    frame = pd.DataFrame({"amount": [1.0, 2.0], "plan": ["free", "pro"], "note": ["a", "b"]})
    with pytest.raises(ValueError) as caught:
        NoriRegressor(
            model_path="unused.pt", categorical_columns=["plan"]
        ).fit(frame, [1.0, 2.0])
    message = str(caught.value)
    assert "'note' (" in message
    assert "text_columns" in message

    estimator = NoriRegressor(model_path="unused.pt").fit(frame, [1.0, 2.0])
    with pytest.raises(ValueError) as caught:
        estimator._prepare_query_features(pd.DataFrame({"amount": [3.0], "plan": ["free"], "extra": [1.0]}))
    assert "missing columns=['note']" in str(caught.value)
    assert "extra columns=['extra']" in str(caught.value)


def test_estimator_feature_configuration_clones_and_pickles():
    estimator = NoriRegressor(
        model_path="unused.pt",
        categorical_columns=["plan"],
        categorical_encoding="onehot",
        max_categorical_cardinality=8,
        text_columns=[],
    )
    cloned = clone(estimator)
    assert cloned.categorical_columns == ["plan"]
    assert cloned.categorical_encoding == "onehot"
    fitted = estimator.fit(pd.DataFrame({"plan": ["a", "b"]}), [1.0, 2.0])
    restored = pickle.loads(pickle.dumps(fitted))
    assert list(restored.feature_names_in_) == ["plan"]
    assert restored._prepare_query_features(pd.DataFrame({"plan": ["new"]})).shape == (1, 2)


def test_dataframe_feature_configuration_survives_grid_search(monkeypatch):
    from sklearn.model_selection import GridSearchCV

    def predict_without_loading_model(self, X, **kwargs):
        del kwargs
        query = self._prepare_query_features(X)
        return np.full(len(query), self.y_mean_, dtype=np.float64)

    monkeypatch.setattr(NoriRegressor, "predict", predict_without_loading_model)
    X = pd.DataFrame(
        {
            "amount": np.arange(8, dtype=np.float32),
            "plan": ["free", "pro"] * 4,
        }
    )
    search = GridSearchCV(
        NoriRegressor(model_path="unused.pt", categorical_columns=["plan"]),
        {"categorical_encoding": ["ordinal", "onehot"]},
        cv=2,
        error_score="raise",
    )

    search.fit(X, np.arange(8, dtype=np.float64))

    assert search.best_estimator_._feature_preprocessor.categorical_columns_ == [
        "plan"
    ]


def test_deprecated_text_cardinality_alias_is_explicit_and_checked():
    frame = pd.DataFrame({"plan": ["a", "b", "c"]})
    with pytest.deprecated_call(match="max_categorical_cardinality"):
        estimator = NoriRegressor(
            model_path="unused.pt",
            categorical_columns=["plan"],
            text_max_cardinality=2,
        ).fit(frame, [1.0, 2.0, 3.0])
    assert len(estimator._feature_preprocessor.category_maps_["plan"]) == 2

    with pytest.raises(ValueError, match="disagree"):
        NoriRegressor(
            model_path="unused.pt",
            categorical_columns=["plan"],
            max_categorical_cardinality=3,
            text_max_cardinality=2,
        ).fit(frame, [1.0, 2.0, 3.0])


def test_positional_raw_strings_report_offending_indices():
    with pytest.raises(ValueError) as caught:
        NoriRegressor(model_path="unused.pt").fit(
            np.asarray([[1.0, "free"], [2.0, "pro"]], dtype=object),
            [1.0, 2.0],
        )
    assert "non-numeric column indices=[1]" in str(caught.value)
    assert "pandas DataFrame" in str(caught.value)
