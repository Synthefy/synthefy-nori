"""Response-shape tests for ``baseten/model/model.py``::``Model.predict``.

These deliberately avoid the heavy inference stack: the regressor is faked and
``fastapi`` (supplied by the Baseten/Truss runtime, not a library dependency, so
absent from the test/library env) is stubbed. That leaves request handling +
response assembly under test — in particular the OpenAI-compatible ``usage``
block. ``baseten/`` is a deployment dir, not part of the installed package, so
the repo root is made importable regardless of how pytest is invoked.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def Model():
    # model.py does `from fastapi import HTTPException` at call time; the real
    # package isn't installed in the library/test env, so stand in a minimal stub.
    if "fastapi" not in sys.modules:
        fastapi = types.ModuleType("fastapi")

        class HTTPException(Exception):
            def __init__(self, status_code, detail=None):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        fastapi.HTTPException = HTTPException
        sys.modules["fastapi"] = fastapi

    from baseten.model.model import Model as _Model

    return _Model


class _FakeRegressor:
    """Stand-in for SynthefyTabularRegressor: one value per query row."""

    def fit(self, X, y):  # noqa: D401 - signature mirrors the real fit
        self._fitted = True

    def predict(self, X_test):
        return np.arange(np.asarray(X_test).shape[0], dtype=np.float64)


def _model_with_fake(Model):
    m = Model()
    m._regressor = _FakeRegressor()
    return m


def test_response_carries_openai_style_usage(Model):
    out = _model_with_fake(Model).predict(
        {
            "task": "regression",
            "X_train": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "y_train": [0.1, 0.2, 0.3],
            "X_test": [[7.0, 8.0], [9.0, 10.0]],
        }
    )
    assert out["task"] == "regression"
    assert out["predictions"] == [0.0, 1.0]  # one per X_test row, from the fake
    # input = 6 (X_train) + 3 (y_train) + 4 (X_test); output = 2 rows.
    assert out["usage"] == {
        "input_tokens": 13,
        "output_tokens": 2,
        "total_tokens": 15,
    }


def test_usage_excludes_null_cells(Model):
    # null cells (JSON null -> None -> NaN) are imputed server-side, never billed.
    out = _model_with_fake(Model).predict(
        {
            "X_train": [[1.0, None], [3.0, 4.0]],  # 3 known
            "y_train": [None, 0.2],  # 1 known
            "X_test": [[5.0, 6.0]],  # 2 known
        }
    )
    assert out["usage"] == {
        "input_tokens": 6,
        "output_tokens": 1,
        "total_tokens": 7,
    }
