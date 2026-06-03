"""Baseten custom model wrapper for the Synthefy Tabular foundation model.

Synthefy Tabular is an in-context learning model: there is no offline training
step. Every request supplies the context rows (``X_train``, ``y_train``) and the
query rows (``X_test``); the model conditions on the context and predicts labels
for the queries in a single forward pass.

Request body (POST /predict):

    {
        "X_train": [[...], ...],   # n_context x n_features
        "y_train": [...],          # n_context targets
        "X_test":  [[...], ...],   # n_query x n_features
        "task":    "regression"    # or "classification" (default: regression)
    }

Response:

    regression     -> {"task": "regression", "predictions": [...]}
    classification -> {"task": "classification",
                       "predictions": [...],        # predicted class labels
                       "probabilities": [[...], ...],  # per-class probabilities
                       "classes": [...]}            # class label order for the cols
"""

from __future__ import annotations

import os
import threading


def _to_jsonable(value):
    """Convert numpy arrays/scalars to plain JSON-serializable Python objects."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Model:
    def __init__(self, **kwargs):
        # Baseten injects the declared secrets here at construction time.
        self._secrets = kwargs.get("secrets") or {}
        self._token = None
        # Long-lived wrappers hold the loaded checkpoint weights in their cached
        # predictor, so weights are read from disk only once (during load()).
        self._regressor = None
        self._classifier = None
        # The underlying predictor's preprocessing uses per-call RNG/shuffling
        # state, so serialize inference to stay correct under concurrent requests.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ load
    def load(self):
        """Download the gated checkpoint and warm up both task heads once."""
        from synthefy_tabular import (
            SynthefyTabularClassifier,
            SynthefyTabularRegressor,
        )

        token = self._secrets.get("hf_access_token")
        if token:
            # Make the token visible to huggingface_hub for any indirect lookups.
            os.environ.setdefault("HF_TOKEN", token)
        self._token = token

        self._regressor = SynthefyTabularRegressor(token=token)
        self._classifier = SynthefyTabularClassifier(token=token)

        # Trigger checkpoint download + weight load now (not on first request) by
        # running a tiny in-context prediction through each head. The loaded
        # predictor is cached on the wrapper and reused for real requests.
        self._warmup()

    def _warmup(self):
        import numpy as np

        x = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]], dtype=np.float32)
        try:
            self._regressor.fit(x, np.array([0.0, 1.0, 0.5, 0.3], dtype=np.float64))
            self._regressor.predict(x[:1])
            self._classifier.fit(x, np.array([0, 1, 0, 1], dtype=np.int64))
            self._classifier.predict(x[:1])
        except Exception as exc:  # pragma: no cover - surfaced in deploy logs
            raise RuntimeError(
                "Failed to load the Synthefy Tabular checkpoint during warmup. "
                "If this is an auth error, ensure the 'hf_access_token' secret is "
                "set in your Baseten workspace and has read access to "
                "'Synthefy/synthefy-tabular'."
            ) from exc

    # --------------------------------------------------------------- predict
    def predict(self, model_input):
        import numpy as np

        task = str(model_input.get("task", "regression")).lower()

        for key in ("X_train", "y_train", "X_test"):
            if key not in model_input:
                return {"error": f"Missing required field '{key}'."}

        X_train = np.asarray(model_input["X_train"], dtype=np.float32)
        X_test = np.asarray(model_input["X_test"], dtype=np.float32)
        y_train = model_input["y_train"]

        if task in ("regression", "reg"):
            with self._lock:
                self._regressor.fit(X_train, np.asarray(y_train, dtype=np.float64))
                preds = self._regressor.predict(X_test)
            preds = np.atleast_1d(preds)
            return {"task": "regression", "predictions": _to_jsonable(preds)}

        if task in ("classification", "cls"):
            with self._lock:
                self._classifier.fit(X_train, np.asarray(y_train))
                proba = self._classifier.predict_proba(X_test)
                classes = self._classifier.classes_
            proba = np.atleast_2d(proba)
            preds = classes[proba.argmax(axis=1)]
            return {
                "task": "classification",
                "predictions": _to_jsonable(preds),
                "probabilities": _to_jsonable(proba),
                "classes": _to_jsonable(classes),
            }

        return {"error": f"Unsupported task: {task!r}. Use 'regression' or 'classification'."}
