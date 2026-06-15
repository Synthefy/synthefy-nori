"""Baseten custom model wrapper for the Nori foundation model.

Nori is an in-context learning model: there is no offline training
step. Every request supplies the context rows (``X_train``, ``y_train``) and the
query rows (``X_test``); the model conditions on the context and predicts targets
for the queries in a single forward pass.

This release is **regression-only**.

Request body (POST /predict):

    {
        "X_train": [[...], ...],   # n_context x n_features
        "y_train": [...],          # n_context targets
        "X_test":  [[...], ...],   # n_query x n_features
        "task":    "regression"    # optional; "regression"/"reg" only (default: regression)
    }

Response:

    {
        "task": "regression",
        "predictions": [...],   # one value per X_test row
        "usage": {              # OpenAI-compatible token accounting
            "input_tokens": ...,   # real (non-null) values sent across the inputs
            "output_tokens": ...,  # one predicted target per X_test row
            "total_tokens": ...,   # input_tokens + output_tokens
        },
    }
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


def _load_token_accounting():
    """Import the sibling ``token_accounting`` module.

    ``token_accounting.py`` lives next to this file in ``model/`` so Truss
    bundles it into the image — only ``model/``, ``packages/`` and ``data/`` are
    shipped, so a module left at the Truss root is silently dropped (which
    previously made every ``predict()`` 500 with ModuleNotFoundError). Add this
    file's own directory to ``sys.path`` and import it directly. Works the same
    in-repo and on the container. Deferred to call time to keep ``model.py``
    import-light (the module pulls in numpy).
    """
    import sys
    from pathlib import Path

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import token_accounting

    return token_accounting


class Model:
    def __init__(self, **kwargs):
        # Baseten injects the declared secrets here at construction time.
        self._secrets = kwargs.get("secrets") or {}
        self._token = None
        # The long-lived wrapper holds the loaded checkpoint weights in its cached
        # predictor, so weights are read from disk only once (during load()).
        self._regressor = None
        # The underlying predictor's preprocessing uses per-call RNG/shuffling
        # state, so serialize inference to stay correct under concurrent requests.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ load
    def load(self):
        """Download the checkpoint and warm up the regression head once."""
        from synthefy_nori import NoriRegressor

        # The default checkpoint is public; a token is only needed if the repo is
        # gated. Honor an optional hf_access_token secret when present.
        token = self._secrets.get("hf_access_token")
        if token:
            os.environ.setdefault("HF_TOKEN", token)
        self._token = token

        self._regressor = NoriRegressor(token=token)

        # Trigger checkpoint download + weight load now (not on first request) by
        # running a tiny in-context prediction. The loaded predictor is cached on
        # the wrapper and reused for real requests.
        self._warmup()

    def _warmup(self):
        import numpy as np

        x = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]], dtype=np.float32)
        try:
            self._regressor.fit(x, np.array([0.0, 1.0, 0.5, 0.3], dtype=np.float64))
            self._regressor.predict(x[:1])
        except Exception as exc:  # pragma: no cover - surfaced in deploy logs
            raise RuntimeError(
                "Failed to load the Nori checkpoint during warmup. "
                "If this is an auth error, set the 'hf_access_token' secret in your "
                "Baseten workspace to a Hugging Face token with read access to "
                "'Synthefy/synthefy-nori'."
            ) from exc

    # --------------------------------------------------------------- predict
    def predict(self, model_input):
        import numpy as np
        from fastapi import HTTPException

        task = str(model_input.get("task", "regression")).lower()

        for key in ("X_train", "y_train", "X_test"):
            if key not in model_input:
                raise HTTPException(
                    status_code=400, detail=f"Missing required field '{key}'."
                )

        if task not in ("regression", "reg"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported task: {task!r}. This deployment is "
                    "regression-only; use 'regression'."
                ),
            )

        X_train = np.asarray(model_input["X_train"], dtype=np.float32)
        X_test = np.asarray(model_input["X_test"], dtype=np.float32)
        y_train = np.asarray(model_input["y_train"], dtype=np.float64)

        with self._lock:
            self._regressor.fit(X_train, y_train)
            preds = self._regressor.predict(X_test)
        preds = np.atleast_1d(preds)

        # Token accounting is pure CPU counting and independent of the model, so
        # it runs outside the inference lock.
        usage = _load_token_accounting().usage(X_train, y_train, X_test)
        return {
            "task": "regression",
            "predictions": _to_jsonable(preds),
            "usage": usage,
        }
