"""shapiq adapters for Nori — Shapley values and Shapley interactions.

Nori is regression-only and follows the sklearn estimator API, so the recommended
explainer is the model-agnostic, imputation-based ``shapiq.TabularExplainer``: the
training context is fixed and query features are removed by imputation, so each
coalition is a single ``predict`` call (cost controlled by ``budget``).
"""

from __future__ import annotations

from typing import Any


def _require_shapiq():
    try:
        import shapiq  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'shapiq is required for interpretability. Install with: pip install "synthefy-nori[interpretability]"'
        ) from exc
    return shapiq


def _get_predict_function(model, data):
    """Return a callable that predicts an array of rows and returns a 1-D float array.

    Passing an explicit prediction function makes the explainer independent of
    shapiq's per-version model auto-detection, and pins the regression point
    estimate to ``predict``'s default (the distribution mean). shapiq passes rows
    as arrays; a regressor fitted on a DataFrame (it records ``feature_names_in_``)
    refuses arrays, so each batch becomes a DataFrame with the fitted column names,
    and with the dtypes of ``data`` when ``data`` is a DataFrame of those columns.
    """
    import numpy as np
    import pandas as pd

    if not hasattr(model, "predict"):
        raise TypeError("model must be a fitted NoriRegressor (or expose .predict)")
    fitted_columns = getattr(model, "feature_names_in_", None)
    if fitted_columns is None:

        def predict(x):
            return np.asarray(model.predict(np.asarray(x)), dtype=np.float64).reshape(-1)

        return predict
    columns = [str(column) for column in fitted_columns]
    dtypes = (
        data.dtypes.to_dict() if isinstance(data, pd.DataFrame) and [str(c) for c in data.columns] == columns else None
    )

    def predict_frame(x):
        frame = pd.DataFrame(np.asarray(x, dtype=object).reshape(-1, len(columns)), columns=columns)
        frame = frame.astype(dtypes) if dtypes is not None else frame.infer_objects()
        return np.asarray(model.predict(frame), dtype=np.float64).reshape(-1)

    return predict_frame


def _get_background_array(data):
    """Return the background rows as the array shapiq imputes from.

    All-numeric data stays a float64 array, as before. Data with a text or other
    non-numeric column becomes an object array, so its values survive and the
    baseline imputer can take their most common value.
    """
    import numpy as np
    import pandas as pd

    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(np.asarray(data))
    numeric = all(
        pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
        for dtype in frame.infer_objects().dtypes
    )
    return frame.to_numpy(dtype=np.float64) if numeric else frame.to_numpy(dtype=object)


def _get_categorical_feature_indices(model, data):
    """Return the positions of the features the imputer should treat as categorical.

    These are the non-numeric columns of ``data`` and the columns the regressor
    declares in ``categorical_columns`` (by name or position), whose codes would
    be meaningless to average.
    """
    import numpy as np
    import pandas as pd

    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(np.asarray(data))
    names = [str(column) for column in getattr(model, "feature_names_in_", frame.columns)]
    indices = {
        position
        for position, dtype in enumerate(frame.infer_objects().dtypes)
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
    }
    get_params = getattr(model, "get_params", None)
    declared = get_params().get("categorical_columns") if callable(get_params) else None
    if isinstance(declared, (list, tuple)):
        for column in declared:
            if isinstance(column, (int, np.integer)):
                indices.add(int(column))
            elif str(column) in names:
                indices.add(names.index(str(column)))
    return sorted(indices)


def get_nori_imputation_explainer(
    model,
    data,
    *,
    index: str = "k-SII",
    max_order: int = 2,
    imputer: str = "baseline",
    random_state: int | None = 0,
    **kwargs: Any,
):
    """Recommended explainer: imputation-based feature removal (model-agnostic).

    Args:
        model: a fitted ``NoriRegressor``.
        data: background data for the imputer, shape ``(n, n_features)``: an
            array, or a DataFrame whose columns may hold text or categories.
            Categorical features (non-numeric columns and the regressor's
            ``categorical_columns``) are imputed by their most common value.
        index: Shapley index — ``"SV"`` (values), ``"k-SII"`` (k-Shapley
            interactions; with ``max_order=1`` it reduces to standard Shapley
            values), or any other shapiq tabular index.
        max_order: maximum interaction order (``1`` = single-feature attributions).
        imputer: ``"baseline"`` (one forward per coalition; recommended),
            ``"marginal"`` or ``"conditional"`` (multi-sample, much slower).
        random_state: seed for reproducible coalition sampling.
        **kwargs: forwarded to ``shapiq.TabularExplainer``.

    Returns:
        ``shapiq.TabularExplainer``. Call ``.explain(x, budget=N)`` with ``x`` of
        shape ``(1, n_features)``; the result exposes ``.plot_waterfall()`` etc.
    """
    shapiq = _require_shapiq()
    categorical = _get_categorical_feature_indices(model, data)
    if imputer == "baseline" and categorical and "categorical_features" not in kwargs:
        kwargs["categorical_features"] = categorical
    return shapiq.TabularExplainer(
        model=_get_predict_function(model, data),
        data=_get_background_array(data),
        index=index,
        max_order=max_order,
        imputer=imputer,
        random_state=random_state,
        **kwargs,
    )
