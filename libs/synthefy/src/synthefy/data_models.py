from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from synthefy.nori_data_models import MemoryPolicyInput, MemoryReport


class NoriPredictRequest(BaseModel):
    """Request payload for a Synthefy Nori prediction.

    Mirrors the hosted inference contract exactly::

        {"X_train": [[...], ...], "y_train": [...], "X_test": [[...], ...],
         "task": "regression"}

    Parameters
    ----------
    X_train : List[List[float | None]]
        Labeled context rows. Shape ``(n_context, n_features)``.
    y_train : List[float]
        Target value for each context row. Length ``n_context``.
    X_test : List[List[float | None]]
        Query rows to predict. Shape ``(n_query, n_features)``.
    task : str, default "regression"
        The prediction task. Currently only ``"regression"`` is supported.
    memory_policy : str or dict, optional
        Serving-memory policy, at parity with the local package's
        ``NoriRegressor(memory_policy=...)``: a preset name (``"exact"``,
        ``"max_context"``, ``"off"``) or an object of policy fields. Omitted from
        the wire payload entirely when unset, so a request that does not use it is
        byte-for-byte what it always was.
    output_type : str or None, optional
        What to return from the predictive distribution (``"mean"``,
        ``"median"``, ``"quantiles"``, ``"full"``). ``None`` means
        "the server default", which is ``"mean"``; it is **omitted from the
        request body** so the default request is byte-for-byte what earlier
        client versions sent.
    quantiles : List[float] or None, optional
        Tau levels in ``(0, 1)`` for ``output_type="quantiles"``, in the caller's
        order. Omitted from the body when ``None``.
    """

    # Coerce assignments just as construction does, so assigning a policy dict
    # cannot leave the field holding a value that contradicts its declared type.
    model_config = ConfigDict(validate_assignment=True)

    X_train: List[List[Optional[float]]]
    y_train: List[float]
    X_test: List[List[Optional[float]]]
    task: str = "regression"
    memory_policy: Optional[MemoryPolicyInput] = None
    output_type: Optional[str] = None
    quantiles: Optional[List[float]] = None

    def to_wire(self) -> Dict[str, Any]:
        """Serialize the request without pinning optional server defaults.

        In particular, a partial ``MemoryPolicy`` must carry only the fields the caller set.
        Sending the model's other defaults would make an older client override future server
        defaults. ``SynthefyNoriClient`` and serving contract tests both use this method so the
        wire representation has one implementation.
        """
        payload = self.model_dump(exclude={"memory_policy"}, exclude_none=True)
        if self.memory_policy is not None:
            payload["memory_policy"] = (
                self.memory_policy
                if isinstance(self.memory_policy, str)
                else self.memory_policy.model_dump(exclude_unset=True)
            )
        return payload


class NoriPredictResponse(BaseModel):
    """Response payload from a Synthefy Nori prediction.

    Parameters
    ----------
    task : str
        The task echoed back by the server (e.g. ``"regression"``).
    predictions : List[float | None]
        One point prediction per row of ``X_test``: the summary named by the
        request's ``output_type``, or the distribution mean when a
        distribution output (``"quantiles"``/``"full"``) was requested. A
        server-side non-finite result is represented as ``None`` on this wire
        model and converted back to ``NaN`` by
        :meth:`synthefy.nori_client.SynthefyNoriClient.predict`.
    model : str or None, optional
        The model identity that produced the response. SageMaker returns this so
        the client can verify that a named endpoint served the requested model and
        fail closed instead of returning valid-looking predictions from the wrong
        model specification.
    memory_report : dict, optional
        Present only when the request set ``memory_policy``: what the server actually did
        about it. See
        :attr:`synthefy.nori_client.SynthefyNoriClient.last_memory_report`.
    output_type : str or None, optional
        The output type the server actually honored. A deployment that predates
        distribution output omits this field, which is how the client detects
        that its ``output_type`` was silently ignored (see
        :meth:`synthefy.nori_client.SynthefyNoriClient._predict_remote`) instead
        of handing back means labeled as something else.
    quantiles : List[List[float | None]] or None, optional
        The predictive quantile function, **one row per query row**:
        ``(n_query, K)`` ascending values in original-``y`` units. For
        ``output_type="quantiles"`` ``K`` is the number of requested levels (in
        the requested order, not sorted); for ``"full"`` it is the checkpoint's
        whole quantile bank. Present only when a distribution output was
        requested. Entries are nullable because JSON has no ``NaN``: the server
        sends ``null`` for a non-finite value and the client maps it back to
        ``NaN``.
    taus : List[float] or None, optional
        The ``K`` quantile levels matching ``quantiles``' columns.
    """

    task: str
    predictions: List[Optional[float]]
    model: Optional[str] = None
    memory_report: Optional[MemoryReport] = None
    output_type: Optional[str] = None
    quantiles: Optional[List[List[Optional[float]]]] = None
    taus: Optional[List[float]] = None


# TODO - Concept of Rows/Scenarios/etc needs to be explained.
# TODO - concept of Leak needs to be explained.
class SingleEvalSamplePayload(BaseModel):
    """
    A single payload for forecasting models.

    This class represents a single time series sample with historical and target
    data, along with metadata about whether the sample should be forecasted,
    contains metadata, or has target leakage.

    Parameters
    ----------
    sample_id : Any
        Unique identifier for the sample. Can be any hashable type.
        This will become a column in the output dataframe.
    history_timestamps : List[str]
        List of timestamp strings for historical data points.
        Must be in ISO format or compatible with pandas datetime parsing.
    history_values : List[Optional[float]]
        List of historical values corresponding to history_timestamps.
        None values represent missing data points.
    target_timestamps : List[str]
        List of timestamp strings for target (future) data points.
        Must be in ISO format or compatible with pandas datetime parsing.
    target_values : List[Optional[float]]
        List of target values corresponding to target_timestamps.
        None values represent missing data points.
    forecast : bool, default True
        Whether this sample should be used for forecasting.
        If False, the sample is used for metadata only.
    metadata : bool, default False
        Whether this sample contains metadata information.
        Metadata samples are not forecasted but provide context.
    leak_target : bool, default False
        Whether this sample contains target leakage.
        Leakage samples should be excluded from training to prevent
        data leakage in model evaluation.
    column_name : str or None, default None
        Name of the column this sample represents.
        Used for identification in multi-column datasets.

    Raises
    ------
    ValueError
        If timestamps or values lists are empty.
        If history and target data have mismatched lengths.
        If values contain NaN that cannot be converted to None.

    Notes
    -----
    - All timestamp lists must be non-empty
    - NaN values in history_values and target_values are converted to None
      for JSON serialization compatibility

    Examples
    --------
    Create a simple forecast sample:

    >>> sample = SingleEvalSamplePayload(
    ...     sample_id="sales",
    ...     history_timestamps=["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     history_values=[100.0, 110.0, 120.0],
    ...     target_timestamps=["2023-01-04", "2023-01-05"],
    ...     target_values=[130.0, 140.0],
    ...     forecast=True,
    ...     metadata=False,
    ...     leak_target=False,
    ...     column_name="sales"
    ... )

    Create a metadata sample with missing values:

    >>> metadata_sample = SingleEvalSamplePayload(
    ...     sample_id="temperature",
    ...     history_timestamps=["2023-01-01", "2023-01-02", "2023-01-03"],
    ...     history_values=[20.5, None, 22.1],  # Missing value on 2023-01-02
    ...     target_timestamps=["2023-01-04", "2023-01-05"],
    ...     target_values=[23.0, 24.5],
    ...     forecast=False,
    ...     metadata=True,
    ...     leak_target=False,
    ...     column_name="temperature"
    ... )

    Create a sample with target leakage:

    >>> leak_sample = SingleEvalSamplePayload(
    ...     sample_id="future_sales",
    ...     history_timestamps=["2023-01-01", "2023-01-02"],
    ...     history_values=[100.0, 110.0],
    ...     target_timestamps=["2023-01-03", "2023-01-04"],
    ...     target_values=[120.0, 130.0],
    ...     forecast=True,
    ...     metadata=False,
    ...     leak_target=True,  # This sample has target leakage
    ...     column_name="future_sales"
    ... )

    See Also
    --------
    ForecastV2Request : Container for multiple evaluation samples
    SingleSampleForecastPayload : Forecast output for a single sample
    """

    sample_id: Any
    history_timestamps: List[str]
    history_values: List[Optional[float]]
    target_timestamps: List[str]
    target_values: List[Optional[float]]
    forecast: bool = True
    metadata: bool = False
    leak_target: bool = False
    column_name: Optional[str] = None

    @field_validator("history_timestamps", "target_timestamps")
    @classmethod
    def validate_timestamps_not_empty(cls, v):
        """Validate that timestamp lists are not empty."""
        if not v:
            raise ValueError("Timestamps cannot be empty")
        return v

    @field_validator("history_values", "target_values")
    @classmethod
    def validate_values_not_empty(cls, v):
        """Validate that values lists are not empty."""
        if not v:
            raise ValueError("Values cannot be empty")
        return v

    @field_validator("history_values", "target_values")
    @classmethod
    def validate_values_json_compliant(cls, v):
        """Convert NaN values to None for JSON compliance."""
        if v is None:
            return v
        return [
            None if (isinstance(val, float) and np.isnan(val)) else val
            for val in v
        ]

    @field_validator("history_values", "target_values")
    @classmethod
    def validate_values_length_match_timestamps(cls, v, info):
        """Validate that values and timestamps have the same length."""
        if info.data.get("history_timestamps") and info.data.get(
            "history_values"
        ):
            if len(info.data["history_timestamps"]) != len(
                info.data["history_values"]
            ):
                raise ValueError(
                    f"History timestamps and values must have the same length. "
                    f"Got {len(info.data['history_timestamps'])} and {len(info.data['history_values'])}"
                )

        if info.data.get("target_timestamps") and info.data.get(
            "target_values"
        ):
            if len(info.data["target_timestamps"]) != len(
                info.data["target_values"]
            ):
                raise ValueError(
                    f"Target timestamps and values must have the same length. "
                    f"Got {len(info.data['target_timestamps'])} and {len(info.data['target_values'])}"
                )

        return v


class SingleSampleForecastPayload(BaseModel):
    """
    A single sample forecast payload containing forecast results.

    This class represents the output of a forecasting model for a single
    time series sample. It contains the forecasted values along with
    metadata about the model used and the sample.

    Parameters
    ----------
    sample_id : Any
        Unique identifier for the sample. Must match the input sample_id.
    timestamps : List[str]
        List of timestamp strings for the forecasted values.
        Must be in ISO format or compatible with pandas datetime parsing.
    values : List[Optional[float]]
        List of forecasted values corresponding to timestamps.
        None values represent missing or invalid forecasts.
    #TODO @Aditya model_name should be literal/enum!
    model_name : str
        Name of the model used to generate the forecast.
        Must be a non-empty string.
    quantiles : Optional[dict[str, List[Optional[float]]]]
        Optional quantile forecasts (0.1 to 0.9). Keys are quantile levels as strings.

    Raises
    ------
    ValueError
        If timestamps or values lists are empty.
        If timestamps and values have mismatched lengths.
        If model_name is empty or whitespace.

    Notes
    -----
    - NaN values in values are converted to None for JSON compatibility
    - model_name is automatically trimmed of whitespace
    - quantiles field is excluded from JSON output when None

    Examples
    --------
    Create a simple forecast:

    >>> forecast = SingleSampleForecastPayload(
    ...     sample_id="sales",
    ...     timestamps=["2023-01-04", "2023-01-05"],
    ...     values=[130.0, 140.0],
    ...     model_name="sfm-moe-v1"
    ... )

    Create a forecast with missing values:

    >>> forecast = SingleSampleForecastPayload(
    ...     sample_id="temperature",
    ...     timestamps=["2023-01-04", "2023-01-05"],
    ...     values=[23.0, None],  # Missing forecast for 2023-01-05
    ...     model_name="sfm-moe-v1"
    ... )

    See Also
    --------
    SingleEvalSamplePayload : Input sample for forecasting
    ForecastV2Response : Container for multiple forecast results
    """

    model_config = {"protected_namespaces": ()}

    sample_id: Any
    timestamps: List[str]
    values: List[Optional[float]]
    model_name: str
    quantiles: Optional[dict[str, List[Optional[float]]]] = None

    @field_validator("values")
    @classmethod
    def validate_values_json_compliant(cls, v):
        """Convert NaN values to None for JSON compliance."""
        if v is None:
            return v
        return [
            None if (isinstance(val, float) and np.isnan(val)) else val
            for val in v
        ]

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles_json_compliant(cls, v):
        """Convert NaN values to None for JSON compliance in quantiles."""
        if v is None:
            return v
        return {
            q_level: [
                None if (isinstance(val, float) and np.isnan(val)) else val
                for val in q_values
            ]
            for q_level, q_values in v.items()
        }

    @field_validator("values")
    @classmethod
    def validate_values_length_match_timestamps(cls, v, info):
        """Validate that values and timestamps have the same length."""
        if info.data.get("timestamps") and len(info.data["timestamps"]) != len(
            v
        ):
            raise ValueError(
                f"Timestamps and values must have the same length. "
                f"Got {len(info.data['timestamps'])} and {len(v)}"
            )
        return v

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles_length_match_timestamps(cls, v, info):
        """Validate that quantile values match timestamps length."""
        if v is None:
            return v
        if info.data.get("timestamps"):
            expected_len = len(info.data["timestamps"])
            for q_level, q_values in v.items():
                if len(q_values) != expected_len:
                    raise ValueError(
                        f"Quantile {q_level} must have the same length as timestamps. "
                        f"Got {len(q_values)} but expected {expected_len}"
                    )
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name_not_empty(cls, v):
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty or whitespace")
        return v.strip()


class ForecastV2Request(BaseModel):
    """
    A request for forecasting and model specification.

    This class represents a complete forecasting request containing multiple
    time series samples and the model to use for forecasting. Each sample
    can contain multiple time series (target + metadata columns).

    Parameters
    ----------
    samples : List[List[SingleEvalSamplePayload]]
        List of sample rows, where each row contains 1 or more time series
        samples (target + metadata). All samples in a row must have the same
        timestamps for consistency.
    #TODO @Aditya model should be literal/enum!
    model : str
        Name of the model to use for forecasting. Must be a non-empty string.
        The model name is trimmed of whitespace.

    Raises
    ------
    ValueError
        If samples list is empty.
        If any sample row is empty.
        If samples in a row have mismatched timestamps.
        If model name is empty or whitespace.

    Notes
    -----
    - Each sample row represents a single forecast / forecasting scenario
    - The model name is automatically trimmed of leading/trailing whitespace
    - Samples are validated for consistency before processing

    Examples
    --------
    Create a request with multiple sample rows:

    >>> # First scenario: high sales, warm weather
    >>> scenario1 = [
    ...     SingleEvalSamplePayload(
    ...         sample_id="sales",
    ...         history_timestamps=["2023-01-01", "2023-01-02"],
    ...         history_values=[100.0, 110.0],
    ...         target_timestamps=["2023-01-03", "2023-01-04"],
    ...         target_values=[120.0, 130.0]
    ...     ),
    ...     SingleEvalSamplePayload(
    ...         sample_id="temperature",
    ...         history_timestamps=["2023-01-01", "2023-01-02"],
    ...         history_values=[25.0, 26.0],
    ...         target_timestamps=["2023-01-03", "2023-01-04"],
    ...         target_values=[27.0, 28.0],
    ...         metadata=True
    ...     )
    ... ]
    >>> # Second scenario: low sales, cold weather
    >>> scenario2 = [
    ...     SingleEvalSamplePayload(
    ...         sample_id="sales",
    ...         history_timestamps=["2023-01-01", "2023-01-02"],
    ...         history_values=[50.0, 55.0],
    ...         target_timestamps=["2023-01-03", "2023-01-04"],
    ...         target_values=[60.0, 65.0]
    ...     ),
    ...     SingleEvalSamplePayload(
    ...         sample_id="temperature",
    ...         history_timestamps=["2023-01-01", "2023-01-02"],
    ...         history_values=[5.0, 6.0],
    ...         target_timestamps=["2023-01-03", "2023-01-04"],
    ...         target_values=[7.0, 8.0],
    ...         metadata=True
    ...     )
    ... ]
    >>> request = ForecastV2Request(
    ...     samples=[scenario1, scenario2],
    ...     model="ensemble_model"
    ... )
    >>> print(f"Number of scenarios: {len(request.samples)}")
    Number of scenarios: 2

    See Also
    --------
    SingleEvalSamplePayload : Individual time series sample
    ForecastV2Response : Response containing forecast results
    """

    samples: List[List[SingleEvalSamplePayload]]
    model: str

    @classmethod
    def from_dfs_pre_split(
        cls,
        dfs: List[pd.DataFrame],
        timestamp_col: str,
        target_cols: List[str],
        model: str,
        cutoff_date: Optional[str] = None,
        num_target_rows: Optional[int] = None,
        metadata_cols: List[str] = [],
        leak_cols: List[str] = [],
        forecast_window: Optional[Union[str, int]] = None,
        stride: Optional[Union[str, int]] = None,
    ) -> "ForecastV2Request":
        """
        Create a ForecastV2Request from pandas DataFrames with backtesting support.

        This method creates evaluation batches from multiple DataFrames, supporting both
        single-window and backtesting scenarios. The method automatically determines the
        appropriate processing mode based on the provided parameters.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of pandas DataFrames to process. Each DataFrame should contain time series
            data with a timestamp column and various feature columns.
        timestamp_col : str
            Name of the column containing timestamps. This column will be converted to
            datetime format and used for splitting data into history and target periods.
        target_cols : List[str]
            Column names to be used as forecast targets. These columns will have
            forecast=True in the resulting SingleEvalSamplePayload objects.
        #TODO @Aditya should this be a literal/enum?
        model : str
            Name of the model to use for forecasting.
        cutoff_date : Optional[str], default=None
            Date string (e.g., "2023-01-01") to split data into history (≤ cutoff) and
            target (> cutoff) periods. Mutually exclusive with num_target_rows.
        num_target_rows : Optional[int], default=None
            Number of rows to use as target period (taken from the end of each DataFrame).
            Mutually exclusive with cutoff_date.
        metadata_cols : List[str], default=[]
            Column names to be used as metadata/correlates. These columns will have
            metadata=True in the resulting SingleEvalSamplePayload objects.
        leak_cols : List[str], default=[]
            Column names that are allowed to leak target information. Must be a subset
            of metadata_cols. These columns will have leak_target=True.
        forecast_window : Optional[Union[str, int]], default=None
            For backtesting: the size of the forecast window. If str, interpreted as
            pandas time offset (e.g., "7D" for 7 days). If int, interpreted as number
            of rows. Must be provided together with stride.
        stride : Optional[Union[str, int]], default=None
            For backtesting: the step size between consecutive forecasts. If str,
            interpreted as pandas time offset. If int, interpreted as number of rows.
            Must be provided together with forecast_window.

        Returns
        -------
        ForecastV2Request
            A fully constructed forecasting request with samples created from the input DataFrames.

        Raises
        ------
        ValueError
            - If both cutoff_date and num_target_rows are provided
            - If neither cutoff_date nor num_target_rows are provided
            - If forecast_window is provided without stride or vice versa
            - If no history rows are found for the given cutoff_date
            - If type mismatches occur (e.g., string vs int for forecast_window/stride)
            - If leak_cols is not a subset of metadata_cols
            - If target_cols overlap with metadata_cols

        Notes
        -----
        The method supports four processing modes:
        1. Single window by date: cutoff_date provided, no forecast_window/stride
        2. Single window by rows: num_target_rows provided, no forecast_window/stride
        3. Backtesting by date: cutoff_date + forecast_window/stride (all strings)
        4. Backtesting by rows: num_target_rows + forecast_window/stride (all integers)

        Each DataFrame is processed independently, and the results are combined into
        a single ForecastV2Request object. The timestamp column is automatically converted
        to datetime format and sorted chronologically.

        Examples
        --------
        # Single window by date
        request = ForecastV2Request.from_df(
            dfs=[df1, df2],
            timestamp_col="date",
            target_cols=["sales"],
            model="synthefy-fm",
            cutoff_date="2023-06-01",
            metadata_cols=["temperature", "holiday"]
        )

        # Backtesting by date
        request = ForecastV2Request.from_df(
            dfs=[df1],
            timestamp_col="date",
            target_cols=["sales"],
            model="synthefy-fm",
            cutoff_date="2023-01-01",
            forecast_window="7D",
            stride="1D"
        )

        # Backtesting by rows
        request = ForecastV2Request.from_df(
            dfs=[df1],
            timestamp_col="date",
            target_cols=["sales"],
            model="synthefy-fm",
            num_target_rows=30,
            forecast_window=7,
            stride=1,
            metadata_cols=["temperature"]
        )
        """
        # Validate inputs
        cls._validate_backtesting_inputs(
            cutoff_date,
            num_target_rows,
            forecast_window,
            stride,
            target_cols,
            metadata_cols,
            leak_cols,
        )

        # Decide if we're doing backtesting or not.
        # We are doing backtesting if we have a forecast_window and stride.
        backtesting = False
        if forecast_window is not None or stride is not None:
            if forecast_window is None or stride is None:
                raise ValueError(
                    "Forecast Window and Stride must be provided together"
                )
            backtesting = True

        # Decide if we're splitting dataframes by date or by rows
        if cutoff_date is not None:
            if num_target_rows is not None:
                raise ValueError(
                    "Only one of cutoff_date or num_target_rows can be provided"
                )
            split_by_date = True
        else:
            if num_target_rows is None:
                raise ValueError(
                    "Either cutoff_date or num_target_rows must be provided"
                )
            split_by_date = False

        # Ensure timestamp_col is datetime in all DataFrames
        for i, df in enumerate(dfs):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            dfs[i] = df.sort_values(by=timestamp_col).reset_index(drop=True)

        # For each dataframe, dispatch to the appropriate helper method
        # Note that we ignore some types since they are validated in validate_backtesting_inputs()
        all_samples = []
        for df in dfs:
            if backtesting:
                if split_by_date:
                    all_samples.extend(
                        cls._backtesting_by_date(
                            df,
                            timestamp_col,
                            cutoff_date,  # type: ignore[arg-type]
                            target_cols,
                            metadata_cols,
                            leak_cols,
                            forecast_window,  # type: ignore[arg-type]
                            stride,  # type: ignore[arg-type]
                        )
                    )
                else:
                    all_samples.extend(
                        cls._backtesting_by_rows(
                            df,
                            timestamp_col,
                            num_target_rows,  # type: ignore[arg-type]
                            target_cols,
                            metadata_cols,
                            leak_cols,
                            forecast_window,  # type: ignore[arg-type]
                            stride,  # type: ignore[arg-type]
                        )
                    )
            else:
                if split_by_date:
                    all_samples.extend(
                        cls._single_window_by_date(
                            df,
                            timestamp_col,
                            cutoff_date,  # type: ignore[arg-type]
                            target_cols,
                            metadata_cols,
                            leak_cols,
                        )
                    )
                else:
                    all_samples.extend(
                        cls._single_window_by_rows(
                            df,
                            timestamp_col,
                            num_target_rows,  # type: ignore[arg-type]
                            target_cols,
                            metadata_cols,
                            leak_cols,
                        )
                    )

        if len(all_samples) == 0:
            raise ValueError(
                "No valid windows could be created from the provided dataframes. "
                "Please check your inputs and try again."
            )

        return cls(samples=all_samples, model=model)

    @classmethod
    def from_dfs(
        cls,
        history_dfs: List[pd.DataFrame],
        target_dfs: List[pd.DataFrame],
        target_col: str,
        timestamp_col: str,
        metadata_cols: List[str],
        leak_cols: List[str],
        model: str,
    ) -> "ForecastV2Request":
        """
        Create a ForecastV2Request from pandas DataFrames.

        This method converts pandas DataFrames into the structured format
        required for forecasting requests. It handles validation, data
        conversion, and sample creation automatically.

        Parameters
        ----------
        history_dfs : List[pd.DataFrame]
            List of DataFrames containing historical data. Each DataFrame
            represents one forecasting scenario.
        target_dfs : List[pd.DataFrame]
            List of DataFrames containing target (future) data. Must have
            the same length as history_dfs.
        target_col : str
            Name of the target column to forecast. Must exist in all
            DataFrames and cannot be in metadata_cols.
        timestamp_col : str
            Name of the timestamp column. Must exist in all DataFrames
            and cannot be in metadata_cols.
        metadata_cols : List[str]
            List of metadata column names. These columns provide context
            but are not forecasted. Cannot include target_col or timestamp_col.
        leak_cols : List[str]
            List of columns that should be marked as leak_target=True.
            Must be a subset of metadata_cols.
        #TODO @Aditya model should be literal/enum!
        model : str
            Name of the model to use for forecasting.

        Returns
        -------
        ForecastV2Request
            A fully constructed forecasting request with samples created
            from the input DataFrames.

        Raises
        ------
        ValueError
            If history_dfs and target_dfs have different lengths.
            If any DataFrame is missing required columns.
            If all DataFrames don't have consistent column structure.
            If leak_cols is not a subset of metadata_cols.
            If target_col or timestamp_col are in metadata_cols.
            If any DataFrame is empty.

        Notes
        -----
        - All DataFrames must have the same column structure
        - NaN values are automatically converted to None for JSON compatibility
        - Each DataFrame pair (history_df, target_df) creates one sample row/forecast scenario
        - Target column creates a forecast sample, metadata columns create metadata samples
        - Leak columns are marked with leak_target=True

        Examples
        --------
        Create a request from simple DataFrames:

        >>> # Create historical data
        >>> history_data = {
        ...     'timestamp': pd.date_range('2023-01-01', periods=3, freq='D'),
        ...     'sales': [100, 110, 120],
        ...     'temperature': [20, 21, 22]
        ... }
        >>> history_df = pd.DataFrame(history_data)
        >>>
        >>> # Create target data
        >>> target_data = {
        ...     'timestamp': pd.date_range('2023-01-04', periods=2, freq='D'),
        ...     'sales': [None, None],
        ...     'temperature': [23, 24]
        ... }
        >>> target_df = pd.DataFrame(target_data)
        >>>
        >>> # Create request
        >>> request = ForecastV2Request.from_dfs(
        ...     history_dfs=[history_df],
        ...     target_dfs=[target_df],
        ...     target_col='sales',
        ...     timestamp_col='timestamp',
        ...     metadata_cols=['temperature'],
        ...     leak_cols=[],
        ...     model='sfm-moe-v1'
        ... )

        Create a request with multiple scenarios and leak columns:

        >>> # Scenario 1: High sales
        >>> scenario1_history = pd.DataFrame({
        ...     'timestamp': pd.date_range('2023-01-01', periods=2, freq='D'),
        ...     'sales': [100, 110],
        ...     'temperature': [25, 26],
        ...     'promotion': [1, 1]  # This will be a leak column
        ... })
        >>> scenario1_target = pd.DataFrame({
        ...     'timestamp': pd.date_range('2023-01-03', periods=2, freq='D'),
        ...     'sales': [None, None],
        ...     'temperature': [27, 28],
        ...     'promotion': [1, 1]
        ... })
        >>>
        >>> # Scenario 2: Low sales
        >>> scenario2_history = pd.DataFrame({
        ...     'timestamp': pd.date_range('2023-01-01', periods=2, freq='D'),
        ...     'sales': [50, 55],
        ...     'temperature': [5, 6],
        ...     'promotion': [0, 0]
        ... })
        >>> scenario2_target = pd.DataFrame({
        ...     'timestamp': pd.date_range('2023-01-03', periods=2, freq='D'),
        ...     'sales': [None, None],
        ...     'temperature': [7, 8],
        ...     'promotion': [0, 0]
        ... })
        >>>
        >>> request = ForecastV2Request.from_dfs(
        ...     history_dfs=[scenario1_history, scenario2_history],
        ...     target_dfs=[scenario1_target, scenario2_target],
        ...     target_col='sales',
        ...     timestamp_col='timestamp',
        ...     metadata_cols=['temperature', 'promotion'],
        ...     leak_cols=['promotion'],  # Promotion data has target leakage
        ...     model='ensemble_model'
        ... )

        See Also
        --------
        ForecastV2Request : The main class for forecasting requests
        SingleEvalSamplePayload : Individual sample structure
        """
        # Validate inputs before processing
        cls._validate_dataframe_inputs(
            history_dfs,
            target_dfs,
            target_col,
            timestamp_col,
            metadata_cols,
            leak_cols,
        )

        samples = []

        for i, (history_df, target_df) in enumerate(
            zip(history_dfs, target_dfs)
        ):
            sample_row = []

            # Create sample for target column
            target_sample = SingleEvalSamplePayload(
                sample_id=target_col,
                history_timestamps=history_df[timestamp_col]
                .astype(str)
                .tolist(),
                history_values=cls._convert_nan_to_none(
                    history_df[target_col].tolist()
                ),
                target_timestamps=target_df[timestamp_col].astype(str).tolist(),
                target_values=cls._convert_nan_to_none(
                    target_df[target_col].tolist()
                ),
                forecast=True,
                metadata=False,
                leak_target=target_col in leak_cols,
                column_name=target_col,
            )
            sample_row.append(target_sample)

            # Create samples for metadata columns
            for col in metadata_cols:
                if col in history_df.columns and col in target_df.columns:
                    metadata_sample = SingleEvalSamplePayload(
                        sample_id=col,
                        history_timestamps=history_df[timestamp_col]
                        .astype(str)
                        .tolist(),
                        history_values=cls._convert_nan_to_none(
                            history_df[col].tolist()
                        ),
                        target_timestamps=target_df[timestamp_col]
                        .astype(str)
                        .tolist(),
                        target_values=cls._convert_nan_to_none(
                            target_df[col].tolist()
                        ),
                        forecast=False,
                        metadata=True,
                        leak_target=col in leak_cols,
                        column_name=col,
                    )
                    sample_row.append(metadata_sample)

            samples.append(sample_row)

        return cls(samples=samples, model=model)

    @staticmethod
    def _convert_nan_to_none(values):
        """
        Convert NaN values to None for JSON compliance.

        Parameters
        ----------
        values : list of float or None
            List of values to process.

        Returns
        -------
        list of float or None
            List with NaN values converted to None.
        """
        return [
            None if (isinstance(val, float) and np.isnan(val)) else val
            for val in values
        ]

    @classmethod
    def _validate_dataframe_inputs(
        cls,
        history_dfs: List[pd.DataFrame],
        target_dfs: List[pd.DataFrame],
        target_col: str,
        timestamp_col: str,
        metadata_cols: List[str],
        leak_cols: List[str],
    ) -> None:
        """
        Validate DataFrame inputs before processing.

        Raises:
            ValueError: If validation fails
        """
        # Check that lists have the same length
        if len(history_dfs) != len(target_dfs):
            raise ValueError(
                f"history_dfs and target_dfs must have the same length. "
                f"Got {len(history_dfs)} and {len(target_dfs)}"
            )

        if not history_dfs:
            raise ValueError("history_dfs and target_dfs cannot be empty")

        # Check that all DataFrames have the same columns
        all_required_cols = {target_col, timestamp_col} | set(metadata_cols)

        for i, (history_df, target_df) in enumerate(
            zip(history_dfs, target_dfs)
        ):
            # Check history DataFrame columns
            missing_in_history = all_required_cols - set(history_df.columns)
            if missing_in_history:
                raise ValueError(
                    f"History DataFrame {i} is missing required columns: {missing_in_history}"
                )

            # Check target DataFrame columns
            missing_in_target = all_required_cols - set(target_df.columns)
            if missing_in_target:
                raise ValueError(
                    f"Target DataFrame {i} is missing required columns: {missing_in_target}"
                )

            # Check that all DataFrames have the same columns (for consistency)
            if i > 0:
                if set(history_dfs[0].columns) != set(history_df.columns):
                    raise ValueError(
                        f"All history DataFrames must have the same columns. "
                        f"DataFrame 0: {set(history_dfs[0].columns)}, "
                        f"DataFrame {i}: {set(history_df.columns)}"
                    )

                if set(target_dfs[0].columns) != set(target_df.columns):
                    raise ValueError(
                        f"All target DataFrames must have the same columns. "
                        f"DataFrame 0: {set(target_dfs[0].columns)}, "
                        f"DataFrame {i}: {set(target_df.columns)}"
                    )

        # Check that leak columns are a strict subset of metadata columns
        leak_set = set(leak_cols)
        metadata_set = set(metadata_cols)

        if not leak_set.issubset(metadata_set):
            invalid_leak_cols = leak_set - metadata_set
            raise ValueError(
                f"Leak columns must be a subset of metadata columns. "
                f"Invalid leak columns: {invalid_leak_cols}"
            )

        # Check that target_col is not in metadata_cols (to avoid duplication)
        if target_col in metadata_cols:
            raise ValueError(
                f"target_col '{target_col}' should not be in metadata_cols to avoid duplication"
            )

        # Check that timestamp_col is not in metadata_cols (to avoid confusion)
        if timestamp_col in metadata_cols:
            raise ValueError(
                f"timestamp_col '{timestamp_col}' should not be in metadata_cols to avoid confusion"
            )

    @classmethod
    def _validate_backtesting_inputs(
        cls,
        cutoff_date: Optional[str],
        num_target_rows: Optional[int],
        forecast_window: Optional[Union[str, int]],
        stride: Optional[Union[str, int]],
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
    ) -> None:
        """Validate backtesting inputs."""
        # Check mutually exclusive parameters
        if cutoff_date is not None and num_target_rows is not None:
            raise ValueError(
                "Only one of cutoff_date or num_target_rows can be provided"
            )

        if cutoff_date is None and num_target_rows is None:
            raise ValueError(
                "Either cutoff_date or num_target_rows must be provided"
            )

        # Check forecast_window and stride consistency
        if (forecast_window is None) != (stride is None):
            raise ValueError(
                "Forecast window and stride must be provided together"
            )

        # Check type consistency
        if forecast_window is not None and stride is not None:
            if cutoff_date is not None:
                # Date-based backtesting requires string parameters
                if not isinstance(cutoff_date, str):
                    raise ValueError("cutoff_date must be a string")
                if not isinstance(forecast_window, str):
                    raise ValueError(
                        "forecast_window must be a string when using cutoff_date"
                    )
                if not isinstance(stride, str):
                    raise ValueError(
                        "stride must be a string when using cutoff_date"
                    )
            else:
                # Row-based backtesting requires integer parameters
                if not isinstance(num_target_rows, int):
                    raise ValueError("num_target_rows must be an integer")
                if not isinstance(forecast_window, int):
                    raise ValueError(
                        "forecast_window must be an integer when using num_target_rows"
                    )
                if not isinstance(stride, int):
                    raise ValueError(
                        "stride must be an integer when using num_target_rows"
                    )

        # Check leak_cols is subset of metadata_cols
        if leak_cols and not set(leak_cols).issubset(set(metadata_cols)):
            raise ValueError(
                f"leak_cols must be a subset of metadata_cols. "
                f"Invalid leak columns: {set(leak_cols) - set(metadata_cols)}"
            )

        # Check target_cols and metadata_cols don't overlap
        overlap = set(target_cols) & set(metadata_cols)
        if overlap:
            raise ValueError(
                f"target_cols and metadata_cols should not overlap. "
                f"Overlapping columns: {overlap}"
            )

        # Validate num_target_rows constraints
        if num_target_rows is not None:
            if num_target_rows <= 0:
                raise ValueError("num_target_rows must be a positive integer")

    @classmethod
    def _backtesting_by_date(
        cls,
        df: pd.DataFrame,
        timestamp_col: str,
        cutoff_date: str,
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
        forecast_window: str,
        stride: str,
    ) -> List[List[SingleEvalSamplePayload]]:
        """Process a single DataFrame for date-based backtesting."""
        # Identify timezone if it exists
        tz = None
        if df[timestamp_col].dt.tz is not None:
            tz = df[timestamp_col].dt.tz
        cutoff_timestamp = pd.Timestamp(cutoff_date, tz=tz)

        windows = []
        while True:
            # Slice the dataframe into history and target based on timestamp
            # We select only targets that are within cutoff_timestamp + forecast_window
            history_df = df[df[timestamp_col] <= cutoff_timestamp]
            target_df = df[
                (df[timestamp_col] > cutoff_timestamp)
                & (
                    df[timestamp_col]
                    <= cutoff_timestamp + pd.Timedelta(forecast_window)  # type: ignore
                )
            ]

            if len(history_df) == 0:
                raise ValueError(
                    f"No history rows found for the given cutoff_date: {cutoff_date}. "
                    "Please check your inputs and try again."
                )

            # We are done if there are no target rows left
            if len(target_df) == 0:
                break

            # Split the dataframe into samples
            windows.append(
                cls.split_df_to_correlates(
                    history_df,  # type: ignore[arg-type]
                    target_df,  # type: ignore[arg-type]
                    timestamp_col,
                    target_cols,
                    metadata_cols,
                    leak_cols,
                )
            )

            # Move the cutoff timestamp forward by the stride
            cutoff_timestamp += pd.Timedelta(stride)  # type: ignore[operator]

        return windows

    @classmethod
    def _backtesting_by_rows(
        cls,
        df: pd.DataFrame,
        timestamp_col: str,
        num_target_rows: int,
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
        forecast_window: int,
        stride: int,
    ) -> List[List[SingleEvalSamplePayload]]:
        """Process a single DataFrame for row-based backtesting."""
        windows = []

        cutoff_idx = len(df) - num_target_rows
        while True:
            if cutoff_idx + forecast_window > len(df):
                break

            history_df = df.iloc[:cutoff_idx]
            target_df = df.iloc[cutoff_idx : cutoff_idx + forecast_window]
            windows.append(
                cls.split_df_to_correlates(
                    history_df,
                    target_df,
                    timestamp_col,
                    target_cols,
                    metadata_cols,
                    leak_cols,
                )
            )
            cutoff_idx += stride

        return windows

    @classmethod
    def _single_window_by_date(
        cls,
        df: pd.DataFrame,
        timestamp_col: str,
        cutoff_date: str,
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
    ) -> List[List[SingleEvalSamplePayload]]:
        """Process a single DataFrame for a single window by date."""
        # Identify timezone if it exists
        tz = None
        if df[timestamp_col].dt.tz is not None:
            tz = df[timestamp_col].dt.tz
        cutoff_timestamp = pd.Timestamp(cutoff_date, tz=tz)

        # Slice the dataframe into history and target based on timestamp
        history_mask = df[timestamp_col] <= cutoff_timestamp
        target_mask = df[timestamp_col] > cutoff_timestamp

        if len(df[history_mask]) == 0:
            raise ValueError(
                f"No history rows found for the given cutoff_date: {cutoff_date}. "
                "Please check your inputs and try again."
            )

        if len(df[target_mask]) == 0:
            return []

        # Split the dataframe into samples
        return [
            cls.split_df_to_correlates(
                df[history_mask],  # type: ignore[arg-type]
                df[target_mask],  # type: ignore[arg-type]
                timestamp_col,
                target_cols,
                metadata_cols,
                leak_cols,
            )
        ]

    @classmethod
    def _single_window_by_rows(
        cls,
        df: pd.DataFrame,
        timestamp_col: str,
        num_target_rows: int,
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
    ) -> List[List[SingleEvalSamplePayload]]:
        """Process a single DataFrame for a single window by rows."""
        if num_target_rows >= len(df):
            raise ValueError(
                "num_target_rows must be less than the number of rows in the dataframe"
            )

        return [
            cls.split_df_to_correlates(
                df.iloc[:-num_target_rows],
                df.iloc[-num_target_rows:],
                timestamp_col,
                target_cols,
                metadata_cols,
                leak_cols,
            )
        ]

    @classmethod
    def split_df_to_correlates(
        cls,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        timestamp_col: str,
        target_cols: List[str],
        metadata_cols: List[str],
        leak_cols: List[str],
    ) -> List[SingleEvalSamplePayload]:
        """Create SingleEvalSamplePayload objects from history and target DataFrames."""
        all_cols = target_cols + [
            x for x in metadata_cols if x not in target_cols
        ]

        samples = []
        for col in all_cols:
            # Convert timestamps to numpy arrays of datetime64[ns] then to string list
            history_timestamps = history_df[timestamp_col].values.astype(
                "datetime64[ns]"
            )
            history_values = history_df[col].values.astype(np.float64)
            target_timestamps = target_df[timestamp_col].values.astype(
                "datetime64[ns]"
            )
            target_values = target_df[col].values.astype(np.float64)

            # Convert timestamps to ISO format strings for the payload
            history_timestamps_str = (
                pd.to_datetime(history_timestamps).astype(str).tolist()
            )
            target_timestamps_str = (
                pd.to_datetime(target_timestamps).astype(str).tolist()
            )

            # Convert values to list and handle NaN
            history_values_list = cls._convert_nan_to_none(
                history_values.tolist()
            )
            target_values_list = cls._convert_nan_to_none(
                target_values.tolist()
            )

            sample = SingleEvalSamplePayload(
                sample_id=col,
                history_timestamps=history_timestamps_str,
                history_values=history_values_list,
                target_timestamps=target_timestamps_str,
                target_values=target_values_list,
                forecast=col in target_cols,
                metadata=col in metadata_cols,
                leak_target=col in leak_cols if leak_cols else False,
                column_name=col,
            )
            samples.append(sample)

        return samples

    @field_validator("samples")
    @classmethod
    def validate_samples_structure(cls, v):
        """Validate that samples have consistent structure."""
        if not v:
            raise ValueError("samples cannot be empty")

        # Check that all sample rows have at least one sample
        for i, sample_row in enumerate(v):
            if not sample_row:
                raise ValueError(f"Sample row {i} cannot be empty")

            # Check that all samples in a row have the same timestamps
            if len(sample_row) > 1:
                first_timestamps = sample_row[0].history_timestamps
                first_target_timestamps = sample_row[0].target_timestamps

                for j, sample in enumerate(sample_row[1:], 1):
                    if sample.history_timestamps != first_timestamps:
                        raise ValueError(
                            f"All samples in row {i} must have the same history timestamps. "
                            f"Sample 0: {len(first_timestamps)} timestamps, "
                            f"Sample {j}: {len(sample.history_timestamps)} timestamps"
                        )

                    if sample.target_timestamps != first_target_timestamps:
                        raise ValueError(
                            f"All samples in row {i} must have the same target timestamps. "
                            f"Sample 0: {len(first_target_timestamps)} timestamps, "
                            f"Sample {j}: {len(sample.target_timestamps)} timestamps"
                        )

        return v

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v):
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("model cannot be empty or whitespace")
        return v.strip()


class SubscriptionInfo(BaseModel):
    """
    Subscription information for managing billing and API limits.

    This class contains a message and URL directing users to the console
    where they can manage their subscription, view usage, and upgrade their plan.

    Parameters
    ----------
    message : str
        Message to display to users about subscription limits or upgrades.
        Example: "To manage your subscription and go beyond monthly limits, visit our console."
    console_url : str
        Full URL to the console/dashboard where users can manage subscriptions.
        Example: https://console.synthefy.com/
    """

    message: str
    console_url: str


class ForecastV2Response(BaseModel):
    """
    A response containing forecast results for multiple samples.

    This class represents the output of a forecasting request, containing
    forecasted values for all requested samples. Each forecast includes
    the model used, timestamps, and predicted values.

    Parameters
    ----------
    forecasts : List[List[SingleSampleForecastPayload]]
        List of forecast scenarios, where each scenario contains forecasts for
        multiple time series samples. Each scenario corresponds to one
        forecasting scenario from the input request.
    subscription_info : Optional[SubscriptionInfo], default None
        Optional subscription information with link to console for managing
        subscription and viewing usage. Included when console URL is configured.

    Notes
    -----
    - Each forecast row represents one forecasting scenario
    - All forecasts in a row should have the same timestamps
    - Empty forecasts (no timestamps/values) are represented as NaN columns
    - Forecasts can be converted back to DataFrames using to_dfs()

    Examples
    --------
    Create a simple forecast response:

    >>> forecast1 = SingleSampleForecastPayload(
    ...     sample_id="sales",
    ...     timestamps=["2023-01-04", "2023-01-05"],
    ...     values=[130.0, 140.0],
    ...     model_name="sfm-moe-v1"
    ... )
    >>> forecast2 = SingleSampleForecastPayload(
    ...     sample_id="temperature",
    ...     timestamps=["2023-01-04", "2023-01-05"],
    ...     values=[23.0, 24.0],
    ...     model_name="sfm-moe-v1"
    ... )
    >>> response = ForecastV2Response(forecasts=[[forecast1, forecast2]])

    Convert forecasts to DataFrames:

    >>> dfs = response.to_dfs()

    See Also
    --------
    SingleSampleForecastPayload : Individual forecast result
    ForecastV2Request : Input request for forecasting
    """

    forecasts: List[List[SingleSampleForecastPayload]]
    subscription_info: Optional[SubscriptionInfo] = None

    def to_dfs(self) -> List[pd.DataFrame]:
        """
        Convert the ForecastResponse to a list of DataFrames.

        This method converts the structured forecast response back into
        pandas DataFrames for easy analysis and visualization. Each
        DataFrame represents one forecasting scenario.

        Returns
        -------
        list of DataFrame
            List of DataFrames where each DataFrame contains forecast columns.
            Empty timestamps/values are converted to NaN columns.

        Notes
        -----
        - Each DataFrame represents one forecasting scenario
        - Timestamps are included as a regular column
        - Empty forecasts result in NaN columns
        - None values in forecasts are converted back to NaN for DataFrame compatibility

        Examples
        --------
        Convert forecasts to DataFrames:

        >>> forecast1 = SingleSampleForecastPayload(
        ...     sample_id="sales",
        ...     timestamps=["2023-01-04", "2023-01-05"],
        ...     values=[130.0, 140.0],
        ...     model_name="sfm-moe-v1"
        ... )
        >>> forecast2 = SingleSampleForecastPayload(
        ...     sample_id="temperature",
        ...     timestamps=["2023-01-04", "2023-01-05"],
        ...     values=[23.0, 24.0],
        ...     model_name="sfm-moe-v1"
        ... )
        >>> response = ForecastV2Response(forecasts=[[forecast1, forecast2]])
        >>> dfs = response.to_dfs()
        >>> print(dfs[0])
           timestamps  sales  temperature
        0  2023-01-04  130.0         23.0
        1  2023-01-05  140.0         24.0


        See Also
        --------
        SingleSampleForecastPayload : Individual forecast structure
        """
        result_dfs = []

        for forecast_row in self.forecasts:
            # Find the first valid timestamp set (all timestamps in a row are assumed equal)
            timestamps = None
            for forecast in forecast_row:
                if forecast.timestamps and forecast.values:
                    timestamps = forecast.timestamps
                    break

            if not timestamps:
                # If no valid timestamps, return empty DataFrame
                result_dfs.append(pd.DataFrame())
                continue

            # Create DataFrame with timestamps as a regular column
            df = pd.DataFrame()
            df["timestamps"] = timestamps

            # Add each forecast as a column
            for forecast in forecast_row:
                column_name = forecast.sample_id

                if not forecast.timestamps or not forecast.values:
                    # Empty timestamps/values indicate NaN column
                    pass  # df[column_name] = np.nan
                else:
                    # Convert None back to NaN for DataFrame compatibility
                    values = [
                        np.nan if val is None else val
                        for val in forecast.values
                    ]
                    df[column_name] = values

                    # Add quantile columns if available
                    if forecast.quantiles is not None:
                        for q_level, q_values in sorted(
                            forecast.quantiles.items()
                        ):
                            quantile_column_name = f"{column_name}[{q_level}]"
                            quantile_values = [
                                np.nan if val is None else val
                                for val in q_values
                            ]
                            df[quantile_column_name] = quantile_values

            result_dfs.append(df)

        return result_dfs


class APISupportedModels(str, Enum):
    # Use these names for legend labels
    PROPHET = "Prophet"
    ARIMA = "ARIMA"
    TABPFN = "TabPFN"
    TOTO = "ToTo"
    MITRA = "MITRA"
    SFM_MOE_V1 = "SFM MoE v1"
    SFM_MOE_V2 = "SFM MoE v2"


PLOT_COLORS = {
    APISupportedModels.PROPHET: "#16a34a",  # Green
    APISupportedModels.ARIMA: "#2563eb",  # Blue
    APISupportedModels.TABPFN: "#9b59b6",  # Purple
    APISupportedModels.TOTO: "#fbc02d",  # Yellow
    APISupportedModels.MITRA: "#605ed1",  # Slate
    APISupportedModels.SFM_MOE_V1: "#e74c3c",  # Vermillion
    APISupportedModels.SFM_MOE_V2: "#fc9260",  # Synthefy Orange
    "Chronos": "#89b23c",  # Light green
    "TimesFM": "#1e3a8a",  # Dark blue
}

MULTIVARIATE_SUPPORT_MODELS = [
    APISupportedModels.MITRA,
    APISupportedModels.TABPFN,
    APISupportedModels.TOTO,
    APISupportedModels.MITRA,
    APISupportedModels.ARIMA,
    APISupportedModels.SFM_MOE_V1,
    APISupportedModels.SFM_MOE_V2,
]


class ModelMetadata(BaseModel):
    model: APISupportedModels
    multivariate: bool = False

    @model_validator(mode="after")
    def validate_model_name(self):
        if self.multivariate:
            assert self.model in MULTIVARIATE_SUPPORT_MODELS, (
                "Model is not supported for multivariate forecasting"
            )
        return self

    @property
    def plot_color(self) -> str:
        return PLOT_COLORS[self.model]

    @property
    def model_name(self) -> str:
        """TODO: To be deprecated once we use this data model in backend."""
        name_primitive = self.model.value.lower().replace(" ", "-")
        if self.multivariate:
            if self.model in [
                APISupportedModels.SFM_MOE_V1,
                APISupportedModels.SFM_MOE_V2,
            ]:
                return name_primitive
            else:
                return f"{name_primitive}_multivariate"
        else:
            return f"{name_primitive}_univariate"
