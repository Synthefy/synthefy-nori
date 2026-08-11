import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd

from synthefy.data_models import ForecastV2Request, ForecastV2Response

BASE_URL = "https://forecast.synthefy.com"
ENDPOINT = "/v2/forecast"


def _is_synthefy_domain(url: str) -> bool:
    """Check if URL contains synthefy.com in its domain."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    # Check if hostname ends with synthefy.com or is synthefy.com
    return hostname == "synthefy.com" or hostname.endswith(".synthefy.com")


class SynthefyError(Exception):
    """Base error for all Synthefy client exceptions."""


class APITimeoutError(SynthefyError):
    """The request timed out before completing."""


class APIConnectionError(SynthefyError):
    """The request failed due to a connection issue."""


class APIStatusError(SynthefyError):
    """Raised when the API returns a non-2xx status code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        request_id: Optional[str] = None,
        error_code: Optional[str] = None,
        response_body: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.error_code = error_code
        self.response_body = response_body


class BadRequestError(APIStatusError):
    pass


class AuthenticationError(APIStatusError):
    pass


class PermissionDeniedError(APIStatusError):
    pass


class NotFoundError(APIStatusError):
    pass


class RateLimitError(APIStatusError):
    pass


class InternalServerError(APIStatusError):
    pass


def _extract_error_details(
    response: httpx.Response,
) -> Tuple[str, Optional[str], Optional[str], Any]:
    """Attempt to extract a professional, user-friendly error message and metadata.

    Returns a tuple of (message, request_id, error_code, parsed_body)
    """
    request_id = response.headers.get("x-request-id") or response.headers.get(
        "X-Request-Id"
    )
    parsed: Any
    message: str = f"HTTP {response.status_code} Error"
    code: Optional[str] = None

    try:
        parsed = response.json()
        # Common error shapes: {"error": {"message": str, "type"/"code": str}}, {"message": str}
        if isinstance(parsed, dict):
            error_obj: Any = parsed.get("error")
            if isinstance(error_obj, dict):
                message = (
                    error_obj.get("message")
                    or error_obj.get("detail")
                    or error_obj.get("error")
                    or message
                )
                code = error_obj.get("code") or error_obj.get("type")
                request_id = request_id or error_obj.get("request_id")
            else:
                message = (
                    parsed.get("message")
                    or parsed.get("detail")
                    or parsed.get("error")
                    or message
                )
                code = parsed.get("code") or parsed.get("type")
                request_id = request_id or parsed.get("request_id")
        else:
            parsed = response.text
            if isinstance(parsed, str) and parsed.strip():
                message = parsed.strip()[:500]
    except Exception:
        parsed = response.text
        if isinstance(parsed, str) and parsed.strip():
            message = parsed.strip()[:500]

    return message, request_id, code, parsed


def _raise_for_status(response: httpx.Response) -> None:
    if 200 <= response.status_code < 300:
        return

    message, request_id, code, parsed = _extract_error_details(response)
    status = response.status_code

    if status == 400 or status == 422:
        raise BadRequestError(
            message,
            status_code=status,
            request_id=request_id,
            error_code=code,
            response_body=parsed,
        )
    if status == 401:
        raise AuthenticationError(
            message,
            status_code=status,
            request_id=request_id,
            error_code=code,
            response_body=parsed,
        )
    if status == 403:
        raise PermissionDeniedError(
            message,
            status_code=status,
            request_id=request_id,
            error_code=code,
            response_body=parsed,
        )
    if status == 404:
        raise NotFoundError(
            message,
            status_code=status,
            request_id=request_id,
            error_code=code,
            response_body=parsed,
        )
    if status == 429:
        raise RateLimitError(
            message,
            status_code=status,
            request_id=request_id,
            error_code=code,
            response_body=parsed,
        )
    if 500 <= status <= 599:
        raise InternalServerError(
            message,
            status_code=status,
            request_id=request_id,
            error_code=code,
            response_body=parsed,
        )

    raise APIStatusError(
        message,
        status_code=status,
        request_id=request_id,
        error_code=code,
        response_body=parsed,
    )


class SynthefyAPIClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 300.0,
        max_retries: int = 2,
        base_url: str = BASE_URL,
        endpoint: str = ENDPOINT,
        organization: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        # Only require API key for synthefy.com domains
        if _is_synthefy_domain(base_url):
            if api_key is None:
                api_key = os.getenv("SYNTHEFY_API_KEY")
                if api_key is None:
                    raise ValueError(
                        "API key must be provided either as a parameter or through SYNTHEFY_API_KEY environment variable"
                    )
        else:
            # For non-synthefy domains, get API key from env if not provided but don't require it
            if api_key is not None:
                raise ValueError(
                    "API key must be set to None for non-synthefy domains"
                )

        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url
        self.endpoint = endpoint
        self.client = httpx.Client(base_url=self.base_url)
        self.organization = organization
        self.user_agent = (
            user_agent or f"synthefy-python httpx/{httpx.__version__}"
        )

    # Context manager support (sync) and utilities
    def __enter__(self) -> "SynthefyAPIClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def _headers(
        self,
        *,
        idempotency_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
        }
        if self.api_key is not None:
            headers["X-API-KEY"] = self.api_key
        if self.organization:
            headers["X-Organization"] = self.organization
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _should_retry(
        self, response: Optional[httpx.Response], exc: Optional[Exception]
    ) -> bool:
        if exc is not None:
            # Connection errors/timeouts are retryable
            return True
        if response is None:
            return False
        if (
            response.status_code in (408, 409, 425, 429)
            or 500 <= response.status_code <= 599
        ):
            return True
        return False

    def _compute_backoff(
        self, attempt: int, response: Optional[httpx.Response]
    ) -> float:
        if response is not None:
            retry_after = response.headers.get(
                "retry-after"
            ) or response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        # Exponential backoff with jitter
        base = min(2**attempt, 30)
        return base * (0.5 + 0.5 * (os.urandom(1)[0] / 255))

    def _post_with_retries(
        self,
        endpoint: str,
        json: Dict[str, Any],
        *,
        headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        last_exc: Optional[Exception] = None
        response: Optional[httpx.Response] = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self.client.post(
                    endpoint,
                    json=json,
                    headers=headers or self._headers(),
                    timeout=timeout or self.timeout,
                )
                if not self._should_retry(response, None):
                    _raise_for_status(response)
                    return response
            except httpx.TimeoutException as exc:
                last_exc = APITimeoutError(str(exc))
            except httpx.HTTPError as exc:
                last_exc = APIConnectionError(str(exc))

            # Decide to retry
            if attempt < attempts - 1 and self._should_retry(
                response, last_exc
            ):
                delay = self._compute_backoff(attempt, response)
                time.sleep(delay)
                continue

            # No more retries
            if last_exc is not None:
                raise last_exc
            if response is not None:
                _raise_for_status(response)
                return response

        # Should not reach here
        raise APIConnectionError("Request failed after retries")

    # TODO @Aditya -> should this be _forecast? does user ever call it?
    def forecast(
        self,
        request: ForecastV2Request,
        *,
        timeout: Optional[float] = None,
        idempotency_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> ForecastV2Response:
        response = self._post_with_retries(
            self.endpoint,
            json=request.model_dump(),
            headers=self._headers(
                idempotency_key=idempotency_key, extra_headers=extra_headers
            ),
            timeout=timeout,
        )
        response_data = response.json()
        return ForecastV2Response(**response_data)

    def forecast_dfs(
        self,
        history_dfs: List[pd.DataFrame],
        target_dfs: List[pd.DataFrame],
        target_col: str,
        timestamp_col: str,
        metadata_cols: List[str],
        leak_cols: List[str],
        model: str,
    ) -> List[pd.DataFrame]:
        """
        Make a forecasting request using pandas DataFrames.

        This is a convenience method that converts pandas DataFrames into
        the structured format required for forecasting, makes the API request,
        and returns the results as DataFrames for easy analysis.

        Parameters
        ----------
        history_dfs : List[pd.DataFrame]
            List of DataFrames containing historical data. Each DataFrame
            represents one forecasting scenario with past observations.
        target_dfs : List[pd.DataFrame]
            List of DataFrames containing target (future) data. Must have
            the same length as history_dfs. Target values should be NaN
            or None for the column being forecasted.
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
            Name of the model to use for forecasting. Common models include
            'synthefy-fm', 'sfm_moe', etc.

        Returns
        -------
        List[pd.DataFrame]
            List of DataFrames where each DataFrame contains forecast results
            for one scenario. Each DataFrame includes:
            #TODO @Aditya no more `split` column?
            - timestamps: Forecast timestamps
            - {target_col}: Forecasted values for the target column
            - {metadata_cols}: Metadata columns (unchanged)

        Raises
        ------
        ValueError
            If history_dfs and target_dfs have different lengths.
            If any DataFrame is missing required columns.
            If all DataFrames don't have consistent column structure.
            If leak_cols is not a subset of metadata_cols.
            If target_col or timestamp_col are in metadata_cols.
        BadRequestError
            If the request data is invalid (400, 422 status codes).
        AuthenticationError
            If the API key is invalid (401 status code).
        PermissionDeniedError
            If access is denied (403 status code).
        RateLimitError
            If rate limit is exceeded (429 status code).
        APITimeoutError
            If the request times out.
        APIConnectionError
            If there are network/connection issues.
        InternalServerError
            If the server encounters an error (5xx status codes).

        Notes
        -----
        - All DataFrames must have the same column structure
        - NaN values are automatically converted to None for JSON compatibility
        - Each DataFrame pair (history_df, target_df) creates one forecast scenario
        - Target column creates a forecast sample, metadata columns create metadata samples
        - Leak columns are marked with leak_target=True
        - The method automatically handles retries for transient errors

        Examples
        --------
        Basic forecasting with sales data:

        >>> # Create historical data
        >>> history_data = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        ...     'sales': np.random.normal(100, 10, 100),
        ...     'store_id': 1,
        ...     'category_id': 101,
        ...     'promotion_active': 0
        ... })
        >>>
        >>> # Create target data (values to forecast)
        >>> target_data = pd.DataFrame({
        ...     'date': pd.date_range('2024-04-11', periods=30, freq='D'),
        ...     'sales': np.nan,  # Values to forecast
        ...     'store_id': 1,
        ...     'category_id': 101,
        ...     'promotion_active': 1  # Promotion active in forecast period
        ... })
        >>>
        >>> # Make forecast
        >>> with SynthefyAPIClient() as client:
        ...     forecast_dfs = client.forecast_dfs(
        ...         history_dfs=[history_df],
        ...         target_dfs=[target_df],
        ...         target_col='sales',
        ...         timestamp_col='date',
        ...         metadata_cols=['store_id', 'category_id', 'promotion_active'],
        ...         leak_cols=[],
        ...         model='synthefy-fm'
        ...     )
        >>>
        >>> # Access forecast results
        >>> forecast_df = forecast_dfs[0]
        >>> print(forecast_df[['date', 'sales']].head())

        Multiple scenarios with different stores:

        >>> # Create multiple scenarios
        >>> scenarios_history = []
        >>> scenarios_target = []
        >>>
        >>> for store_id in [1, 2, 3]:
        ...     # Historical data for each store
        ...     hist_data = {
        ...         'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        ...         'sales': np.random.normal(100 + store_id * 10, 10, 100),
        ...         'store_id': store_id,
        ...         'category_id': 101,
        ...         'promotion_active': 0
        ...     }
        ...     scenarios_history.append(pd.DataFrame(hist_data))
        ...
        ...     # Target data for each store
        ...     target_data = {
        ...         'date': pd.date_range('2024-04-11', periods=30, freq='D'),
        ...         'sales': np.nan,
        ...         'store_id': store_id,
        ...         'category_id': 101,
        ...         'promotion_active': 1
        ...     }
        ...     scenarios_target.append(pd.DataFrame(target_data))
        >>>
        >>> # Forecast all scenarios
        >>> with SynthefyAPIClient() as client:
        ...     forecast_dfs = client.forecast_dfs(
        ...         history_dfs=scenarios_history,
        ...         target_dfs=scenarios_target,
        ...         target_col='sales',
        ...         timestamp_col='date',
        ...         metadata_cols=['store_id', 'category_id', 'promotion_active'],
        ...         leak_cols=[],
        ...         model='synthefy-fm'
        ...     )
        >>>
        >>> # Process results for each scenario
        >>> for i, forecast_df in enumerate(forecast_dfs):
        ...     print(f"Store {i+1} forecast: {len(forecast_df)} predictions")

        Using leak columns (columns with target leakage):

        >>> # Historical data with future information
        >>> history_data = {
        ...     'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        ...     'sales': np.random.normal(100, 10, 100),
        ...     'future_promotion': np.random.choice([0, 1], 100)  # Future promotion info
        ... }
        >>> history_df = pd.DataFrame(history_data)
        >>>
        >>> target_data = {
        ...     'date': pd.date_range('2024-04-11', periods=30, freq='D'),
        ...     'sales': np.nan,
        ...     'future_promotion': np.random.choice([0, 1], 30)  # Known future promotions
        ... }
        >>> target_df = pd.DataFrame(target_data)
        >>>
        >>> with SynthefyAPIClient() as client:
        ...     forecast_dfs = client.forecast_dfs(
        ...         history_dfs=[history_df],
        ...         target_dfs=[target_df],
        ...         target_col='sales',
        ...         timestamp_col='date',
        ...         metadata_cols=['future_promotion'],
        ...         leak_cols=['future_promotion'],  # Mark as leak column
        ...         model='synthefy-fm'
        ...     )

        Error handling:

        >>> try:
        ...     with SynthefyAPIClient() as client:
        ...         forecast_dfs = client.forecast_dfs(
        ...             history_dfs=[history_df],
        ...             target_dfs=[target_df],
        ...             target_col='sales',
        ...             timestamp_col='date',
        ...             metadata_cols=['store_id'],
        ...             leak_cols=[],
        ...             model='synthefy-fm'
        ...         )
        ... except BadRequestError as e:
        ...     print(f"Invalid request: {e}")
        ...     print(f"Status code: {e.status_code}")
        ... except RateLimitError as e:
        ...     print(f"Rate limited: {e}")
        ...     # Client automatically retries with exponential backoff
        ... except APITimeoutError as e:
        ...     print(f"Request timed out: {e}")

        See Also
        --------
        forecast : Make a direct forecast request with ForecastV2Request
        """
        request = ForecastV2Request.from_dfs(
            history_dfs,
            target_dfs,
            target_col,
            timestamp_col,
            metadata_cols,
            leak_cols,
            model,
        )

        response = self.forecast(request)

        return response.to_dfs()


class SynthefyAsyncAPIClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 300.0,
        max_retries: int = 2,
        base_url: str = BASE_URL,
        endpoint: str = ENDPOINT,
        organization: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        # Only require API key for synthefy.com domains
        if _is_synthefy_domain(base_url):
            if api_key is None:
                api_key = os.getenv("SYNTHEFY_API_KEY")
                if api_key is None:
                    raise ValueError(
                        "API key must be provided either as a parameter or through SYNTHEFY_API_KEY environment variable"
                    )
        else:
            # For non-synthefy domains, get API key from env if not provided but don't require it
            if api_key is not None:
                raise ValueError(
                    "API key must be set to None for non-synthefy domains"
                )

        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url
        self.endpoint = endpoint
        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout
        )
        self.organization = organization
        self.user_agent = (
            user_agent or f"synthefy-python httpx/{httpx.__version__}"
        )

    async def __aenter__(self) -> "SynthefyAsyncAPIClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        try:
            await self.client.aclose()
        except Exception:
            pass

    def _headers(
        self,
        *,
        idempotency_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
        }
        if self.api_key is not None:
            headers["X-API-KEY"] = self.api_key
        if self.organization:
            headers["X-Organization"] = self.organization
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _should_retry(
        self, response: Optional[httpx.Response], exc: Optional[Exception]
    ) -> bool:
        if exc is not None:
            return True
        if response is None:
            return False
        if (
            response.status_code in (408, 409, 425, 429)
            or 500 <= response.status_code <= 599
        ):
            return True
        return False

    def _compute_backoff(
        self, attempt: int, response: Optional[httpx.Response]
    ) -> float:
        if response is not None:
            retry_after = response.headers.get(
                "retry-after"
            ) or response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        base = min(2**attempt, 30)
        return base * (0.5 + 0.5 * (os.urandom(1)[0] / 255))

    async def _post_with_retries(
        self,
        endpoint: str,
        json: Dict[str, Any],
        *,
        headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        last_exc: Optional[Exception] = None
        response: Optional[httpx.Response] = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = await self.client.post(
                    endpoint,
                    json=json,
                    headers=headers or self._headers(),
                    timeout=timeout or self.timeout,
                )
                if not self._should_retry(response, None):
                    _raise_for_status(response)
                    return response
            except httpx.TimeoutException as exc:
                last_exc = APITimeoutError(str(exc))
            except httpx.HTTPError as exc:
                last_exc = APIConnectionError(str(exc))

            if attempt < attempts - 1 and self._should_retry(
                response, last_exc
            ):
                delay = self._compute_backoff(attempt, response)
                await asyncio.sleep(delay)
                continue

            if last_exc is not None:
                raise last_exc
            if response is not None:
                _raise_for_status(response)
                return response

        raise APIConnectionError("Request failed after retries")

    async def forecast(
        self,
        request: ForecastV2Request,
        *,
        timeout: Optional[float] = None,
        idempotency_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> ForecastV2Response:
        response = await self._post_with_retries(
            self.endpoint,
            json=request.model_dump(),
            headers=self._headers(
                idempotency_key=idempotency_key, extra_headers=extra_headers
            ),
            timeout=timeout,
        )
        response_data = response.json()
        return ForecastV2Response(**response_data)

    async def forecast_dfs(
        self,
        history_dfs: List[pd.DataFrame],
        target_dfs: List[pd.DataFrame],
        target_col: str,
        timestamp_col: str,
        metadata_cols: List[str],
        leak_cols: List[str],
        model: str,
    ) -> List[pd.DataFrame]:
        """
        Make a forecasting request using pandas DataFrames.

        This is a convenience method that converts pandas DataFrames into
        the structured format required for forecasting, makes the API request,
        and returns the results as DataFrames for easy analysis.

        Parameters
        ----------
        history_dfs : List[pd.DataFrame]
            List of DataFrames containing historical data. Each DataFrame
            represents one forecasting scenario with past observations.
        target_dfs : List[pd.DataFrame]
            List of DataFrames containing target (future) data. Must have
            the same length as history_dfs. Target values should be NaN
            or None for the column being forecasted.
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
            Name of the model to use for forecasting. Common models include
            'synthefy-fm', 'sfm_moe', etc.

        Returns
        -------
        List[pd.DataFrame]
            List of DataFrames where each DataFrame contains forecast results
            for one scenario. Each DataFrame includes:
            #TODO @Aditya no more `split` column?
            - timestamps: Forecast timestamps
            - {target_col}: Forecasted values for the target column
            - {metadata_cols}: Metadata columns (unchanged)

        Raises
        ------
        ValueError
            If history_dfs and target_dfs have different lengths.
            If any DataFrame is missing required columns.
            If all DataFrames don't have consistent column structure.
            If leak_cols is not a subset of metadata_cols.
            If target_col or timestamp_col are in metadata_cols.
        BadRequestError
            If the request data is invalid (400, 422 status codes).
        AuthenticationError
            If the API key is invalid (401 status code).
        PermissionDeniedError
            If access is denied (403 status code).
        RateLimitError
            If rate limit is exceeded (429 status code).
        APITimeoutError
            If the request times out.
        APIConnectionError
            If there are network/connection issues.
        InternalServerError
            If the server encounters an error (5xx status codes).

        Notes
        -----
        - All DataFrames must have the same column structure
        - NaN values are automatically converted to None for JSON compatibility
        - Each DataFrame pair (history_df, target_df) creates one forecast scenario
        - Target column creates a forecast sample, metadata columns create metadata samples
        - Leak columns are marked with leak_target=True
        - The method automatically handles retries for transient errors
        - Use this method for concurrent requests and non-blocking operations

        Examples
        --------
        Basic async forecasting:

        >>> # Create historical data
        >>> history_data = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        ...     'sales': np.random.normal(100, 10, 100),
        ...     'store_id': 1,
        ...     'category_id': 101,
        ...     'promotion_active': 0
        ... })
        >>>
        >>> # Create target data (values to forecast)
        >>> target_data = pd.DataFrame({
        ...     'date': pd.date_range('2024-04-11', periods=30, freq='D'),
        ...     'sales': np.nan,  # Values to forecast
        ...     'store_id': 1,
        ...     'category_id': 101,
        ...     'promotion_active': 1  # Promotion active in forecast period
        ... })
        >>>
        >>> # Make async forecast
        >>> async with SynthefyAsyncAPIClient() as client:
        ...     forecast_dfs = await client.forecast_dfs(
        ...         history_dfs=[history_data],
        ...         target_dfs=[target_data],
        ...         target_col='sales',
        ...         timestamp_col='date',
        ...         metadata_cols=['store_id', 'category_id', 'promotion_active'],
        ...         leak_cols=[],
        ...         model='synthefy-fm'
        ...     )
        >>>
        >>> # Access forecast results
        >>> forecast_df = forecast_dfs[0]
        >>> print(forecast_df[['date', 'sales']].head())

        Concurrent forecasts for multiple datasets:

        >>> # Create multiple scenarios
        >>> scenarios_history = []
        >>> scenarios_target = []
        >>>
        >>> for store_id in [1, 2, 3]:
        ...     # Historical data for each store
        ...     hist_data = {
        ...         'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        ...         'sales': np.random.normal(100 + store_id * 10, 10, 100),
        ...         'store_id': store_id,
        ...         'category_id': 101,
        ...         'promotion_active': 0
        ...     }
        ...     scenarios_history.append(pd.DataFrame(hist_data))
        ...
        ...     # Target data for each store
        ...     target_data = {
        ...         'date': pd.date_range('2024-04-11', periods=30, freq='D'),
        ...         'sales': np.nan,
        ...         'store_id': store_id,
        ...         'category_id': 101,
        ...         'promotion_active': 1
        ...     }
        ...     scenarios_target.append(pd.DataFrame(target_data))
        >>>
        >>> # Forecast all scenarios concurrently
        >>> async with SynthefyAsyncAPIClient() as client:
        ...     tasks = []
        ...     for i in range(3):
        ...         # Create variations of your data
        ...         modified_history = scenarios_history[i].copy()
        ...         modified_target = scenarios_target[i].copy()
        ...
        ...         task = client.forecast_dfs(
        ...             history_dfs=[modified_history],
        ...             target_dfs=[modified_target],
        ...             target_col='sales',
        ...             timestamp_col='date',
        ...             metadata_cols=['store_id', 'category_id', 'promotion_active'],
        ...             leak_cols=[],
        ...             model='synthefy-fm'
        ...         )
        ...         tasks.append(task)
        ...
        ...     # Execute all forecasts concurrently
        ...     results = await asyncio.gather(*tasks)
        ...
        ...     for i, forecast_dfs in enumerate(results):
        ...         print(f"Store {i+1} forecast: {len(forecast_dfs[0])} predictions")

        Using leak columns (columns with target leakage):

        >>> # Historical data with future information
        >>> history_data = {
        ...     'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        ...     'sales': np.random.normal(100, 10, 100),
        ...     'future_promotion': np.random.choice([0, 1], 100)  # Future promotion info
        ... }
        >>> history_df = pd.DataFrame(history_data)
        >>>
        >>> target_data = {
        ...     'date': pd.date_range('2024-04-11', periods=30, freq='D'),
        ...     'sales': np.nan,
        ...     'future_promotion': np.random.choice([0, 1], 30)  # Known future promotions
        ... }
        >>> target_df = pd.DataFrame(target_data)
        >>>
        >>> async with SynthefyAsyncAPIClient() as client:
        ...     forecast_dfs = await client.forecast_dfs(
        ...         history_dfs=[history_df],
        ...         target_dfs=[target_df],
        ...         target_col='sales',
        ...         timestamp_col='date',
        ...         metadata_cols=['future_promotion'],
        ...         leak_cols=['future_promotion'],  # Mark as leak column
        ...         model='synthefy-fm'
        ...     )

        Error handling with async:

        >>> try:
        ...     async with SynthefyAsyncAPIClient() as client:
        ...         forecast_dfs = await client.forecast_dfs(
        ...             history_dfs=[history_df],
        ...             target_dfs=[target_df],
        ...             target_col='sales',
        ...             timestamp_col='date',
        ...             metadata_cols=['store_id'],
        ...             leak_cols=[],
        ...             model='synthefy-fm'
        ...         )
        ... except BadRequestError as e:
        ...     print(f"Invalid request: {e}")
        ...     print(f"Status code: {e.status_code}")
        ... except RateLimitError as e:
        ...     print(f"Rate limited: {e}")
        ...     # Client automatically retries with exponential backoff
        ... except APITimeoutError as e:
        ...     print(f"Request timed out: {e}")

        See Also
        --------
        forecast : Make a direct async forecast request with ForecastV2Request
        """
        request = ForecastV2Request.from_dfs(
            history_dfs,
            target_dfs,
            target_col,
            timestamp_col,
            metadata_cols,
            leak_cols,
            model,
        )
        response = await self.forecast(request)
        return response.to_dfs()
