"""Shared Synthefy SDK exceptions and HTTP status mapping."""

from typing import Any, Dict, Optional, Tuple

import httpx


class SynthefyError(Exception):
    """Base error for all Synthefy client exceptions."""


class APITimeoutError(SynthefyError):
    """The request timed out before completing."""


class APIConnectionError(SynthefyError):
    """The request failed due to a connection issue."""


class SageMakerInvocationError(SynthefyError):
    """A queued SageMaker inference wrote a failure object instead of predictions."""

    def __init__(
        self,
        message: str,
        *,
        invocation_id: str,
        failure_location: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.invocation_id = invocation_id
        self.failure_location = failure_location


class FineTuningError(SynthefyError):
    """A fine-tuning job, or the deployment of its result, cannot give the caller a model.

    Raised for a job that failed or produced nothing to ship, a missing training report, a job
    SageMaker started whose first status read failed, and a deployment that failed part way.
    ``job_arn`` is set when SageMaker accepted the job, so the caller still holds its receipt.
    ``created`` maps what a deployment had already made (model package ARN, SageMaker model,
    endpoint config, endpoint) to its name, so the caller can delete those resources or retry
    the deployment instead of hunting for them.
    """

    def __init__(
        self,
        message: str,
        *,
        job_name: Optional[str] = None,
        job_arn: Optional[str] = None,
        status: Optional[str] = None,
        failure_reason: Optional[str] = None,
        created: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.job_name = job_name
        self.job_arn = job_arn
        self.status = status
        self.failure_reason = failure_reason
        self.created: Dict[str, str] = dict(created or {})


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
    request_id = response.headers.get("x-request-id") or response.headers.get("X-Request-Id")
    parsed: Any
    message: str = f"HTTP {response.status_code} Error"
    code: Optional[str] = None

    try:
        parsed = response.json()
        # Common error shapes: {"error": {"message": str, "type"/"code": str}}, {"message": str}
        if isinstance(parsed, dict):
            error_obj: Any = parsed.get("error")
            if isinstance(error_obj, dict):
                message = error_obj.get("message") or error_obj.get("detail") or error_obj.get("error") or message
                code = error_obj.get("code") or error_obj.get("type")
                request_id = request_id or error_obj.get("request_id")
            else:
                message = parsed.get("message") or parsed.get("detail") or parsed.get("error") or message
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
