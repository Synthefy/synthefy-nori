"""Shared Synthefy SDK exceptions and HTTP status mapping."""

from typing import Any, Optional, Tuple

import httpx


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
