"""Transport-independent request building, retry policy and error mapping (shared by the sync
and async clients)."""

from __future__ import annotations

import email.utils
import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Mapping, Optional, Tuple, TypeVar, Union
from urllib.parse import quote

import httpx

from ._errors import (
    APIConnectionError,
    APIResponseError,
    APIStatusError,
    APITimeoutError,
    BreadBowlError,
    error_class_for,
)
from ._generated import DEFAULT_BASE_URL, OPERATIONS, Operation
from ._models import ResponseMeta
from ._version import __version__

USER_AGENT = f"breadbowl-python/{__version__}"
DEFAULT_TIMEOUT = 60.0
# The contract: upserts run at about 2.5-3.5 chunks/s on the hosted CPU, so a request at the
# chunk limit takes 40-60 s (longer when queued); it recommends client timeouts of at least 120 s.
DEFAULT_UPSERT_TIMEOUT = 180.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_RETRY_WAIT = 60.0
INITIAL_BACKOFF = 0.5
MAX_BACKOFF = 8.0

# How an operation may be retried.
#   probe: health checks report the current state; never retried.
#   retry: everything else, following the contract's guidance. Mutations carry one
#          Idempotency-Key for all attempts; encode/score/search store nothing and only 2xx
#          responses are billed, so a failed attempt can be retried without double billing.
RetryKind = str


def retry_kind(operation_id: str, op: Operation) -> RetryKind:
    return "retry" if op.authenticated else "probe"


T = TypeVar("T")


@dataclass
class Call(Generic[T]):
    """One API call: which operation, its parameters and how to parse a 2xx body."""

    operation_id: str
    parse: Callable[[Any, ResponseMeta], T]
    path_params: Dict[str, str] = field(default_factory=dict)
    query: Optional[Dict[str, Any]] = None
    body: Any = None
    idempotency_key: Optional[str] = None
    # Non-2xx statuses whose body is parsed as a result instead of raised (readyz's 503).
    accept_statuses: Tuple[int, ...] = ()
    # Per-call timeout overriding the client's (upserts are slow).
    timeout: Union[float, httpx.Timeout, None] = None

    @property
    def operation(self) -> Operation:
        return OPERATIONS[self.operation_id]


def compact(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop ``None`` values: the contract rejects unknown or null fields, so omitted means default."""
    return {k: v for k, v in values.items() if v is not None}


def query_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def require_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        seconds = when.timestamp() - time.time()
    return max(0.0, seconds)


class BaseClient:
    def __init__(
        self,
        api_key: Optional[str],
        *,
        base_url: Optional[str],
        timeout: Union[float, httpx.Timeout, None],
        max_retries: int,
        max_retry_wait: float,
        default_headers: Optional[Mapping[str, str]],
        upsert_timeout: Union[float, httpx.Timeout, None] = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("BREADBOWL_API_KEY")
        if not key:
            raise BreadBowlError(
                "No API key. Pass api_key=... or set the BREADBOWL_API_KEY environment variable."
            )
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.api_key = key
        self.base_url = (base_url or os.environ.get("BREADBOWL_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout: Union[float, httpx.Timeout, None] = DEFAULT_TIMEOUT if timeout is None else timeout
        self.upsert_timeout: Union[float, httpx.Timeout, None] = (
            DEFAULT_UPSERT_TIMEOUT if upsert_timeout is None else upsert_timeout
        )
        self.max_retries = max_retries
        self.max_retry_wait = max_retry_wait
        self.default_headers: Dict[str, str] = dict(default_headers or {})

    # ------------------------------------------------------------------ request building
    def _url(self, op: Operation, path_params: Mapping[str, str]) -> str:
        path = op.path
        for name, value in path_params.items():
            # One path segment each: "/" in a doc_id is sent as %2F.
            path = path.replace("{" + name + "}", quote(value, safe=""))
        return self.base_url + path

    def _headers(self, call: Call[Any]) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            **self.default_headers,
        }
        if call.operation.authenticated:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if call.body is not None:
            headers["Content-Type"] = "application/json"
        if call.idempotency_key is not None:
            headers["Idempotency-Key"] = call.idempotency_key
        return headers

    @staticmethod
    def _content(call: Call[Any]) -> Optional[bytes]:
        if call.body is None:
            return None
        # allow_nan=False: metadata numbers must be finite; fail here rather than with a 400.
        return json.dumps(call.body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )

    @staticmethod
    def new_idempotency_key() -> str:
        return str(uuid.uuid4())

    # ------------------------------------------------------------------ responses
    @staticmethod
    def _meta(response: httpx.Response, call: Call[Any]) -> ResponseMeta:
        return ResponseMeta(
            request_id=response.headers.get("x-request-id"),
            status_code=response.status_code,
            headers={k.lower(): v for k, v in response.headers.items()},
            idempotency_key=call.idempotency_key,
        )

    def _parse_success(self, response: httpx.Response, call: Call[T]) -> T:
        meta = self._meta(response, call)
        try:
            data = response.json()
        except ValueError as exc:
            raise APIResponseError(
                f"expected a JSON body from {call.operation_id}, got {response.text[:200]!r}",
                status_code=response.status_code,
                request_id=meta.request_id,
                body=response.text,
            ) from exc
        if not isinstance(data, dict):
            raise APIResponseError(
                f"expected a JSON object from {call.operation_id}",
                status_code=response.status_code,
                request_id=meta.request_id,
                body=data,
            )
        return call.parse(data, meta)

    @staticmethod
    def _make_error(response: httpx.Response) -> APIStatusError:
        request_id = response.headers.get("x-request-id")
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text
        code: Optional[str] = None
        message = response.reason_phrase or f"HTTP {response.status_code}"
        details: Optional[Dict[str, Any]] = None
        issues = None
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            err = body["error"]
            code = err.get("code") if isinstance(err.get("code"), str) else None
            message = str(err.get("message") or message)
            details = err.get("details") if isinstance(err.get("details"), dict) else None
            issues = err.get("issues") if isinstance(err.get("issues"), list) else None
        elif isinstance(body, str) and body.strip():
            message = body.strip()[:500]
        cls = error_class_for(response.status_code, code)
        return cls(
            message,
            status_code=response.status_code,
            code=code,
            request_id=request_id,
            details=details,
            issues=issues,
            retry_after=parse_retry_after(response.headers.get("retry-after")),
            headers={k.lower(): v for k, v in response.headers.items()},
            body=body,
        )

    # ------------------------------------------------------------------ retries
    def _backoff(self, attempt: int) -> float:
        base: float = min(MAX_BACKOFF, INITIAL_BACKOFF * (2.0**attempt))
        return base * (0.5 + random.random() / 2)

    def _retry_delay(
        self,
        kind: RetryKind,
        attempt: int,
        *,
        error: Optional[APIStatusError] = None,
        exc: Optional[Exception] = None,
    ) -> Optional[float]:
        """Seconds to wait before the next attempt, or ``None`` to give up."""
        if kind == "probe" or attempt >= self.max_retries:
            return None
        if exc is not None:
            return self._backoff(attempt)  # timeouts and connection failures
        assert error is not None
        status, code, retry_after = error.status_code, error.code, error.retry_after
        if status == 429:
            retryable = code != "usage_limit_exceeded"  # that one resets next month
        elif status == 409:
            retryable = code == "idempotency_request_in_progress" or (
                code == "index_not_ready" and error.details.get("status") == "creating"
            )
        else:
            retryable = status in (500, 502, 503, 504)
        if not retryable:
            return None
        if retry_after is not None:
            return retry_after if retry_after <= self.max_retry_wait else None
        return self._backoff(attempt)

    def _connection_error(self, exc: Exception, request: Optional[httpx.Request]) -> APIConnectionError:
        if isinstance(exc, httpx.TimeoutException):
            return APITimeoutError(f"request timed out: {exc}", request=request)
        return APIConnectionError(f"connection failed: {exc}", request=request)
