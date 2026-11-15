"""Typed exceptions mapped from the contract's ``{error: {code, message}}`` envelope."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar, cast

import httpx

from ._generated import ERROR_DETAILS

_T = TypeVar("_T")

__all__ = [
    "BreadBowlError",
    "TensorDecodeError",
    "APIConnectionError",
    "APITimeoutError",
    "APIResponseError",
    "APIStatusError",
    "BadRequestError",
    "AuthenticationError",
    "ApiKeyExpiredError",
    "PermissionDeniedError",
    "TenantSuspendedError",
    "InsufficientScopeError",
    "NotFoundError",
    "ConflictError",
    "IndexNameConflictError",
    "IndexNotReadyError",
    "ModelRevisionUnavailableError",
    "RepresentationUnavailableError",
    "IdempotencyRequestInProgressError",
    "PayloadTooLargeError",
    "UnsupportedMediaTypeError",
    "UnprocessableEntityError",
    "RequestValidationError",
    "InvalidCursorError",
    "UnsupportedModelError",
    "UnsupportedSearchProfileError",
    "UnsupportedFilterError",
    "AttentionLimitExceededError",
    "DocumentLimitExceededError",
    "ChunkLimitExceededError",
    "IdempotencyKeyReusedError",
    "RateLimitError",
    "RateLimitExceededError",
    "ConcurrencyLimitExceededError",
    "UsageLimitExceededError",
    "InternalServerError",
    "ServiceUnavailableError",
    "error_class_for",
]


class BreadBowlError(Exception):
    """Base class of every exception the SDK raises."""


class TensorDecodeError(BreadBowlError, ValueError):
    """A tensor's data does not match its ``dtype``/``shape``/``encoding``."""


class APIConnectionError(BreadBowlError):
    """The request never produced an HTTP response (DNS, TCP, TLS, connection reset)."""

    def __init__(self, message: str, *, request: Optional[httpx.Request] = None) -> None:
        super().__init__(message)
        self.request = request


class APITimeoutError(APIConnectionError):
    """The request timed out."""


class APIResponseError(BreadBowlError):
    """A 2xx response whose body is not the JSON the contract promises."""

    def __init__(
        self, message: str, *, status_code: int, request_id: Optional[str], body: Any = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.body = body


class APIStatusError(BreadBowlError):
    """A non-2xx response. ``code`` is the stable machine-readable error code.

    Subclasses exist per status and per known code; an unknown code maps to its status class,
    as the contract requires.
    """

    status_code: int = 0

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: Optional[str],
        request_id: Optional[str],
        details: Optional[Dict[str, Any]] = None,
        issues: Optional[List[Dict[str, Any]]] = None,
        retry_after: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Any = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.details: Dict[str, Any] = details or {}
        self.issues: List[Dict[str, Any]] = issues or []
        self.retry_after = retry_after
        self.headers: Mapping[str, str] = headers or {}
        self.body = body
        rid = f" (request id {request_id})" if request_id else ""
        super().__init__(f"{status_code} {code or 'error'}: {message}{rid}")

    def details_for(self, code: str) -> Optional[Dict[str, Any]]:
        """``details`` if this error has ``code`` and the contract defines details for it
        (``x-error-details``); ``breadbowl.types`` has a TypedDict for each shape."""
        if self.code != code or code not in ERROR_DETAILS or not self.details:
            return None
        return self.details

    def _detail(self, name: str, kind: Type[_T]) -> Optional[_T]:
        value = self.details.get(name)
        if isinstance(value, kind) and not isinstance(value, bool):
            return value
        return None

    def _str_list(self, name: str) -> Optional[List[str]]:
        value = self.details.get(name)
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return cast(List[str], value)
        return None


class BadRequestError(APIStatusError):
    status_code = 400


class AuthenticationError(APIStatusError):
    status_code = 401


class ApiKeyExpiredError(AuthenticationError):
    pass


class PermissionDeniedError(APIStatusError):
    status_code = 403


class TenantSuspendedError(PermissionDeniedError):
    pass


class InsufficientScopeError(PermissionDeniedError):
    @property
    def required_scope(self) -> Optional[str]:
        return self._detail("required_scope", str)


class NotFoundError(APIStatusError):
    status_code = 404


class ConflictError(APIStatusError):
    status_code = 409


class IndexNameConflictError(ConflictError):
    @property
    def index_id(self) -> Optional[str]:
        """The existing index with that name."""
        return self._detail("index_id", str)


class IndexNotReadyError(ConflictError):
    @property
    def index_status(self) -> Optional[str]:
        """``creating`` (retried automatically) or ``error`` (delete and recreate)."""
        return self._detail("status", str)


class ModelRevisionUnavailableError(ConflictError):
    @property
    def model(self) -> Optional[Dict[str, str]]:
        """``{"id", "revision"}`` of the model the index is pinned to."""
        value = self.details.get("model")
        return cast(Dict[str, str], value) if isinstance(value, dict) else None


class RepresentationUnavailableError(ConflictError):
    """The pinned model revision is served, but with another representation (for example another
    compute dtype). Not retried: it resolves when the deployment serves the pinned representation
    again; otherwise create a new index and re-ingest."""

    @property
    def representation(self) -> Optional[Dict[str, Any]]:
        """The representation the index is pinned to."""
        value = self.details.get("representation")
        return cast(Dict[str, Any], value) if isinstance(value, dict) else None

    @property
    def served(self) -> Optional[Dict[str, Any]]:
        """What the deployment serves now (``None`` while it cannot tell)."""
        value = self.details.get("served")
        return cast(Dict[str, Any], value) if isinstance(value, dict) else None


class IdempotencyRequestInProgressError(ConflictError):
    pass


class PayloadTooLargeError(APIStatusError):
    status_code = 413


class UnsupportedMediaTypeError(APIStatusError):
    status_code = 415


class UnprocessableEntityError(APIStatusError):
    status_code = 422


class RequestValidationError(UnprocessableEntityError):
    """``validation_error``: see ``issues`` for each problem (``path``, ``code``, ``message``)."""


class InvalidCursorError(UnprocessableEntityError):
    pass


class UnsupportedModelError(UnprocessableEntityError):
    @property
    def served(self) -> Optional[List[str]]:
        """Model ids and aliases this deployment serves."""
        return self._str_list("served")


class UnsupportedSearchProfileError(UnprocessableEntityError):
    @property
    def search_profiles(self) -> Optional[List[str]]:
        """Profile names the model offers."""
        return self._str_list("search_profiles")


class UnsupportedFilterError(UnprocessableEntityError):
    @property
    def field(self) -> Optional[str]:
        return self._detail("field", str)

    @property
    def supported_operations(self) -> Optional[List[str]]:
        return self._str_list("supported_operations")

    @property
    def declared_type(self) -> Optional[str]:
        return self._detail("declared_type", str)


class AttentionLimitExceededError(UnprocessableEntityError):
    @property
    def requested_values(self) -> Optional[int]:
        return self._detail("requested_values", int)

    @property
    def limit(self) -> Optional[int]:
        return self._detail("limit", int)


class DocumentLimitExceededError(UnprocessableEntityError):
    @property
    def limit(self) -> Optional[int]:
        return self._detail("limit", int)

    @property
    def current(self) -> Optional[int]:
        return self._detail("current", int)

    @property
    def new_documents(self) -> Optional[int]:
        return self._detail("new_documents", int)


class ChunkLimitExceededError(UnprocessableEntityError):
    """The documents of one upsert chunk into more than ``limits.max_chunks_per_upsert`` chunks.
    ``documents.upsert_batched`` splits the batch and retries automatically."""

    @property
    def limit(self) -> Optional[int]:
        return self._detail("limit", int)

    @property
    def requested(self) -> Optional[int]:
        return self._detail("requested", int)


class IdempotencyKeyReusedError(UnprocessableEntityError):
    pass


class RateLimitError(APIStatusError):
    status_code = 429


class RateLimitExceededError(RateLimitError):
    pass


class ConcurrencyLimitExceededError(RateLimitError):
    pass


class UsageLimitExceededError(RateLimitError):
    """The monthly input-token allowance is exhausted; ``retry_after`` points at next month."""


class InternalServerError(APIStatusError):
    """500 and any other 5xx except 503."""

    status_code = 500


class ServiceUnavailableError(APIStatusError):
    status_code = 503


_BY_CODE: Dict[str, Type[APIStatusError]] = {
    "bad_request": BadRequestError,
    "authentication_failed": AuthenticationError,
    "api_key_expired": ApiKeyExpiredError,
    "tenant_suspended": TenantSuspendedError,
    "insufficient_scope": InsufficientScopeError,
    "not_found": NotFoundError,
    "index_name_conflict": IndexNameConflictError,
    "index_not_ready": IndexNotReadyError,
    "model_revision_unavailable": ModelRevisionUnavailableError,
    "representation_unavailable": RepresentationUnavailableError,
    "idempotency_request_in_progress": IdempotencyRequestInProgressError,
    "payload_too_large": PayloadTooLargeError,
    "unsupported_media_type": UnsupportedMediaTypeError,
    "validation_error": RequestValidationError,
    "invalid_cursor": InvalidCursorError,
    "unsupported_model": UnsupportedModelError,
    "unsupported_search_profile": UnsupportedSearchProfileError,
    "unsupported_filter": UnsupportedFilterError,
    "attention_limit_exceeded": AttentionLimitExceededError,
    "document_limit_exceeded": DocumentLimitExceededError,
    "chunk_limit_exceeded": ChunkLimitExceededError,
    "idempotency_key_reused": IdempotencyKeyReusedError,
    "rate_limit_exceeded": RateLimitExceededError,
    "concurrency_limit_exceeded": ConcurrencyLimitExceededError,
    "usage_limit_exceeded": UsageLimitExceededError,
    "internal_error": InternalServerError,
    "service_unavailable": ServiceUnavailableError,
}

_BY_STATUS: Dict[int, Type[APIStatusError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    413: PayloadTooLargeError,
    415: UnsupportedMediaTypeError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    503: ServiceUnavailableError,
}


def _status_class(status: int) -> Type[APIStatusError]:
    if status in _BY_STATUS:
        return _BY_STATUS[status]
    if status >= 500:
        return InternalServerError
    return APIStatusError


def error_class_for(status: int, code: Optional[str]) -> Type[APIStatusError]:
    """The exception class for a status and code. A known code only wins when its status
    matches; anything else is handled by status, as the contract requires."""
    base = _status_class(status)
    specific = _BY_CODE.get(code or "")
    if specific is not None and issubclass(specific, base):
        return specific
    return base
