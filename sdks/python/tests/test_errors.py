from __future__ import annotations

import pytest

import breadbowl
from breadbowl import (
    APIResponseError,
    APIStatusError,
    BreadBowl,
    InsufficientScopeError,
    InternalServerError,
    UnprocessableEntityError,
)
from breadbowl._errors import error_class_for
from tests.conftest import FIXTURES, MockAPI

EXPECTED = {
    "bad_request": "BadRequestError",
    "authentication_failed": "AuthenticationError",
    "api_key_expired": "ApiKeyExpiredError",
    "tenant_suspended": "TenantSuspendedError",
    "insufficient_scope": "InsufficientScopeError",
    "not_found": "NotFoundError",
    "index_name_conflict": "IndexNameConflictError",
    "index_not_ready": "IndexNotReadyError",
    "model_revision_unavailable": "ModelRevisionUnavailableError",
    "representation_unavailable": "RepresentationUnavailableError",
    "idempotency_request_in_progress": "IdempotencyRequestInProgressError",
    "payload_too_large": "PayloadTooLargeError",
    "unsupported_media_type": "UnsupportedMediaTypeError",
    "validation_error": "RequestValidationError",
    "invalid_cursor": "InvalidCursorError",
    "unsupported_model": "UnsupportedModelError",
    "unsupported_search_profile": "UnsupportedSearchProfileError",
    "unsupported_filter": "UnsupportedFilterError",
    "attention_limit_exceeded": "AttentionLimitExceededError",
    "document_limit_exceeded": "DocumentLimitExceededError",
    "chunk_limit_exceeded": "ChunkLimitExceededError",
    "index_chunk_limit_exceeded": "IndexChunkLimitExceededError",
    "idempotency_key_reused": "IdempotencyKeyReusedError",
    "rate_limit_exceeded": "RateLimitExceededError",
    "concurrency_limit_exceeded": "ConcurrencyLimitExceededError",
    "usage_limit_exceeded": "UsageLimitExceededError",
    "internal_error": "InternalServerError",
    "service_unavailable": "ServiceUnavailableError",
}


@pytest.mark.parametrize("case", FIXTURES["errors"], ids=lambda e: f"{e['status']}-{e['code']}")
def test_contract_error_examples_map_to_typed_errors(case: dict, api: MockAPI) -> None:
    client = BreadBowl("k", base_url="https://api.test", max_retries=0, http_client=_http(api))
    headers = {"x-request-id": "req-err-1"}
    if case["status"] in (429,):
        headers["Retry-After"] = "12"
    api.add(case["body"], case["status"], headers=headers)
    with pytest.raises(APIStatusError) as info:
        client.usage.get()
    err = info.value
    assert type(err).__name__ == EXPECTED[case["code"]]
    assert err.status_code == case["status"] and err.code == case["code"]
    assert err.message == case["body"]["error"]["message"]
    assert err.request_id == "req-err-1" and "req-err-1" in str(err)
    assert err.details == case["body"]["error"].get("details", {})
    if "details" in case["body"]["error"]:
        assert err.details_for(case["code"]) == case["body"]["error"]["details"]
    assert err.issues == case["body"]["error"].get("issues", [])
    if case["status"] == 429:
        assert err.retry_after == 12.0


def test_every_documented_code_has_a_specific_class() -> None:
    for status, codes in FIXTURES["error_codes"].items():
        for code in codes:
            cls = error_class_for(int(status), code)
            assert cls.__name__ == EXPECTED[code], (status, code)
            assert getattr(breadbowl, cls.__name__) is cls


def test_unknown_code_falls_back_to_status_class() -> None:
    assert error_class_for(422, "brand_new_code") is UnprocessableEntityError
    assert error_class_for(429, "brand_new_code") is breadbowl.RateLimitError
    assert error_class_for(502, None) is InternalServerError
    assert error_class_for(418, "teapot") is APIStatusError
    # A known code with a different status is handled by its status.
    assert error_class_for(400, "not_found") is breadbowl.BadRequestError


def test_typed_details() -> None:
    err = InsufficientScopeError(
        "x",
        status_code=403,
        code="insufficient_scope",
        request_id=None,
        details={"required_scope": "indexes:write"},
    )
    assert err.required_scope == "indexes:write"
    conflict = error_class_for(409, "index_name_conflict")(
        "x", status_code=409, code="index_name_conflict", request_id=None, details={"index_id": "idx"}
    )
    assert conflict.index_id == "idx"  # type: ignore[attr-defined]
    details = {"limit": 100000, "current": 99990, "requested": 46, "replaced": 6}
    full = breadbowl.IndexChunkLimitExceededError(
        "x", status_code=422, code="index_chunk_limit_exceeded", request_id=None, details=details
    )
    assert (full.limit, full.current, full.requested, full.replaced) == (100000, 99990, 46, 6)
    assert full.details_for("index_chunk_limit_exceeded") == details
    assert isinstance(full, UnprocessableEntityError)
    assert not isinstance(full, breadbowl.ChunkLimitExceededError)


def test_non_json_error_body(api: MockAPI) -> None:
    client = BreadBowl("k", base_url="https://api.test", max_retries=0, http_client=_http(api))
    api.add(None, 502, text="<html>Bad gateway</html>", request_id=None)
    with pytest.raises(InternalServerError) as info:
        client.models.get("bb-embed")
    assert info.value.code is None and "Bad gateway" in info.value.message and info.value.request_id is None


def test_401_exposes_www_authenticate(api: MockAPI) -> None:
    client = BreadBowl("k", base_url="https://api.test", http_client=_http(api))
    api.add(
        {"error": {"code": "authentication_failed", "message": "Authentication failed"}},
        401,
        headers={"WWW-Authenticate": 'Bearer realm="bb-embed", error="invalid_token"'},
    )
    with pytest.raises(breadbowl.AuthenticationError) as info:
        client.usage.get()
    assert info.value.headers["www-authenticate"].startswith("Bearer")
    assert len(api.requests) == 1  # 4xx is never retried


def test_success_without_json_body(api: MockAPI) -> None:
    client = BreadBowl("k", base_url="https://api.test", http_client=_http(api))
    api.add(None, 200, text="not json")
    with pytest.raises(APIResponseError, match="expected a JSON body"):
        client.usage.get()


def test_hierarchy() -> None:
    assert issubclass(breadbowl.APITimeoutError, breadbowl.APIConnectionError)
    assert issubclass(breadbowl.UsageLimitExceededError, breadbowl.RateLimitError)
    assert issubclass(breadbowl.RequestValidationError, breadbowl.UnprocessableEntityError)
    for name in EXPECTED.values():
        assert issubclass(getattr(breadbowl, name), breadbowl.BreadBowlError)


def _http(api: MockAPI):  # type: ignore[no-untyped-def]
    import httpx

    return httpx.Client(transport=httpx.MockTransport(api.handler))
