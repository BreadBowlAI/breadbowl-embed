"""Bounded retries: what is retried per operation kind, Retry-After, and idempotency keys."""

from __future__ import annotations

import httpx
import pytest

from breadbowl import (
    APIConnectionError,
    APITimeoutError,
    AsyncBreadBowl,
    BreadBowl,
    IndexNotReadyError,
    InternalServerError,
    RateLimitExceededError,
    RequestValidationError,
    ServiceUnavailableError,
    UsageLimitExceededError,
)
from tests.conftest import MockAPI, error_body, example

INDEX_ID = "0b8a3d0e-5c1f-4f7e-9a51-3e2d6c4b1a90"
RATE = error_body("rate_limit_exceeded")
INTERNAL = error_body("internal_error")
UNAVAILABLE = error_body("service_unavailable")


def test_429_honours_retry_after_then_succeeds(api: MockAPI, client: BreadBowl) -> None:
    api.add(RATE, 429, headers={"Retry-After": "2"}).add(example("encode"))
    res = client.encode("q", input_type="query")
    assert api.sleeps == [2.0] and len(api.requests) == 2
    assert res.request_id == "req-test-1"


def test_retry_after_http_date_is_accepted(api: MockAPI, client: BreadBowl) -> None:
    api.add(RATE, 429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}).add(example("getUsage"))
    client.usage.get()
    assert api.sleeps == [0.0]  # a date in the past means "now"


def test_usage_limit_is_not_retried(api: MockAPI, client: BreadBowl) -> None:
    api.add(error_body("usage_limit_exceeded"), 429, headers={"Retry-After": "86400"})
    with pytest.raises(UsageLimitExceededError) as info:
        client.encode("q", input_type="query")
    assert info.value.retry_after == 86400 and api.sleeps == [] and len(api.requests) == 1


def test_retry_after_longer_than_max_wait_is_raised(api: MockAPI, client: BreadBowl) -> None:
    api.add(RATE, 429, headers={"Retry-After": "600"})
    with pytest.raises(RateLimitExceededError):
        client.usage.get()
    assert api.sleeps == []


def test_gives_up_after_max_retries(api: MockAPI, client: BreadBowl) -> None:
    for _ in range(3):
        api.add(UNAVAILABLE, 503)
    with pytest.raises(ServiceUnavailableError):
        client.models.list()
    assert len(api.requests) == 3  # 1 + max_retries(2)
    assert len(api.sleeps) == 2
    assert 0.25 <= api.sleeps[0] <= 0.5 and 0.5 <= api.sleeps[1] <= 1.0  # jittered exponential backoff


def test_max_retries_zero(api: MockAPI, client: BreadBowl) -> None:
    api.add(UNAVAILABLE, 503)
    with pytest.raises(ServiceUnavailableError):
        client.with_options(max_retries=0).usage.get()
    assert len(api.requests) == 1


@pytest.mark.parametrize("status", [500, 502, 504])
def test_server_errors_retry_reads(api: MockAPI, client: BreadBowl, status: int) -> None:
    api.add(INTERNAL, status).add(example("getIndex"))
    assert client.indexes.get(INDEX_ID).id == INDEX_ID
    assert len(api.requests) == 2


@pytest.mark.parametrize("status", [500, 502, 504])
def test_server_errors_retry_encode_score_search(api: MockAPI, client: BreadBowl, status: int) -> None:
    # Only 2xx responses are billed, so a failed attempt is retried without double billing.
    api.add(INTERNAL, status).add(example("searchIndex"))
    api.add(INTERNAL, status).add(example("encode"))
    api.add(INTERNAL, status).add(example("score", 200, "basic"))
    client.indexes.search(INDEX_ID, "q")
    client.encode("q", input_type="query")
    client.score("q", ["a"])
    assert len(api.requests) == 6


def test_server_error_raised_after_retries(api: MockAPI, client: BreadBowl) -> None:
    for _ in range(3):
        api.add(INTERNAL, 500)
    with pytest.raises(InternalServerError):
        client.indexes.search(INDEX_ID, "q")
    assert len(api.requests) == 3


def test_503_retries_score(api: MockAPI, client: BreadBowl) -> None:
    api.add(UNAVAILABLE, 503, headers={"Retry-After": "1"}).add(example("score", 200, "basic"))
    client.score("q", ["a", "b"])
    assert api.sleeps == [1.0]


def test_mutation_reuses_idempotency_key_across_retries(api: MockAPI, client: BreadBowl) -> None:
    api.add(INTERNAL, 500).add(UNAVAILABLE, 503).add(example("createIndex", 201), 201)
    index = client.indexes.create("production")
    keys = {r.headers["idempotency-key"] for r in api.requests}
    assert len(api.requests) == 3 and len(keys) == 1
    assert index.meta is not None and index.meta.idempotency_key in keys


def test_each_mutation_call_gets_a_fresh_key(api: MockAPI, client: BreadBowl) -> None:
    api.add(example("deleteDocument")).add(example("deleteDocument"))
    client.indexes.documents.delete(INDEX_ID, "doc-1")
    client.indexes.documents.delete(INDEX_ID, "doc-1")
    assert api.requests[0].headers["idempotency-key"] != api.requests[1].headers["idempotency-key"]


def test_idempotency_in_progress_without_retry_after_backs_off(api: MockAPI, client: BreadBowl) -> None:
    api.add(error_body("idempotency_request_in_progress"), 409).add(example("deleteIndex"))
    client.indexes.delete(INDEX_ID, idempotency_key="k1")
    assert [r.headers["idempotency-key"] for r in api.requests] == ["k1", "k1"]
    assert len(api.sleeps) == 1


def test_idempotency_in_progress_with_retry_after_is_retried(api: MockAPI, client: BreadBowl) -> None:
    api.add(error_body("idempotency_request_in_progress"), 409, headers={"Retry-After": "3"})
    api.add(example("upsertDocuments"))
    client.indexes.documents.upsert(INDEX_ID, [{"doc_id": "d", "text": "t"}], idempotency_key="fixed")
    assert api.sleeps == [3.0]
    assert [r.headers["idempotency-key"] for r in api.requests] == ["fixed", "fixed"]


def test_index_not_ready_retried_only_while_creating(api: MockAPI, client: BreadBowl) -> None:
    creating = {
        "error": {"code": "index_not_ready", "message": "creating", "details": {"status": "creating"}}
    }
    failed = {"error": {"code": "index_not_ready", "message": "error", "details": {"status": "error"}}}
    api.add(creating, 409, headers={"Retry-After": "5"}).add(example("searchIndex"))
    client.indexes.search(INDEX_ID, "q")
    assert api.sleeps == [5.0]
    api.add(failed, 409, headers={"Retry-After": "5"})  # typed status decides, not the header
    with pytest.raises(IndexNotReadyError) as info:
        client.indexes.search(INDEX_ID, "q")
    assert info.value.index_status == "error" and len(api.requests) == 3


def test_validation_errors_are_never_retried(api: MockAPI, client: BreadBowl) -> None:
    api.add(error_body("validation_error"), 422)
    with pytest.raises(RequestValidationError) as info:
        client.indexes.documents.upsert(INDEX_ID, [{"doc_id": "d", "text": "t"}])
    assert info.value.issues[0]["path"] == ["documents", 3, "doc_id"]
    assert len(api.requests) == 1


def test_connect_errors_are_retried_for_every_kind(api: MockAPI, client: BreadBowl) -> None:
    api.fail(httpx.ConnectError).add(example("encode"))
    client.encode("q", input_type="query")
    assert len(api.requests) == 2


def test_timeouts_are_retried_then_raised(api: MockAPI, client: BreadBowl) -> None:
    api.fail(httpx.ReadTimeout).add(example("encode"))
    client.encode("q", input_type="query")
    for _ in range(3):
        api.fail(httpx.ReadTimeout)
    with pytest.raises(APITimeoutError):
        client.encode("q", input_type="query")
    assert len(api.requests) == 5


def test_read_timeout_retried_for_reads_and_mutations(api: MockAPI, client: BreadBowl) -> None:
    api.fail(httpx.ReadTimeout).add(example("getUsage"))
    client.usage.get()
    api.fail(httpx.RemoteProtocolError).add(example("deleteIndex"))
    client.indexes.delete(INDEX_ID)
    assert len(api.requests) == 4


def test_connection_error_after_retries(api: MockAPI, client: BreadBowl) -> None:
    for _ in range(3):
        api.fail(httpx.ConnectError, "refused")
    with pytest.raises(APIConnectionError, match="refused") as info:
        client.usage.get()
    assert not isinstance(info.value, APITimeoutError) and info.value.request is not None


@pytest.mark.anyio
async def test_async_retries_and_key_reuse(api: MockAPI, aclient: AsyncBreadBowl) -> None:
    api.add(RATE, 429, headers={"Retry-After": "1"}).fail(httpx.ConnectError).add(example("deleteIndex"))
    await aclient.indexes.delete(INDEX_ID)
    assert api.sleeps[0] == 1.0 and len(api.sleeps) == 2
    assert len({r.headers["idempotency-key"] for r in api.requests}) == 1


@pytest.mark.anyio
async def test_async_score_500_is_retried(api: MockAPI, aclient: AsyncBreadBowl) -> None:
    api.add(INTERNAL, 500).add(example("score", 200, "basic"))
    await aclient.score("q", ["d"])
    assert len(api.requests) == 2
