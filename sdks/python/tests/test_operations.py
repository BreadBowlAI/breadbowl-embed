"""Every operation, sync and async: method, path, query, headers, body and parsed result."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pytest

import breadbowl
from breadbowl import (
    AsyncBreadBowl,
    BreadBowl,
    BreadBowlError,
    DeleteDocumentResponse,
    DeleteIndexResponse,
    Document,
    EncodeResponse,
    HealthStatus,
    Index,
    Model,
    ReadinessStatus,
    ScoreResponse,
    SearchResponse,
    StoredDocument,
    UpsertDocumentsResponse,
    UsageSummary,
)
from tests.conftest import API_KEY, BASE_URL, MockAPI, example

INDEX_ID = "0b8a3d0e-5c1f-4f7e-9a51-3e2d6c4b1a90"


@dataclass
class Case:
    operation_id: str
    sync: Callable[[BreadBowl], Any]
    run_async: Callable[[AsyncBreadBowl], Any]
    status: int
    response: Dict[str, Any]
    method: str
    path: str
    result_type: type
    body: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, str]] = None
    idempotent: bool = False
    authenticated: bool = True
    check: Optional[Callable[[Any], None]] = None


def _check_encode(r: EncodeResponse) -> bool:
    assert r.data[0].tokens is not None and r.data[0].tokens[7].span == (0, 6)
    assert r.usage.input_tokens == r.data[0].token_count == len(r.data[0].tokens) and r.instruction
    return True


def _check_score(r: ScoreResponse) -> bool:
    assert [x.doc_id for x in r.results] == ["a", "b"]
    assert r.results[0].score == pytest.approx(0.6125) and r.scoring.routing_weight == 0.75
    return True


def _check_upsert(r: UpsertDocumentsResponse) -> bool:
    assert [d.doc_id for d in r.failed] == ["doc-2"] and r.retryable == r.failed
    assert r.documents[0].content_sha256 and r.failed_documents == 1
    return True


CASES = [
    Case(
        "getHealth",
        lambda c: c.health.live(),
        lambda c: c.health.live(),
        200,
        example("getHealth"),
        "GET",
        "/healthz",
        HealthStatus,
        authenticated=False,
        check=lambda r: r.ok is True,
    ),
    Case(
        "getReadiness",
        lambda c: c.health.ready(),
        lambda c: c.health.ready(),
        200,
        example("getReadiness"),
        "GET",
        "/readyz",
        ReadinessStatus,
        authenticated=False,
        check=lambda r: r.ok and r.model.revision,
    ),
    Case(
        "listModels",
        lambda c: c.models.list(limit=5),
        lambda c: c.models.list(limit=5),
        200,
        example("listModels"),
        "GET",
        "/v1/models",
        breadbowl.SyncPage,
        params={"limit": "5"},
        check=lambda p: isinstance(p.data[0], Model) and p.data[0].dimensions.slots == 16 and not p.has_more,
    ),
    Case(
        "getModel",
        lambda c: c.models.get("bb-embed"),
        lambda c: c.models.get("bb-embed"),
        200,
        example("getModel"),
        "GET",
        "/v1/models/bb-embed",
        Model,
        check=lambda m: m.search_profile().name == m.default_search_profile and m.limits.max_top_k == 100,
    ),
    Case(
        "encode",
        lambda c: c.encode(
            "rotate key",
            input_type="query",
            return_=["routing", "attention", "tokens"],
            attention_heads="mean",
        ),
        lambda c: c.encode(
            "rotate key",
            input_type="query",
            return_=["routing", "attention", "tokens"],
            attention_heads="mean",
        ),
        200,
        example("encode"),
        "POST",
        "/v1/encode",
        EncodeResponse,
        body={
            "input": "rotate key",
            "input_type": "query",
            "return": ["routing", "attention", "tokens"],
            "attention_heads": "mean",
        },
        check=_check_encode,
    ),
    Case(
        "score",
        lambda c: c.score("refund policy", [Document("a", "Refunds.", {"source": "help-center"}), "Keys."]),
        lambda c: c.score("refund policy", [Document("a", "Refunds.", {"source": "help-center"}), "Keys."]),
        200,
        example("score", 200, "basic"),
        "POST",
        "/v1/score",
        ScoreResponse,
        body={
            "query": "refund policy",
            "documents": [
                {"doc_id": "a", "text": "Refunds.", "metadata": {"source": "help-center"}},
                {"doc_id": "1", "text": "Keys."},
            ],
        },
        check=_check_score,
    ),
    Case(
        "createIndex",
        lambda c: c.indexes.create(
            "production", chunking={"max_tokens": 256}, metadata_schema={"year": "number"}
        ),
        lambda c: c.indexes.create(
            "production", chunking={"max_tokens": 256}, metadata_schema={"year": "number"}
        ),
        201,
        example("createIndex", 201),
        "POST",
        "/v1/indexes",
        Index,
        body={"name": "production", "chunking": {"max_tokens": 256}, "metadata_schema": {"year": "number"}},
        idempotent=True,
        check=lambda i: i.id == INDEX_ID and i.search_profile.scoring.tau_k == 0.02,
    ),
    Case(
        "listIndexes",
        lambda c: c.indexes.list(name="production", limit=1),
        lambda c: c.indexes.list(name="production", limit=1),
        200,
        example("listIndexes"),
        "GET",
        "/v1/indexes",
        breadbowl.SyncPage,
        params={"name": "production", "limit": "1"},
        check=lambda p: p.has_more and p.next_cursor and p.data[0].name == "production",
    ),
    Case(
        "getIndex",
        lambda c: c.indexes.get(INDEX_ID),
        lambda c: c.indexes.get(INDEX_ID),
        200,
        example("getIndex"),
        "GET",
        f"/v1/indexes/{INDEX_ID}",
        Index,
        check=lambda i: i.status == "ready" and i.chunking.max_tokens == 384,
    ),
    Case(
        "deleteIndex",
        lambda c: c.indexes.delete(INDEX_ID),
        lambda c: c.indexes.delete(INDEX_ID),
        200,
        example("deleteIndex"),
        "DELETE",
        f"/v1/indexes/{INDEX_ID}",
        DeleteIndexResponse,
        idempotent=True,
        check=lambda r: r.deleted and r.documents_deleted == 1204,
    ),
    Case(
        "upsertDocuments",
        lambda c: c.indexes.documents.upsert(
            INDEX_ID,
            [{"doc_id": "doc-1", "text": "Keys.", "metadata": {"year": 2026}}, Document("doc-2", "Refunds.")],
        ),
        lambda c: c.indexes.documents.upsert(
            INDEX_ID,
            [{"doc_id": "doc-1", "text": "Keys.", "metadata": {"year": 2026}}, Document("doc-2", "Refunds.")],
        ),
        200,
        example("upsertDocuments"),
        "POST",
        f"/v1/indexes/{INDEX_ID}/documents",
        UpsertDocumentsResponse,
        body={
            "documents": [
                {"doc_id": "doc-1", "text": "Keys.", "metadata": {"year": 2026}},
                {"doc_id": "doc-2", "text": "Refunds."},
            ]
        },
        idempotent=True,
        check=_check_upsert,
    ),
    Case(
        "getDocument",
        lambda c: c.indexes.documents.get(INDEX_ID, "a/b c", return_vectors=True, dtype="float32"),
        lambda c: c.indexes.documents.get(INDEX_ID, "a/b c", return_vectors=True, dtype="float32"),
        200,
        example("getDocument"),
        "GET",
        f"/v1/indexes/{INDEX_ID}/documents/a%2Fb%20c",
        StoredDocument,
        params={"return_vectors": "true", "dtype": "float32"},
        check=lambda d: d.chunks[0].text("x" * 100) == "x" * 69 and d.metadata["year"] == 2026,
    ),
    Case(
        "deleteDocument",
        lambda c: c.indexes.documents.delete(INDEX_ID, "doc-1"),
        lambda c: c.indexes.documents.delete(INDEX_ID, "doc-1"),
        200,
        example("deleteDocument"),
        "DELETE",
        f"/v1/indexes/{INDEX_ID}/documents/doc-1",
        DeleteDocumentResponse,
        idempotent=True,
        check=lambda r: r.deleted is True,
    ),
    Case(
        "searchIndex",
        lambda c: c.indexes.search(
            INDEX_ID, "How do I rotate a key?", top_k=10, candidate_k=100, filter={"source": "docs"}
        ),
        lambda c: c.indexes.search(
            INDEX_ID, "How do I rotate a key?", top_k=10, candidate_k=100, filter={"source": "docs"}
        ),
        200,
        example("searchIndex"),
        "POST",
        f"/v1/indexes/{INDEX_ID}/search",
        SearchResponse,
        body={
            "query": "How do I rotate a key?",
            "top_k": 10,
            "candidate_k": 100,
            "filter": {"source": "docs"},
        },
        check=lambda r: r.results[1].text("y" * 5000) == "y" * (4630 - 2911) and r.search_profile,
    ),
    Case(
        "getUsage",
        lambda c: c.usage.get(),
        lambda c: c.usage.get(),
        200,
        example("getUsage"),
        "GET",
        "/v1/usage",
        UsageSummary,
        check=lambda u: u.usage.input_tokens == "1834221" and u.limits.requests_per_minute == 60,
    ),
]

IDS = [c.operation_id for c in CASES]


def test_cases_cover_every_operation() -> None:
    assert sorted(IDS) == sorted(breadbowl._generated.OPERATIONS)


def _assert_request(api: MockAPI, case: Case) -> None:
    req = api.last
    assert req.method == case.method
    assert req.raw_path == case.path
    assert req.params == (case.params or {})
    assert req.body == case.body
    assert req.headers["user-agent"] == f"breadbowl-python/{breadbowl.__version__}"
    assert req.headers["accept"] == "application/json"
    if case.authenticated:
        assert req.headers["authorization"] == f"Bearer {API_KEY}"
    else:
        assert "authorization" not in req.headers
    if case.body is not None:
        assert req.headers["content-type"] == "application/json"
    if case.idempotent:
        uuid.UUID(req.headers["idempotency-key"])
    else:
        assert "idempotency-key" not in req.headers


def _assert_result(result: Any, case: Case) -> None:
    assert isinstance(result, case.result_type)
    assert result.request_id == "req-test-1"
    if case.check is not None:
        assert case.check(result)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_sync_operation(case: Case, api: MockAPI, client: BreadBowl) -> None:
    api.add(case.response, case.status)
    result = case.sync(client)
    _assert_request(api, case)
    _assert_result(result, case)


@pytest.mark.anyio
@pytest.mark.parametrize("case", CASES, ids=IDS)
async def test_async_operation(case: Case, api: MockAPI, aclient: AsyncBreadBowl) -> None:
    api.add(case.response, case.status)
    result = case.run_async(aclient)
    result = await result
    _assert_request(api, case)
    if case.result_type is breadbowl.SyncPage:
        assert isinstance(result, breadbowl.AsyncPage)
        assert result.request_id == "req-test-1"
    else:
        _assert_result(result, case)


def test_meta_and_raw(api: MockAPI, client: BreadBowl) -> None:
    body = example("deleteIndex")
    body["future_field"] = {"x": 1}
    api.add(body, headers={"Idempotent-Replayed": "true"})
    res = client.indexes.delete(INDEX_ID, idempotency_key="my-key-1")
    assert api.last.headers["idempotency-key"] == "my-key-1"
    assert res.meta is not None and res.meta.idempotent_replayed and res.meta.idempotency_key == "my-key-1"
    assert res.raw["future_field"] == {"x": 1}  # unknown fields survive in .raw
    assert res.to_dict() == example("deleteIndex")


def test_readiness_503_is_a_result_not_an_error(api: MockAPI, client: BreadBowl) -> None:
    api.add({"ok": False}, 503)
    res = client.health.ready()
    assert res.ok is False and res.model is None
    assert len(api.requests) == 1  # probes are never retried


def test_index_handle(api: MockAPI, client: BreadBowl) -> None:
    idx = client.index(INDEX_ID)
    api.add(example("searchIndex")).add(example("upsertDocuments")).add(example("getDocument"))
    api.add(example("deleteDocument")).add(example("getIndex")).add(example("deleteIndex"))
    idx.search("q", top_k=3)
    assert api.last.path.endswith("/search") and api.last.body == {"query": "q", "top_k": 3}
    idx.upsert([Document("doc-1", "t")])
    idx.get_document("doc-1")
    idx.delete_document("doc-1")
    assert idx.get().id == INDEX_ID
    assert idx.delete().deleted
    assert [r.method for r in api.requests] == ["POST", "POST", "GET", "DELETE", "GET", "DELETE"]


@pytest.mark.anyio
async def test_async_index_handle(api: MockAPI, aclient: AsyncBreadBowl) -> None:
    idx = aclient.index(INDEX_ID)
    api.add(example("searchIndex")).add(example("upsertDocuments")).add(example("getDocument"))
    api.add(example("deleteDocument")).add(example("getIndex")).add(example("deleteIndex"))
    await idx.search("q")
    await idx.upsert([Document("doc-1", "t")])
    await idx.get_document("doc-1")
    await idx.delete_document("doc-1")
    await idx.get()
    await idx.delete()
    assert len(api.requests) == 6


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch, api: MockAPI) -> None:
    import httpx

    monkeypatch.setenv("BREADBOWL_API_KEY", "from-env")
    monkeypatch.setenv("BREADBOWL_BASE_URL", "https://self-hosted.example:8080/")
    c = BreadBowl(http_client=httpx.Client(transport=httpx.MockTransport(api.handler)))
    api.add(example("getUsage"))
    c.usage.get()
    assert api.last.headers["authorization"] == "Bearer from-env"
    assert c.base_url == "https://self-hosted.example:8080"


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BREADBOWL_API_KEY", raising=False)
    with pytest.raises(BreadBowlError, match="BREADBOWL_API_KEY"):
        BreadBowl()


def test_default_base_url_comes_from_the_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BREADBOWL_BASE_URL", raising=False)
    assert BreadBowl("k").base_url == breadbowl.DEFAULT_BASE_URL.rstrip("/")
    assert re.match(r"^https?://", breadbowl.DEFAULT_BASE_URL)


def test_client_side_argument_checks(client: BreadBowl) -> None:
    with pytest.raises(ValueError, match="index_id"):
        client.indexes.get("")
    with pytest.raises(ValueError, match="doc_id"):
        client.indexes.documents.delete(INDEX_ID, "")
    with pytest.raises(TypeError, match="sequence"):
        client.indexes.documents.upsert(INDEX_ID, Document("a", "b"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Document"):
        client.score("q", [42])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        client.indexes.documents.upsert(
            INDEX_ID, [{"doc_id": "a", "text": "b", "metadata": {"x": float("nan")}}]
        )


def test_default_headers_and_with_options(api: MockAPI, client: BreadBowl) -> None:
    import httpx

    c = BreadBowl(
        "k",
        base_url=BASE_URL,
        default_headers={"X-Trace": "1"},
        http_client=httpx.Client(transport=httpx.MockTransport(api.handler)),
    )
    api.add(example("getUsage"))
    c.with_options(timeout=5.0, max_retries=0).usage.get()
    assert api.last.headers["x-trace"] == "1"
    assert c.with_options(max_retries=0).max_retries == 0 and c.max_retries == 2


def test_context_managers() -> None:
    with BreadBowl("k") as c:
        assert not c._http.is_closed
    assert c._http.is_closed


@pytest.mark.anyio
async def test_async_context_manager() -> None:
    async with AsyncBreadBowl("k") as c:
        assert not c._http.is_closed
    assert c._http.is_closed
