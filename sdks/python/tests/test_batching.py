"""documents.upsert_batched and the upsert timeout, against a simulated server that chunks
documents (``metadata.chunks`` chunks each) and enforces a chunk limit per request."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx
import pytest

from breadbowl import (
    AsyncBreadBowl,
    BatchUpsertError,
    BreadBowl,
    Document,
    IndexChunkLimitExceededError,
    RepresentationUnavailableError,
    UpsertBatchResult,
)

INDEX_ID = "idx-1"


class ChunkingServer:
    def __init__(
        self,
        limit: int = 6,
        fail_doc: Optional[str] = None,
        break_after: Optional[int] = None,
        index_limit: Optional[int] = None,
    ) -> None:
        self.limit = limit
        self.fail_doc = fail_doc
        self.break_after = break_after
        self.index_limit = index_limit
        self.stored = 0
        self.batches: List[List[str]] = []
        self.keys: List[str] = []
        self.timeouts: List[Any] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.timeouts.append(request.extensions.get("timeout"))
        if request.method != "POST":
            return httpx.Response(200, json={"ok": True}, headers={"x-request-id": "r"})
        docs = json.loads(request.content)["documents"]
        self.batches.append([d["doc_id"] for d in docs])
        self.keys.append(request.headers["idempotency-key"])
        if self.break_after is not None and len(self.batches) > self.break_after:
            body = {
                "error": {
                    "code": "representation_unavailable",
                    "message": "dtype changed",
                    "details": {"representation": {"compute_dtype": "bfloat16"}, "served": None},
                }
            }
            return httpx.Response(409, json=body, headers={"x-request-id": "r-409"})
        requested = sum(int((d.get("metadata") or {}).get("chunks", 1)) for d in docs)
        if requested > self.limit:
            body = {
                "error": {
                    "code": "chunk_limit_exceeded",
                    "message": "too many chunks",
                    "details": {"limit": self.limit, "requested": requested},
                }
            }
            return httpx.Response(422, json=body, headers={"x-request-id": "r-422"})
        if self.index_limit is not None and self.stored + requested > self.index_limit:
            details = {"limit": self.index_limit, "current": self.stored, "requested": requested}
            body = {
                "error": {
                    "code": "index_chunk_limit_exceeded",
                    "message": "index full",
                    "details": {**details, "replaced": 0},
                }
            }
            return httpx.Response(422, json=body, headers={"x-request-id": "r-422"})
        self.stored += requested
        outcomes: List[Dict[str, Any]] = []
        for d in reversed(docs):  # any order: the helper matches outcomes by doc_id
            if d["doc_id"] == self.fail_doc:
                outcomes.append(
                    {
                        "doc_id": d["doc_id"],
                        "status": "failed",
                        "error": {"code": "encoding_failed", "message": "worker", "retryable": True},
                    }
                )
            else:
                outcomes.append(
                    {
                        "doc_id": d["doc_id"],
                        "status": "upserted",
                        "content_sha256": "0" * 64,
                        "token_count": 10,
                        "chunk_count": int((d.get("metadata") or {}).get("chunks", 1)),
                        "truncated": False,
                    }
                )
        ok = [o for o in outcomes if o["status"] == "upserted"]
        body = {
            "index_id": INDEX_ID,
            "documents": outcomes,
            "upserted_documents": len(ok),
            "upserted_chunks": sum(o["chunk_count"] for o in ok),
            "failed_documents": len(outcomes) - len(ok),
            "usage": {
                "input_tokens": 10 * len(ok),
                "document_count": len(ok),
                "chunk_count": 0,
                "candidate_count": 0,
                "latency_ms": 5,
            },
        }
        return httpx.Response(200, json=body, headers={"x-request-id": f"r-{len(self.batches)}"})


def client_for(server: ChunkingServer, **kwargs: Any) -> BreadBowl:
    c = BreadBowl(
        "k",
        base_url="https://api.test",
        http_client=httpx.Client(transport=httpx.MockTransport(server.handler)),
        **kwargs,
    )
    c._sleep = lambda s: None
    return c


def docs(n: int, chunks: int = 1) -> List[Document]:
    return [Document(f"d{i}", f"text {i}", {"chunks": chunks}) for i in range(n)]


def test_splits_by_document_count_and_merges_in_input_order() -> None:
    server = ChunkingServer(limit=1000)
    result = client_for(server).indexes.documents.upsert_batched(INDEX_ID, docs(120))
    assert [len(b) for b in server.batches] == [50, 50, 20]
    assert isinstance(result, UpsertBatchResult)
    assert [d.doc_id for d in result.documents] == [f"d{i}" for i in range(120)]
    assert result.upserted_documents == 120 and result.failed_documents == 0 and result.upserted_chunks == 120
    assert result.input_tokens == 1200 and result.request_ids == ["r-1", "r-2", "r-3"]
    assert len(set(server.keys)) == 3  # one Idempotency-Key per request


def test_chunk_limit_splits_the_batch_and_retries() -> None:
    server = ChunkingServer(limit=6)
    result = client_for(server).indexes.documents.upsert_batched(INDEX_ID, docs(10, chunks=2))
    # 20 chunks > 6: ceil(20/6) = 4 parts of 3 docs (6 chunks) ... then 1 doc.
    assert server.batches[0] == [f"d{i}" for i in range(10)]
    assert all(sum(2 for _ in b) <= 6 for b in server.batches[1:])
    assert result.upserted_documents == 10
    assert [d.doc_id for d in result.documents] == [f"d{i}" for i in range(10)]
    assert len(set(server.keys)) == len(server.keys)


def test_a_single_document_over_the_limit_is_a_failed_outcome() -> None:
    server = ChunkingServer(limit=6)
    batch = [
        Document("small", "a", {"chunks": 1}),
        Document("huge", "b", {"chunks": 9}),
        Document("tail", "c", {"chunks": 2}),
    ]
    result = client_for(server).indexes.documents.upsert_batched(INDEX_ID, batch)
    assert [d.doc_id for d in result.documents] == ["small", "huge", "tail"]
    huge = result.documents[1]
    assert huge.status == "failed" and huge.error is not None
    assert (
        huge.error.code == "chunk_limit_exceeded"
        and huge.error.retryable is False
        and "9 chunks" in huge.error.message
    )
    assert [d.doc_id for d in result.failed] == ["huge"] and result.upserted_documents == 2


def test_per_document_failures_are_merged() -> None:
    server = ChunkingServer(limit=1000, fail_doc="d3")
    result = client_for(server).indexes.documents.upsert_batched(INDEX_ID, docs(5), batch_size=2)
    assert [len(b) for b in server.batches] == [2, 2, 1]
    assert [d.doc_id for d in result.failed] == ["d3"] and [d.doc_id for d in result.retryable] == ["d3"]


def test_a_request_failure_stops_with_partial_results() -> None:
    server = ChunkingServer(limit=1000, break_after=1)
    with pytest.raises(BatchUpsertError) as info:
        client_for(server).indexes.documents.upsert_batched(INDEX_ID, docs(7), batch_size=3)
    err = info.value
    assert isinstance(err.error, RepresentationUnavailableError) and err.__cause__ is err.error
    assert err.error.served is None and err.error.representation == {"compute_dtype": "bfloat16"}
    assert [d.doc_id for d in err.result.documents] == ["d0", "d1", "d2"]
    assert err.pending == ["d3", "d4", "d5", "d6"]
    assert len(server.batches) == 2  # 409 representation_unavailable is not retried


def test_index_chunk_limit_stops_without_splitting_or_retrying() -> None:
    server = ChunkingServer(limit=1000, index_limit=5)
    with pytest.raises(BatchUpsertError) as info:
        client_for(server).indexes.documents.upsert_batched(INDEX_ID, docs(7), batch_size=3)
    err = info.value
    assert isinstance(err.error, IndexChunkLimitExceededError) and err.error.current == 3
    assert [d.doc_id for d in err.result.documents] == ["d0", "d1", "d2"]
    assert err.pending == ["d3", "d4", "d5", "d6"]
    assert len(server.batches) == 2


def test_invalid_input_is_rejected_before_any_request() -> None:
    server = ChunkingServer()
    c = client_for(server)
    with pytest.raises(ValueError, match="d1"):
        c.indexes.documents.upsert_batched(INDEX_ID, [Document("d1", "a"), Document("d1", "b")])
    with pytest.raises(ValueError, match="batch_size"):
        c.indexes.documents.upsert_batched(INDEX_ID, docs(2), batch_size=0)
    assert server.batches == []


def test_index_handle_and_empty_input() -> None:
    server = ChunkingServer()
    c = client_for(server)
    assert c.index(INDEX_ID).upsert_batched(docs(3)).upserted_documents == 3
    assert c.indexes.documents.upsert_batched(INDEX_ID, []).documents == []


def test_upserts_get_the_longer_timeout() -> None:
    server = ChunkingServer()
    c = client_for(server)
    c.indexes.documents.upsert(INDEX_ID, docs(1))
    c.health.live()
    upsert_timeout, other_timeout = server.timeouts
    assert upsert_timeout["read"] == 180.0 and other_timeout["read"] == 60.0
    c.with_options(upsert_timeout=300.0).indexes.documents.upsert(INDEX_ID, docs(1))
    assert server.timeouts[-1]["read"] == 300.0
    c2 = client_for(server, timeout=10.0, upsert_timeout=240.0)
    c2.indexes.documents.upsert(INDEX_ID, docs(1))
    assert server.timeouts[-1]["read"] == 240.0


@pytest.mark.anyio
async def test_async_batched_upsert() -> None:
    server = ChunkingServer(limit=6)
    c = AsyncBreadBowl(
        "k",
        base_url="https://api.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(server.handler)),
    )
    result = await c.index(INDEX_ID).upsert_batched(docs(10, chunks=2))
    assert result.upserted_documents == 10 and len(server.batches) > 1
    assert server.timeouts[0]["read"] == 180.0
