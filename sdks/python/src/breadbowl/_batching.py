"""Batched upserts: any number of documents, split into requests the API accepts.

The API takes at most 50 documents and ``limits.max_chunks_per_upsert`` (128) chunks per
request. Chunk counts are only known after the server has tokenized, so the helper starts with
batches of ``batch_size`` documents. On 422 ``chunk_limit_exceeded`` it splits that batch into
``ceil(requested / limit)`` parts (at least two) and retries them. A single document that alone
exceeds the limit is reported as a failed outcome with code ``chunk_limit_exceeded``.

Per-document outcomes from every request are merged back into input order. Any other error stops
the run: ``BatchUpsertError`` carries the outcomes so far and the ``doc_id``s still pending.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from ._errors import BreadBowlError, ChunkLimitExceededError
from ._models import DocumentError, DocumentOutcome, UpsertDocumentsResponse

__all__ = ["UpsertBatchResult", "BatchUpsertError", "DEFAULT_BATCH_SIZE"]

DEFAULT_BATCH_SIZE = 50  # the contract's limits.max_documents_per_upsert

Batch = List[Tuple[int, Dict[str, Any]]]  # (input position, document)


@dataclass
class UpsertBatchResult:
    """The merged outcome of a batched upsert."""

    index_id: str
    documents: List[DocumentOutcome]
    """One outcome per input document, in input order."""
    responses: List[UpsertDocumentsResponse] = field(default_factory=list)
    """Every successful request's response, in the order they were sent."""

    @property
    def upserted_documents(self) -> int:
        return sum(1 for d in self.documents if d.status == "upserted")

    @property
    def failed_documents(self) -> int:
        return sum(1 for d in self.documents if d.status != "upserted")

    @property
    def upserted_chunks(self) -> int:
        return sum(r.upserted_chunks for r in self.responses)

    @property
    def input_tokens(self) -> int:
        return sum(r.usage.input_tokens for r in self.responses)

    @property
    def failed(self) -> List[DocumentOutcome]:
        return [d for d in self.documents if d.status != "upserted"]

    @property
    def retryable(self) -> List[DocumentOutcome]:
        return [d for d in self.failed if d.error is not None and d.error.retryable]

    @property
    def request_ids(self) -> List[Optional[str]]:
        return [r.request_id for r in self.responses]


class BatchUpsertError(BreadBowlError):
    """A request of a batched upsert failed as a whole (after the client's own retries).

    Earlier batches are durable: ``result`` holds their outcomes, ``pending`` the ``doc_id``s that
    were not written, and ``error`` (also ``__cause__``) the error that stopped the run."""

    def __init__(self, error: BaseException, result: UpsertBatchResult, pending: List[str]) -> None:
        self.error = error
        self.result = result
        self.pending = pending
        super().__init__(
            f"batched upsert stopped with {len(pending)} documents pending "
            f"({result.upserted_documents} upserted): {error}"
        )


def prepare(documents: Sequence[Mapping[str, Any]], batch_size: int) -> Deque[Batch]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    seen: Dict[str, int] = {}
    for i, doc in enumerate(documents):
        doc_id = doc.get("doc_id")
        if doc_id in seen:
            raise ValueError(f"doc_id {doc_id!r} appears at positions {seen[doc_id]} and {i}")
        seen[str(doc_id)] = i
    items = list(enumerate(dict(d) for d in documents))
    return deque(items[i : i + batch_size] for i in range(0, len(items), batch_size))


def split(batch: Batch, error: ChunkLimitExceededError) -> List[Batch]:
    """Split a batch the server rejected for its chunk count into parts that should fit."""
    parts = 2
    if error.limit and error.requested:
        parts = max(2, math.ceil(error.requested / error.limit))
    parts = min(parts, len(batch))
    size = math.ceil(len(batch) / parts)
    return [batch[i : i + size] for i in range(0, len(batch), size)]


def too_large(doc_id: str, error: ChunkLimitExceededError) -> DocumentOutcome:
    return DocumentOutcome(
        doc_id=doc_id,
        status="failed",
        error=DocumentError(
            code="chunk_limit_exceeded",
            message=(
                f"this document alone produces {error.requested} chunks; the limit per request is "
                f"{error.limit}. Shorten it or use larger chunks."
            ),
            retryable=False,
        ),
    )


def record(outcomes: Dict[int, DocumentOutcome], batch: Batch, response: UpsertDocumentsResponse) -> None:
    by_id = {o.doc_id: o for o in response.documents}
    for position, doc in batch:
        outcome = by_id.get(doc["doc_id"])
        if outcome is None:  # a server that omits an outcome breaks the contract; report it
            outcome = DocumentOutcome(
                doc_id=doc["doc_id"],
                status="failed",
                error=DocumentError(
                    code="missing_outcome", message="no outcome in the response", retryable=True
                ),
            )
        outcomes[position] = outcome


def result(
    index_id: str, total: int, outcomes: Dict[int, DocumentOutcome], responses: List[Any]
) -> UpsertBatchResult:
    return UpsertBatchResult(
        index_id=index_id,
        documents=[outcomes[i] for i in range(total) if i in outcomes],
        responses=responses,
    )


def pending_ids(current: Batch, queue: Deque[Batch]) -> List[str]:
    return [doc["doc_id"] for batch in (current, *queue) for _, doc in batch]
