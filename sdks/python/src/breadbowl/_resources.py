"""Resource namespaces. Each operation is built once as a ``Call`` and executed by the sync or
async client; only the execution differs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from . import _batching as batching
from ._base import Call, compact, query_value, require_id
from ._batching import DEFAULT_BATCH_SIZE, BatchUpsertError, UpsertBatchResult
from ._errors import BreadBowlError, ChunkLimitExceededError
from ._generated import (
    AttentionHeads,
    ChunkingConfig,
    EncodeReturnField,
    EncodingFormat,
    InputType,
    MetadataFilter,
    MetadataSchema,
    TensorDtype,
)
from ._models import (
    BaseModel,
    DeleteDocumentResponse,
    DeleteIndexResponse,
    DocumentOutcome,
    EncodeResponse,
    HealthStatus,
    Index,
    Metadata,
    Model,
    ReadinessStatus,
    ResponseMeta,
    ScoreResponse,
    SearchResponse,
    StoredDocument,
    UpsertDocumentsResponse,
    UsageSummary,
)
from ._pagination import AsyncPage, AsyncPaginator, SyncPage, parse_page_data

if TYPE_CHECKING:  # pragma: no cover
    from ._client import AsyncBreadBowl, BreadBowl

__all__ = ["Document", "DocumentLike"]

M = TypeVar("M", bound=BaseModel)


@dataclass
class Document:
    """A document to score or upsert. Plain dicts with the same keys work too."""

    doc_id: str
    text: str
    metadata: Optional[Metadata] = None


DocumentLike = Union[Document, Mapping[str, Any]]


def _document(value: Any, position: int, *, allow_str: bool) -> Dict[str, Any]:
    if isinstance(value, Document):
        return compact({"doc_id": value.doc_id, "text": value.text, "metadata": value.metadata})
    if isinstance(value, Mapping):
        return compact(dict(value))
    if allow_str and isinstance(value, str):
        return {"doc_id": str(position), "text": value}
    raise TypeError(
        f"document {position} must be a breadbowl.Document or a mapping with doc_id and text, "
        f"got {type(value).__name__}"
    )


def _model_parser[M: BaseModel](cls: Type[M]) -> Callable[[Any, ResponseMeta], M]:
    return lambda data, meta: cls.from_dict(data)._with_meta(meta)


# ---------------------------------------------------------------------------- call builders


def encode_call(
    input: Union[str, Sequence[str]],
    *,
    input_type: InputType,
    model: Optional[str],
    instruction: Optional[str],
    return_: Optional[Sequence[EncodeReturnField]],
    encoding_format: Optional[EncodingFormat],
    dtype: Optional[TensorDtype],
    attention_heads: Optional[AttentionHeads],
) -> Call[EncodeResponse]:
    if not isinstance(input, str):
        input = list(input)
    body = compact(
        {
            "model": model,
            "input": input,
            "input_type": input_type,
            "instruction": instruction,
            "return": list(return_) if return_ is not None else None,
            "encoding_format": encoding_format,
            "dtype": dtype,
            "attention_heads": attention_heads,
        }
    )
    return Call("encode", _model_parser(EncodeResponse), body=body)


def score_call(
    query: str,
    documents: Sequence[Union[DocumentLike, str]],
    *,
    model: Optional[str],
    instruction: Optional[str],
    explain: Optional[bool],
    return_attention: Optional[bool],
    attention_heads: Optional[AttentionHeads],
    encoding_format: Optional[EncodingFormat],
    dtype: Optional[TensorDtype],
) -> Call[ScoreResponse]:
    body = compact(
        {
            "model": model,
            "query": query,
            "instruction": instruction,
            "documents": [_document(d, i, allow_str=True) for i, d in enumerate(documents)],
            "explain": explain,
            "return_attention": return_attention,
            "attention_heads": attention_heads,
            "encoding_format": encoding_format,
            "dtype": dtype,
        }
    )
    return Call("score", _model_parser(ScoreResponse), body=body)


def search_call(
    index_id: str,
    query: str,
    *,
    top_k: Optional[int],
    candidate_k: Optional[int],
    filter: Optional[MetadataFilter],
    instruction: Optional[str],
    explain: Optional[bool],
    return_attention: Optional[bool],
    return_vectors: Optional[bool],
    attention_heads: Optional[AttentionHeads],
    encoding_format: Optional[EncodingFormat],
    dtype: Optional[TensorDtype],
) -> Call[SearchResponse]:
    body = compact(
        {
            "query": query,
            "instruction": instruction,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "filter": dict(filter) if filter is not None else None,
            "explain": explain,
            "return_attention": return_attention,
            "return_vectors": return_vectors,
            "attention_heads": attention_heads,
            "encoding_format": encoding_format,
            "dtype": dtype,
        }
    )
    return Call(
        "searchIndex",
        _model_parser(SearchResponse),
        path_params={"index_id": require_id("index_id", index_id)},
        body=body,
    )


def create_index_call(
    name: str,
    *,
    model: Optional[str],
    search_profile: Optional[str],
    text_retention: Optional[str],
    chunking: Optional[ChunkingConfig],
    metadata_schema: Optional[MetadataSchema],
    idempotency_key: str,
) -> Call[Index]:
    body = compact(
        {
            "name": name,
            "model": model,
            "search_profile": search_profile,
            "text_retention": text_retention,
            "chunking": compact(dict(chunking)) if chunking is not None else None,
            "metadata_schema": dict(metadata_schema) if metadata_schema is not None else None,
        }
    )
    return Call("createIndex", _model_parser(Index), body=body, idempotency_key=idempotency_key)


def _documents(documents: Sequence[DocumentLike]) -> List[Dict[str, Any]]:
    if isinstance(documents, (str, bytes, Mapping, Document)):
        raise TypeError("documents must be a sequence of documents")
    return [_document(d, i, allow_str=False) for i, d in enumerate(documents)]


def upsert_call(
    index_id: str, documents: Sequence[DocumentLike], idempotency_key: str, timeout: Any = None
) -> Call[UpsertDocumentsResponse]:
    return Call(
        "upsertDocuments",
        _model_parser(UpsertDocumentsResponse),
        path_params={"index_id": require_id("index_id", index_id)},
        body={"documents": _documents(documents)},
        idempotency_key=idempotency_key,
        timeout=timeout,
    )


def get_document_call(
    index_id: str,
    doc_id: str,
    *,
    return_vectors: Optional[bool],
    encoding_format: Optional[EncodingFormat],
    dtype: Optional[TensorDtype],
) -> Call[StoredDocument]:
    query = {
        k: query_value(v)
        for k, v in compact(
            {"return_vectors": return_vectors, "encoding_format": encoding_format, "dtype": dtype}
        ).items()
    }
    return Call(
        "getDocument",
        _model_parser(StoredDocument),
        path_params={"index_id": require_id("index_id", index_id), "doc_id": require_id("doc_id", doc_id)},
        query=query or None,
    )


def delete_document_call(index_id: str, doc_id: str, idempotency_key: str) -> Call[DeleteDocumentResponse]:
    return Call(
        "deleteDocument",
        _model_parser(DeleteDocumentResponse),
        path_params={"index_id": require_id("index_id", index_id), "doc_id": require_id("doc_id", doc_id)},
        idempotency_key=idempotency_key,
    )


def get_index_call(index_id: str) -> Call[Index]:
    return Call("getIndex", _model_parser(Index), path_params={"index_id": require_id("index_id", index_id)})


def delete_index_call(index_id: str, idempotency_key: str) -> Call[DeleteIndexResponse]:
    return Call(
        "deleteIndex",
        _model_parser(DeleteIndexResponse),
        path_params={"index_id": require_id("index_id", index_id)},
        idempotency_key=idempotency_key,
    )


def _readiness(data: Any, meta: ResponseMeta) -> ReadinessStatus:
    return ReadinessStatus.from_dict(data)._with_meta(meta)


def _list_query(limit: Optional[int], cursor: Optional[str], **filters: Any) -> Dict[str, Any]:
    return compact({**filters, "limit": limit, "cursor": cursor})


# ---------------------------------------------------------------------------- sync resources


class _SyncResource:
    def __init__(self, client: BreadBowl) -> None:
        self._client = client

    def _list(
        self, operation_id: str, item_cls: Type[M], query: Dict[str, Any], cursor: Optional[str]
    ) -> SyncPage[M]:
        def fetch(cur: Optional[str]) -> SyncPage[M]:
            q = compact({**query, "cursor": cur})

            def parse(data: Any, meta: ResponseMeta) -> SyncPage[M]:
                items, has_more, next_cursor = parse_page_data(item_cls, data)
                return SyncPage(items, has_more, next_cursor, meta, cur, fetch)

            page: SyncPage[M] = self._client._call(Call(operation_id, parse, query=q))
            return page

        return fetch(cursor)


class Models(_SyncResource):
    def list(self, *, limit: Optional[int] = None, cursor: Optional[str] = None) -> SyncPage[Model]:
        """Models served by this deployment, default first. Iterate the page for all of them."""
        return self._list("listModels", Model, _list_query(limit, None), cursor)

    def get(self, model: str) -> Model:
        """A model by immutable id or alias."""
        return self._client._call(
            Call("getModel", _model_parser(Model), path_params={"model": require_id("model", model)})
        )


class Documents(_SyncResource):
    def upsert(
        self,
        index_id: str,
        documents: Sequence[DocumentLike],
        *,
        idempotency_key: Optional[str] = None,
    ) -> UpsertDocumentsResponse:
        """Upsert 1–50 documents (at most ``limits.max_chunks_per_upsert`` chunks). Always check
        ``.failed``: a 200 can contain per-document failures. Uses the client's
        ``upsert_timeout`` (180 s by default: about 3 chunks are encoded per second)."""
        key = idempotency_key or self._client.new_idempotency_key()
        return self._client._call(upsert_call(index_id, documents, key, self._client.upsert_timeout))

    def upsert_batched(
        self,
        index_id: str,
        documents: Sequence[DocumentLike],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> UpsertBatchResult:
        """Upsert any number of documents in requests of up to ``batch_size`` (≤ 50) documents.

        A batch rejected with ``chunk_limit_exceeded`` is split and retried; a single document
        over the chunk limit is reported as a failed outcome. Outcomes are merged in input order.
        Any other failure raises ``BatchUpsertError`` with the outcomes so far and the pending
        ``doc_id``s. Each request has its own ``Idempotency-Key``."""
        queue = batching.prepare(_documents(documents), batch_size)
        total = sum(len(b) for b in queue)
        outcomes: Dict[int, DocumentOutcome] = {}
        responses: List[UpsertDocumentsResponse] = []
        while queue:
            batch = queue.popleft()
            try:
                response = self.upsert(index_id, [doc for _, doc in batch])
            except ChunkLimitExceededError as err:
                if len(batch) == 1:
                    outcomes[batch[0][0]] = batching.too_large(batch[0][1]["doc_id"], err)
                else:
                    queue.extendleft(reversed(batching.split(batch, err)))
                continue
            except BreadBowlError as err:
                partial = batching.result(index_id, total, outcomes, responses)
                raise BatchUpsertError(err, partial, batching.pending_ids(batch, queue)) from err
            responses.append(response)
            batching.record(outcomes, batch, response)
        return batching.result(index_id, total, outcomes, responses)

    def get(
        self,
        index_id: str,
        doc_id: str,
        *,
        return_vectors: Optional[bool] = None,
        encoding_format: Optional[EncodingFormat] = None,
        dtype: Optional[TensorDtype] = None,
    ) -> StoredDocument:
        return self._client._call(
            get_document_call(
                index_id, doc_id, return_vectors=return_vectors, encoding_format=encoding_format, dtype=dtype
            )
        )

    def delete(
        self, index_id: str, doc_id: str, *, idempotency_key: Optional[str] = None
    ) -> DeleteDocumentResponse:
        """Delete a document; ``deleted`` is ``False`` when it did not exist."""
        key = idempotency_key or self._client.new_idempotency_key()
        return self._client._call(delete_document_call(index_id, doc_id, key))


class Indexes(_SyncResource):
    def __init__(self, client: BreadBowl) -> None:
        super().__init__(client)
        self.documents = Documents(client)

    def create(
        self,
        name: str,
        *,
        model: Optional[str] = None,
        search_profile: Optional[str] = None,
        chunking: Optional[ChunkingConfig] = None,
        metadata_schema: Optional[MetadataSchema] = None,
        text_retention: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Index:
        key = idempotency_key or self._client.new_idempotency_key()
        return self._client._call(
            create_index_call(
                name,
                model=model,
                search_profile=search_profile,
                text_retention=text_retention,
                chunking=chunking,
                metadata_schema=metadata_schema,
                idempotency_key=key,
            )
        )

    def list(
        self, *, name: Optional[str] = None, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> SyncPage[Index]:
        """Indexes, newest first. Iterate the page to walk every page."""
        return self._list("listIndexes", Index, _list_query(limit, None, name=name), cursor)

    def get(self, index_id: str) -> Index:
        return self._client._call(get_index_call(index_id))

    def delete(self, index_id: str, *, idempotency_key: Optional[str] = None) -> DeleteIndexResponse:
        key = idempotency_key or self._client.new_idempotency_key()
        return self._client._call(delete_index_call(index_id, key))

    def search(
        self,
        index_id: str,
        query: str,
        *,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        filter: Optional[MetadataFilter] = None,
        instruction: Optional[str] = None,
        explain: Optional[bool] = None,
        return_attention: Optional[bool] = None,
        return_vectors: Optional[bool] = None,
        attention_heads: Optional[AttentionHeads] = None,
        encoding_format: Optional[EncodingFormat] = None,
        dtype: Optional[TensorDtype] = None,
    ) -> SearchResponse:
        return self._client._call(
            search_call(
                index_id,
                query,
                top_k=top_k,
                candidate_k=candidate_k,
                filter=filter,
                instruction=instruction,
                explain=explain,
                return_attention=return_attention,
                return_vectors=return_vectors,
                attention_heads=attention_heads,
                encoding_format=encoding_format,
                dtype=dtype,
            )
        )


class IndexHandle:
    """Operations bound to one index: ``client.index("idx_...")``."""

    def __init__(self, client: BreadBowl, index_id: str) -> None:
        self._client = client
        self.id = require_id("index_id", index_id)

    def get(self) -> Index:
        return self._client.indexes.get(self.id)

    def delete(self, *, idempotency_key: Optional[str] = None) -> DeleteIndexResponse:
        return self._client.indexes.delete(self.id, idempotency_key=idempotency_key)

    def search(self, query: str, **kwargs: Any) -> SearchResponse:
        return self._client.indexes.search(self.id, query, **kwargs)

    def upsert(
        self, documents: Sequence[DocumentLike], *, idempotency_key: Optional[str] = None
    ) -> UpsertDocumentsResponse:
        return self._client.indexes.documents.upsert(self.id, documents, idempotency_key=idempotency_key)

    def upsert_batched(
        self, documents: Sequence[DocumentLike], *, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> UpsertBatchResult:
        return self._client.indexes.documents.upsert_batched(self.id, documents, batch_size=batch_size)

    def get_document(self, doc_id: str, **kwargs: Any) -> StoredDocument:
        return self._client.indexes.documents.get(self.id, doc_id, **kwargs)

    def delete_document(
        self, doc_id: str, *, idempotency_key: Optional[str] = None
    ) -> DeleteDocumentResponse:
        return self._client.indexes.documents.delete(self.id, doc_id, idempotency_key=idempotency_key)

    def __repr__(self) -> str:
        return f"IndexHandle({self.id!r})"


class UsageResource(_SyncResource):
    def get(self) -> UsageSummary:
        """Current-month usage, limits, remaining allowance and the calling key's settings."""
        return self._client._call(Call("getUsage", _model_parser(UsageSummary)))


class Health(_SyncResource):
    def live(self) -> HealthStatus:
        """``GET /healthz`` (no key needed)."""
        return self._client._call(Call("getHealth", _model_parser(HealthStatus)))

    def ready(self) -> ReadinessStatus:
        """``GET /readyz``. Returns ``ok=False`` on 503 instead of raising."""
        return self._client._call(Call("getReadiness", _readiness, accept_statuses=(503,)))


# ---------------------------------------------------------------------------- async resources


class _AsyncResource:
    def __init__(self, client: AsyncBreadBowl) -> None:
        self._client = client

    def _list(
        self, operation_id: str, item_cls: Type[M], query: Dict[str, Any], cursor: Optional[str]
    ) -> AsyncPaginator[M]:
        def fetch(cur: Optional[str]) -> Awaitable[AsyncPage[M]]:
            q = compact({**query, "cursor": cur})

            def parse(data: Any, meta: ResponseMeta) -> AsyncPage[M]:
                items, has_more, next_cursor = parse_page_data(item_cls, data)
                return AsyncPage(items, has_more, next_cursor, meta, cur, fetch)

            result: Awaitable[AsyncPage[M]] = self._client._call(Call(operation_id, parse, query=q))
            return result

        return AsyncPaginator(lambda: fetch(cursor))


class AsyncModels(_AsyncResource):
    def list(self, *, limit: Optional[int] = None, cursor: Optional[str] = None) -> AsyncPaginator[Model]:
        return self._list("listModels", Model, _list_query(limit, None), cursor)

    async def get(self, model: str) -> Model:
        result: Model = await self._client._call(
            Call("getModel", _model_parser(Model), path_params={"model": require_id("model", model)})
        )
        return result


class AsyncDocuments(_AsyncResource):
    async def upsert(
        self,
        index_id: str,
        documents: Sequence[DocumentLike],
        *,
        idempotency_key: Optional[str] = None,
    ) -> UpsertDocumentsResponse:
        key = idempotency_key or self._client.new_idempotency_key()
        result: UpsertDocumentsResponse = await self._client._call(
            upsert_call(index_id, documents, key, self._client.upsert_timeout)
        )
        return result

    async def upsert_batched(
        self,
        index_id: str,
        documents: Sequence[DocumentLike],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> UpsertBatchResult:
        """Async :meth:`Documents.upsert_batched`."""
        queue = batching.prepare(_documents(documents), batch_size)
        total = sum(len(b) for b in queue)
        outcomes: Dict[int, DocumentOutcome] = {}
        responses: List[UpsertDocumentsResponse] = []
        while queue:
            batch = queue.popleft()
            try:
                response = await self.upsert(index_id, [doc for _, doc in batch])
            except ChunkLimitExceededError as err:
                if len(batch) == 1:
                    outcomes[batch[0][0]] = batching.too_large(batch[0][1]["doc_id"], err)
                else:
                    queue.extendleft(reversed(batching.split(batch, err)))
                continue
            except BreadBowlError as err:
                partial = batching.result(index_id, total, outcomes, responses)
                raise BatchUpsertError(err, partial, batching.pending_ids(batch, queue)) from err
            responses.append(response)
            batching.record(outcomes, batch, response)
        return batching.result(index_id, total, outcomes, responses)

    async def get(
        self,
        index_id: str,
        doc_id: str,
        *,
        return_vectors: Optional[bool] = None,
        encoding_format: Optional[EncodingFormat] = None,
        dtype: Optional[TensorDtype] = None,
    ) -> StoredDocument:
        result: StoredDocument = await self._client._call(
            get_document_call(
                index_id, doc_id, return_vectors=return_vectors, encoding_format=encoding_format, dtype=dtype
            )
        )
        return result

    async def delete(
        self, index_id: str, doc_id: str, *, idempotency_key: Optional[str] = None
    ) -> DeleteDocumentResponse:
        key = idempotency_key or self._client.new_idempotency_key()
        result: DeleteDocumentResponse = await self._client._call(delete_document_call(index_id, doc_id, key))
        return result


class AsyncIndexes(_AsyncResource):
    def __init__(self, client: AsyncBreadBowl) -> None:
        super().__init__(client)
        self.documents = AsyncDocuments(client)

    async def create(
        self,
        name: str,
        *,
        model: Optional[str] = None,
        search_profile: Optional[str] = None,
        chunking: Optional[ChunkingConfig] = None,
        metadata_schema: Optional[MetadataSchema] = None,
        text_retention: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Index:
        key = idempotency_key or self._client.new_idempotency_key()
        result: Index = await self._client._call(
            create_index_call(
                name,
                model=model,
                search_profile=search_profile,
                text_retention=text_retention,
                chunking=chunking,
                metadata_schema=metadata_schema,
                idempotency_key=key,
            )
        )
        return result

    def list(
        self, *, name: Optional[str] = None, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> AsyncPaginator[Index]:
        return self._list("listIndexes", Index, _list_query(limit, None, name=name), cursor)

    async def get(self, index_id: str) -> Index:
        result: Index = await self._client._call(get_index_call(index_id))
        return result

    async def delete(self, index_id: str, *, idempotency_key: Optional[str] = None) -> DeleteIndexResponse:
        key = idempotency_key or self._client.new_idempotency_key()
        result: DeleteIndexResponse = await self._client._call(delete_index_call(index_id, key))
        return result

    async def search(
        self,
        index_id: str,
        query: str,
        *,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        filter: Optional[MetadataFilter] = None,
        instruction: Optional[str] = None,
        explain: Optional[bool] = None,
        return_attention: Optional[bool] = None,
        return_vectors: Optional[bool] = None,
        attention_heads: Optional[AttentionHeads] = None,
        encoding_format: Optional[EncodingFormat] = None,
        dtype: Optional[TensorDtype] = None,
    ) -> SearchResponse:
        result: SearchResponse = await self._client._call(
            search_call(
                index_id,
                query,
                top_k=top_k,
                candidate_k=candidate_k,
                filter=filter,
                instruction=instruction,
                explain=explain,
                return_attention=return_attention,
                return_vectors=return_vectors,
                attention_heads=attention_heads,
                encoding_format=encoding_format,
                dtype=dtype,
            )
        )
        return result


class AsyncIndexHandle:
    def __init__(self, client: AsyncBreadBowl, index_id: str) -> None:
        self._client = client
        self.id = require_id("index_id", index_id)

    async def get(self) -> Index:
        return await self._client.indexes.get(self.id)

    async def delete(self, *, idempotency_key: Optional[str] = None) -> DeleteIndexResponse:
        return await self._client.indexes.delete(self.id, idempotency_key=idempotency_key)

    async def search(self, query: str, **kwargs: Any) -> SearchResponse:
        return await self._client.indexes.search(self.id, query, **kwargs)

    async def upsert(
        self, documents: Sequence[DocumentLike], *, idempotency_key: Optional[str] = None
    ) -> UpsertDocumentsResponse:
        return await self._client.indexes.documents.upsert(
            self.id, documents, idempotency_key=idempotency_key
        )

    async def upsert_batched(
        self, documents: Sequence[DocumentLike], *, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> UpsertBatchResult:
        return await self._client.indexes.documents.upsert_batched(self.id, documents, batch_size=batch_size)

    async def get_document(self, doc_id: str, **kwargs: Any) -> StoredDocument:
        return await self._client.indexes.documents.get(self.id, doc_id, **kwargs)

    async def delete_document(
        self, doc_id: str, *, idempotency_key: Optional[str] = None
    ) -> DeleteDocumentResponse:
        return await self._client.indexes.documents.delete(self.id, doc_id, idempotency_key=idempotency_key)

    def __repr__(self) -> str:
        return f"AsyncIndexHandle({self.id!r})"


class AsyncUsageResource(_AsyncResource):
    async def get(self) -> UsageSummary:
        result: UsageSummary = await self._client._call(Call("getUsage", _model_parser(UsageSummary)))
        return result


class AsyncHealth(_AsyncResource):
    async def live(self) -> HealthStatus:
        result: HealthStatus = await self._client._call(Call("getHealth", _model_parser(HealthStatus)))
        return result

    async def ready(self) -> ReadinessStatus:
        result: ReadinessStatus = await self._client._call(
            Call("getReadiness", _readiness, accept_statuses=(503,))
        )
        return result
