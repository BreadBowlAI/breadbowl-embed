"""``BreadBowl`` (sync) and ``AsyncBreadBowl`` (async) clients."""

from __future__ import annotations

import time
from types import TracebackType
from typing import Any, Callable, Mapping, Optional, Sequence, Type, TypeVar, Union

import anyio
import httpx

from ._base import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_RETRY_WAIT,
    BaseClient,
    Call,
    retry_kind,
)
from ._generated import AttentionHeads, EncodeReturnField, EncodingFormat, InputType, TensorDtype
from ._models import EncodeResponse, ScoreResponse
from ._resources import (
    AsyncHealth,
    AsyncIndexes,
    AsyncIndexHandle,
    AsyncModels,
    AsyncUsageResource,
    DocumentLike,
    Health,
    Indexes,
    IndexHandle,
    Models,
    UsageResource,
    encode_call,
    score_call,
)

__all__ = ["BreadBowl", "AsyncBreadBowl"]

T = TypeVar("T")

_UNSET: Any = object()


class BreadBowl(BaseClient):
    """Synchronous client.

    >>> client = BreadBowl()  # reads BREADBOWL_API_KEY
    >>> hits = client.indexes.search("idx_123", "How do I rotate a key?", top_k=5)

    ``base_url`` defaults to ``BREADBOWL_BASE_URL`` or the hosted service; point it at a
    self-hosted deployment as needed. Pass ``http_client`` to control proxies, TLS or the
    transport (tests use ``httpx.MockTransport``).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_retry_wait: float = DEFAULT_MAX_RETRY_WAIT,
        default_headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        upsert_timeout: Union[float, httpx.Timeout, None] = None,
    ) -> None:
        super().__init__(
            api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            max_retry_wait=max_retry_wait,
            default_headers=default_headers,
            upsert_timeout=upsert_timeout,
        )
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.Client()
        self._sleep: Callable[[float], None] = time.sleep
        self.models = Models(self)
        self.indexes = Indexes(self)
        self.usage = UsageResource(self)
        self.health = Health(self)

    # ------------------------------------------------------------------ operations
    def encode(
        self,
        input: Union[str, Sequence[str]],
        *,
        input_type: InputType,
        model: Optional[str] = None,
        instruction: Optional[str] = None,
        return_: Optional[Sequence[EncodeReturnField]] = None,
        encoding_format: Optional[EncodingFormat] = None,
        dtype: Optional[TensorDtype] = None,
        attention_heads: Optional[AttentionHeads] = None,
    ) -> EncodeResponse:
        """Encode 1–50 queries or documents. ``return_`` is the wire field ``return``
        (default ``["routing", "value"]``)."""
        return self._call(
            encode_call(
                input,
                input_type=input_type,
                model=model,
                instruction=instruction,
                return_=return_,
                encoding_format=encoding_format,
                dtype=dtype,
                attention_heads=attention_heads,
            )
        )

    def score(
        self,
        query: str,
        documents: Sequence[Union[DocumentLike, str]],
        *,
        model: Optional[str] = None,
        instruction: Optional[str] = None,
        explain: Optional[bool] = None,
        return_attention: Optional[bool] = None,
        attention_heads: Optional[AttentionHeads] = None,
        encoding_format: Optional[EncodingFormat] = None,
        dtype: Optional[TensorDtype] = None,
    ) -> ScoreResponse:
        """Rank 1–50 transient documents against one query. A plain string document gets its
        position as ``doc_id``."""
        return self._call(
            score_call(
                query,
                documents,
                model=model,
                instruction=instruction,
                explain=explain,
                return_attention=return_attention,
                attention_heads=attention_heads,
                encoding_format=encoding_format,
                dtype=dtype,
            )
        )

    def index(self, index_id: str) -> IndexHandle:
        """A handle bound to one index (``.search``, ``.upsert``, ``.get_document``, ...)."""
        return IndexHandle(self, index_id)

    def with_options(
        self,
        *,
        timeout: Union[float, httpx.Timeout, None] = _UNSET,
        max_retries: int = _UNSET,
        max_retry_wait: float = _UNSET,
        upsert_timeout: Union[float, httpx.Timeout, None] = _UNSET,
    ) -> BreadBowl:
        """A copy with other timeout/retry settings that shares this client's connection pool."""
        clone = BreadBowl(
            self.api_key,
            base_url=self.base_url,
            timeout=self.timeout if timeout is _UNSET else timeout,
            max_retries=self.max_retries if max_retries is _UNSET else max_retries,
            max_retry_wait=self.max_retry_wait if max_retry_wait is _UNSET else max_retry_wait,
            default_headers=self.default_headers,
            http_client=self._http,
            upsert_timeout=self.upsert_timeout if upsert_timeout is _UNSET else upsert_timeout,
        )
        clone._sleep = self._sleep
        return clone

    # ------------------------------------------------------------------ plumbing
    def _call(self, call: Call[T]) -> T:
        op = call.operation
        url = self._url(op, call.path_params)
        headers = self._headers(call)
        content = self._content(call)
        kind = retry_kind(call.operation_id, op)
        attempt = 0
        while True:
            request = self._http.build_request(
                op.method,
                url,
                params=call.query,
                content=content,
                headers=headers,
                timeout=self.timeout if call.timeout is None else call.timeout,
            )
            try:
                response = self._http.send(request)
            except httpx.TransportError as exc:
                delay = self._retry_delay(kind, attempt, exc=exc)
                if delay is None:
                    raise self._connection_error(exc, request) from exc
            else:
                if response.is_success or response.status_code in call.accept_statuses:
                    return self._parse_success(response, call)
                error = self._make_error(response)
                delay = self._retry_delay(kind, attempt, error=error)
                if delay is None:
                    raise error
            self._sleep(delay)
            attempt += 1

    def close(self) -> None:
        """Close the connection pool (only if this client created it)."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> BreadBowl:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"BreadBowl(base_url={self.base_url!r})"


class AsyncBreadBowl(BaseClient):
    """Asynchronous client (asyncio or trio). Same surface as :class:`BreadBowl`, awaited.

    ``list()`` methods return a paginator: ``await`` it for the first page or ``async for``
    over it for every item.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_retry_wait: float = DEFAULT_MAX_RETRY_WAIT,
        default_headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        upsert_timeout: Union[float, httpx.Timeout, None] = None,
    ) -> None:
        super().__init__(
            api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            max_retry_wait=max_retry_wait,
            default_headers=default_headers,
            upsert_timeout=upsert_timeout,
        )
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.AsyncClient()
        self._sleep: Callable[[float], Any] = anyio.sleep
        self.models = AsyncModels(self)
        self.indexes = AsyncIndexes(self)
        self.usage = AsyncUsageResource(self)
        self.health = AsyncHealth(self)

    async def encode(
        self,
        input: Union[str, Sequence[str]],
        *,
        input_type: InputType,
        model: Optional[str] = None,
        instruction: Optional[str] = None,
        return_: Optional[Sequence[EncodeReturnField]] = None,
        encoding_format: Optional[EncodingFormat] = None,
        dtype: Optional[TensorDtype] = None,
        attention_heads: Optional[AttentionHeads] = None,
    ) -> EncodeResponse:
        result: EncodeResponse = await self._call(
            encode_call(
                input,
                input_type=input_type,
                model=model,
                instruction=instruction,
                return_=return_,
                encoding_format=encoding_format,
                dtype=dtype,
                attention_heads=attention_heads,
            )
        )
        return result

    async def score(
        self,
        query: str,
        documents: Sequence[Union[DocumentLike, str]],
        *,
        model: Optional[str] = None,
        instruction: Optional[str] = None,
        explain: Optional[bool] = None,
        return_attention: Optional[bool] = None,
        attention_heads: Optional[AttentionHeads] = None,
        encoding_format: Optional[EncodingFormat] = None,
        dtype: Optional[TensorDtype] = None,
    ) -> ScoreResponse:
        result: ScoreResponse = await self._call(
            score_call(
                query,
                documents,
                model=model,
                instruction=instruction,
                explain=explain,
                return_attention=return_attention,
                attention_heads=attention_heads,
                encoding_format=encoding_format,
                dtype=dtype,
            )
        )
        return result

    def index(self, index_id: str) -> AsyncIndexHandle:
        return AsyncIndexHandle(self, index_id)

    def with_options(
        self,
        *,
        timeout: Union[float, httpx.Timeout, None] = _UNSET,
        max_retries: int = _UNSET,
        max_retry_wait: float = _UNSET,
        upsert_timeout: Union[float, httpx.Timeout, None] = _UNSET,
    ) -> AsyncBreadBowl:
        clone = AsyncBreadBowl(
            self.api_key,
            base_url=self.base_url,
            timeout=self.timeout if timeout is _UNSET else timeout,
            max_retries=self.max_retries if max_retries is _UNSET else max_retries,
            max_retry_wait=self.max_retry_wait if max_retry_wait is _UNSET else max_retry_wait,
            default_headers=self.default_headers,
            http_client=self._http,
            upsert_timeout=self.upsert_timeout if upsert_timeout is _UNSET else upsert_timeout,
        )
        clone._sleep = self._sleep
        return clone

    async def _call(self, call: Call[T]) -> T:
        op = call.operation
        url = self._url(op, call.path_params)
        headers = self._headers(call)
        content = self._content(call)
        kind = retry_kind(call.operation_id, op)
        attempt = 0
        while True:
            request = self._http.build_request(
                op.method,
                url,
                params=call.query,
                content=content,
                headers=headers,
                timeout=self.timeout if call.timeout is None else call.timeout,
            )
            try:
                response = await self._http.send(request)
            except httpx.TransportError as exc:
                delay = self._retry_delay(kind, attempt, exc=exc)
                if delay is None:
                    raise self._connection_error(exc, request) from exc
            else:
                if response.is_success or response.status_code in call.accept_statuses:
                    return self._parse_success(response, call)
                error = self._make_error(response)
                delay = self._retry_delay(kind, attempt, error=error)
                if delay is None:
                    raise error
            await self._sleep(delay)
            attempt += 1

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncBreadBowl:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"AsyncBreadBowl(base_url={self.base_url!r})"
