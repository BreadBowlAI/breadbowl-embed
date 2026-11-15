"""Cursor pagination: ``{data, has_more, next_cursor}`` pages and item iterators."""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
)

from ._errors import APIResponseError
from ._models import BaseModel, ResponseMeta

__all__ = ["SyncPage", "AsyncPage", "AsyncPaginator"]

T = TypeVar("T", bound=BaseModel)


def parse_page_data[T: BaseModel](item_cls: Type[T], data: Any) -> tuple[List[T], bool, Optional[str]]:
    items = [item_cls.from_dict(x) for x in data.get("data") or []]
    has_more = bool(data.get("has_more"))
    next_cursor = data.get("next_cursor")
    return items, has_more, next_cursor if isinstance(next_cursor, str) else None


class _PageBase[T: BaseModel]:
    def __init__(
        self,
        data: List[T],
        has_more: bool,
        next_cursor: Optional[str],
        meta: ResponseMeta,
        cursor: Optional[str],
    ) -> None:
        self.data = data
        self.has_more = has_more
        self.next_cursor = next_cursor
        self.meta = meta
        self._cursor = cursor

    @property
    def request_id(self) -> Optional[str]:
        return self.meta.request_id

    def has_next_page(self) -> bool:
        return self.has_more

    def _check_progress(self) -> str:
        # The contract guarantees next_cursor is a string exactly when has_more is true.
        if self.next_cursor is None:
            raise APIResponseError(
                "server returned has_more: true without a next_cursor",
                status_code=self.meta.status_code,
                request_id=self.meta.request_id,
            )
        if self.next_cursor == self._cursor:
            raise APIResponseError(
                "server returned the same cursor twice; stopping to avoid an endless loop",
                status_code=self.meta.status_code,
                request_id=self.meta.request_id,
            )
        return self.next_cursor

    def __repr__(self) -> str:
        return f"{type(self).__name__}(items={len(self.data)}, has_more={self.has_more})"


class SyncPage(_PageBase[T]):
    """One page. Iterating it yields every item of this page and the following ones."""

    def __init__(
        self,
        data: List[T],
        has_more: bool,
        next_cursor: Optional[str],
        meta: ResponseMeta,
        cursor: Optional[str],
        fetch: Callable[[str], SyncPage[T]],
    ) -> None:
        super().__init__(data, has_more, next_cursor, meta, cursor)
        self._fetch = fetch

    def next_page(self) -> Optional[SyncPage[T]]:
        """The next page, or ``None`` after the last one."""
        if not self.has_next_page():
            return None
        return self._fetch(self._check_progress())

    def iter_pages(self) -> Iterator[SyncPage[T]]:
        page: Optional[SyncPage[T]] = self
        while page is not None:
            yield page
            page = page.next_page()

    def __iter__(self) -> Iterator[T]:
        for page in self.iter_pages():
            yield from page.data


class AsyncPage(_PageBase[T]):
    """One page. ``async for`` over it yields every item of this page and the following ones."""

    def __init__(
        self,
        data: List[T],
        has_more: bool,
        next_cursor: Optional[str],
        meta: ResponseMeta,
        cursor: Optional[str],
        fetch: Callable[[str], Awaitable[AsyncPage[T]]],
    ) -> None:
        super().__init__(data, has_more, next_cursor, meta, cursor)
        self._fetch = fetch

    async def next_page(self) -> Optional[AsyncPage[T]]:
        if not self.has_next_page():
            return None
        return await self._fetch(self._check_progress())

    async def iter_pages(self) -> AsyncIterator[AsyncPage[T]]:
        page: Optional[AsyncPage[T]] = self
        while page is not None:
            yield page
            page = await page.next_page()

    async def __aiter__(self) -> AsyncIterator[T]:
        async for page in self.iter_pages():
            for item in page.data:
                yield item


class AsyncPaginator[T: BaseModel]:
    """What async ``list()`` returns: ``await`` it for the first page, or ``async for`` over
    it for every item across pages."""

    def __init__(self, first: Callable[[], Awaitable[AsyncPage[T]]]) -> None:
        self._first = first

    def __await__(self) -> Generator[Any, None, AsyncPage[T]]:
        return self._first().__await__()

    async def __aiter__(self) -> AsyncIterator[T]:
        page = await self._first()
        async for item in page:
            yield item
