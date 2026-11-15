from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from breadbowl import APIResponseError, AsyncBreadBowl, AsyncPage, BreadBowl, Index, SyncPage
from tests.conftest import MockAPI, example


def page(names: List[str], cursor: Optional[str]) -> Dict[str, Any]:
    template = example("listIndexes")["data"][0]
    data = [dict(template, id=f"id-{n}", name=n) for n in names]
    return {"data": data, "has_more": cursor is not None, "next_cursor": cursor}


def queue_three_pages(api: MockAPI) -> None:
    api.add(page(["a", "b"], "c1"), request_id="r1")
    api.add(page(["c", "d"], "c2"), request_id="r2")
    api.add(page(["e"], None), request_id="r3")


def test_iterating_a_page_walks_every_page(api: MockAPI, client: BreadBowl) -> None:
    queue_three_pages(api)
    first = client.indexes.list(limit=2, name="x")
    assert isinstance(first, SyncPage) and [i.name for i in first.data] == ["a", "b"]
    assert all(isinstance(i, Index) for i in first.data)
    names = [i.name for i in first]
    assert names == ["a", "b", "c", "d", "e"]
    assert [r.params for r in api.requests] == [
        {"limit": "2", "name": "x"},
        {"limit": "2", "name": "x", "cursor": "c1"},
        {"limit": "2", "name": "x", "cursor": "c2"},
    ]


def test_iter_pages_and_request_ids(api: MockAPI, client: BreadBowl) -> None:
    queue_three_pages(api)
    pages = list(client.indexes.list().iter_pages())
    assert [p.request_id for p in pages] == ["r1", "r2", "r3"]
    assert pages[-1].next_page() is None and not pages[-1].has_next_page()


def test_starting_cursor_is_sent(api: MockAPI, client: BreadBowl) -> None:
    api.add(page(["z"], None))
    client.models.list(cursor="abc")
    assert api.last.params == {"cursor": "abc"}


def test_has_more_without_cursor_is_a_contract_violation(api: MockAPI, client: BreadBowl) -> None:
    api.add({"data": [], "has_more": True, "next_cursor": None})
    with pytest.raises(APIResponseError, match="without a next_cursor"):
        list(client.indexes.list())


def test_repeated_cursor_is_an_error(api: MockAPI, client: BreadBowl) -> None:
    api.add(page(["a"], "same")).add(page(["b"], "same"))
    with pytest.raises(APIResponseError, match="same cursor"):
        list(client.indexes.list())


@pytest.mark.anyio
async def test_async_paginator(api: MockAPI, aclient: AsyncBreadBowl) -> None:
    queue_three_pages(api)
    names = [i.name async for i in aclient.indexes.list(limit=2)]
    assert names == ["a", "b", "c", "d", "e"]
    queue_three_pages(api)
    first = await aclient.indexes.list(limit=2)
    assert isinstance(first, AsyncPage) and first.request_id == "r1"
    second = await first.next_page()
    assert second is not None and [i.name for i in second.data] == ["c", "d"]
    pages = [p async for p in second.iter_pages()]
    assert [p.request_id for p in pages] == ["r2", "r3"]


@pytest.mark.anyio
async def test_async_models_list(api: MockAPI, aclient: AsyncBreadBowl) -> None:
    api.add(example("listModels"))
    models = [m.id async for m in aclient.models.list()]
    assert models == [example("listModels")["data"][0]["id"]]
