from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
import pytest

from breadbowl import AsyncBreadBowl, BreadBowl

HERE = Path(__file__).resolve().parent
FIXTURES: Dict[str, Any] = json.loads((HERE / "fixtures" / "contract.json").read_text(encoding="utf-8"))
REPO_ROOT = HERE.parents[2]
SPEC_PATH = REPO_ROOT / "sdks" / "openapi" / "bb-embed.v1.yaml"
SIBLING_FIXTURE = REPO_ROOT / "sdks" / "typescript" / "test" / "fixtures" / "contract.json"
BASE_URL = "https://api.test"
API_KEY = "test-key-not-real"


def example(operation_id: str, status: Union[int, str] = 200, name: Optional[str] = None) -> Dict[str, Any]:
    """A response example from the contract (verbatim)."""
    examples = FIXTURES["operations"][operation_id]["responses"][str(status)]["examples"]
    key = name if name is not None else next(iter(examples))
    return copy.deepcopy(examples[key])


def expanded(operation_id: str, status: Union[int, str], name: str) -> Dict[str, Any]:
    """A response example whose abbreviated tensors were tiled to their full shape."""
    return copy.deepcopy(FIXTURES["expanded"][f"{operation_id}/{status}/{name}"])


def request_example(operation_id: str, name: str = "default") -> Dict[str, Any]:
    return copy.deepcopy(FIXTURES["operations"][operation_id]["request_examples"][name])


def error_body(code: str) -> Dict[str, Any]:
    for e in FIXTURES["errors"]:
        if e["code"] == code:
            return copy.deepcopy(e["body"])
    raise KeyError(code)


@dataclass
class Recorded:
    method: str
    path: str
    raw_path: str
    params: Dict[str, str]
    headers: httpx.Headers
    body: Any


@dataclass
class MockAPI:
    """A scripted server: queue responses (or exceptions), then inspect the recorded requests."""

    queue: List[Any] = field(default_factory=list)
    requests: List[Recorded] = field(default_factory=list)
    sleeps: List[float] = field(default_factory=list)

    def add(
        self,
        json_body: Any = None,
        status: int = 200,
        headers: Optional[Dict[str, str]] = None,
        *,
        text: Optional[str] = None,
        request_id: Optional[str] = "req-test-1",
    ) -> MockAPI:
        hdrs = dict(headers or {})
        if request_id is not None:
            hdrs.setdefault("x-request-id", request_id)
        self.queue.append(("response", status, json_body, hdrs, text))
        return self

    def fail(self, exc_type: Callable[..., Exception], message: str = "boom") -> MockAPI:
        self.queue.append(("raise", exc_type, message))
        return self

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append(
            Recorded(
                method=request.method,
                path=request.url.path,
                raw_path=request.url.raw_path.decode("ascii").split("?")[0],
                params=dict(request.url.params),
                headers=request.headers,
                body=body,
            )
        )
        if not self.queue:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        item = self.queue.pop(0)
        if item[0] == "raise":
            raise item[1](item[2], request=request)
        _, status, json_body, headers, text = item
        if text is not None:
            return httpx.Response(status, text=text, headers=headers)
        if json_body is None:
            return httpx.Response(status, headers=headers)
        return httpx.Response(status, json=json_body, headers=headers)

    @property
    def last(self) -> Recorded:
        return self.requests[-1]


LOOPBACK = {"127.0.0.1", "localhost", "::1"}


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only MockTransport and loopback (the in-process reference API of the integration tests)
    are allowed; the default base URL is a live service and must never be contacted."""
    real_sync = httpx.HTTPTransport.handle_request
    real_async = httpx.AsyncHTTPTransport.handle_async_request

    def guarded(self: Any, request: httpx.Request) -> Any:
        if request.url.host not in LOOPBACK:
            raise AssertionError(f"real network request attempted: {request.method} {request.url}")
        return real_sync(self, request)

    async def guarded_async(self: Any, request: httpx.Request) -> Any:
        if request.url.host not in LOOPBACK:
            raise AssertionError(f"real network request attempted: {request.method} {request.url}")
        return await real_async(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guarded)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", guarded_async)


@pytest.fixture
def api() -> MockAPI:
    return MockAPI()


@pytest.fixture
def client(api: MockAPI) -> BreadBowl:
    c = BreadBowl(
        API_KEY, base_url=BASE_URL, http_client=httpx.Client(transport=httpx.MockTransport(api.handler))
    )
    c._sleep = api.sleeps.append
    return c


@pytest.fixture
def aclient(api: MockAPI) -> AsyncBreadBowl:
    c = AsyncBreadBowl(
        API_KEY, base_url=BASE_URL, http_client=httpx.AsyncClient(transport=httpx.MockTransport(api.handler))
    )

    async def sleep(seconds: float) -> None:
        api.sleeps.append(seconds)

    c._sleep = sleep
    return c


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
