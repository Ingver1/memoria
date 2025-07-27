import asyncio
import json
from types import TracebackType
from typing import Any, Callable, Optional, cast
from urllib.parse import urlencode, urlparse, parse_qs

from fastapi.testclient import ClientHelper


class _AsyncLock:
    """Lightweight asyncio lock compatible with older Python versions."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> None:
        await self._lock.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._lock.release()


class URL:
    """Very small URL representation used in tests."""

    def __init__(self, url: str) -> None:
        self._url = url
        parsed = urlparse(url)
        self.path = parsed.path
        self.params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed.query).items()}

    def __str__(self) -> str:  # noqa: D401
        return self._url


class Request:
    """Minimal ``httpx.Request`` used by ``MockTransport``."""

    def __init__(self, method: str, url: str, *, content: bytes | None = None) -> None:
        self.method = method
        self.url = URL(url)
        self.content = content or b""


class Response:
    """Minimal ``httpx.Response`` stub."""

    def __init__(self, resp: Any, *, json: Any | None = None, content: bytes | None = None) -> None:
        if isinstance(resp, int):
            self._resp: Any | None = None
            self._status_code = resp
            self._json = json
            self._content = content
            self._headers: dict[str, str] = {}
            self._cookies: dict[str, str] = {}
        else:
            self._resp = resp

    @property
    def status_code(self) -> int:
        if self._resp is not None:
            return cast(int, self._resp.status_code)
        return self._status_code

    def json(self) -> Any:
        if self._resp is not None:
            return self._resp.json()
        return self._json

    @property
    def text(self) -> str:
        if self._resp is not None:
            return cast(str, getattr(self._resp, "text", ""))
        if self._content is not None:
            return self._content.decode()
        if self._json is not None:
            return json.dumps(self._json)
        return ""

    @property
    def content(self) -> bytes:
        if self._resp is not None:
            return getattr(self._resp, "content", b"")
        if self._content is not None:
            return self._content
        if self._json is not None:
            return json.dumps(self._json).encode()
        return b""

    @property
    def headers(self) -> Any:
        if self._resp is not None:
            return self._resp.headers
        return self._headers

    @property
    def cookies(self) -> Any:
        if self._resp is not None:
            return getattr(self._resp, "cookies", {})
        return self._cookies

    def raise_for_status(self) -> None:
        if self._resp is not None and hasattr(self._resp, "raise_for_status"):
            self._resp.raise_for_status()
        elif self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def __repr__(self) -> str:
        return f"<Response status={self.status_code}>"


class MockTransport:
    """Simple transport that delegates to a handler function."""

    def __init__(self, handler: Callable[[Request], Response]) -> None:
        self._handler = handler

    async def __call__(self, request: Request) -> Response:
        resp = self._handler(request)
        if asyncio.iscoroutine(resp):
            resp = await resp
        assert isinstance(resp, Response)
        return resp


class AsyncClient:
    """Minimal httpx.AsyncClient stub for FastAPI testing."""

    def __init__(
        self,
        app: Any | None = None,
        base_url: str = "http://test",
        timeout: float | None = None,
        transport: MockTransport | None = None,
    ) -> None:
        self._app = app
        self._base_url = base_url.rstrip("/")
        self._client: Optional[ClientHelper] = None
        self._timeout = timeout
        self._lock = _AsyncLock()
        self._transport = transport

    async def __aenter__(self) -> "AsyncClient":
        if self._transport is None:
            self._client = ClientHelper(cast(Any, self._app), base_url=self._base_url)
            try:
                asyncio.get_running_loop()
                await self._client._startup()
            except RuntimeError:
                # No running loop, fall back to synchronous context manager
                self._client.__enter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:
        if self._client:
            try:
                asyncio.get_running_loop()
                await self._client._shutdown()
            except RuntimeError:
                self._client.__exit__(exc_type, exc, tb)

    async def _request(self, method: str, url: str, **kwargs: Any) -> Response:
        if self._transport is not None:
            params = kwargs.get("params")
            full_url = self._base_url + url
            if params:
                full_url += "?" + urlencode(params, doseq=True)
            content = kwargs.get("content")
            if "json" in kwargs:
                content = json.dumps(kwargs["json"]).encode()
            req = Request(method, full_url, content=content)
            return await self._transport(req)

        assert self._client is not None
        async with self._lock:
            call = getattr(self._client, method.lower())
            resp = await asyncio.to_thread(call, url, **kwargs)
        return Response(resp)

    async def get(self, url: str, **kwargs: Any) -> Response:
        return await self._request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> Response:
        return await self._request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> Response:
        return await self._request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> Response:
        return await self._request("PATCH", url, **kwargs)

    async def options(self, url: str, **kwargs: Any) -> Response:
        return await self._request("OPTIONS", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> Response:
        return await self._request("DELETE", url, **kwargs)

    async def close(self) -> None:
        if self._client:
            await self.__aexit__(None, None, None)

    def __repr__(self) -> str:
        return f"<AsyncClient app={self._app!r} base_url={self._base_url}>"
