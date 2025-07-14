import asyncio
from types import TracebackType
from typing import Any, Optional, cast

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


class Response:
    def __init__(self, resp: Any) -> None:
        self._resp = resp

    @property
    def status_code(self) -> int:
        return cast(int, self._resp.status_code)

    def json(self) -> Any:
        return self._resp.json()

    @property
    def text(self) -> str:
        return cast(str, self._resp.text)

    @property
    def headers(self) -> Any:
        return self._resp.headers

    def raise_for_status(self) -> None:
        if hasattr(self._resp, "raise_for_status"):
            self._resp.raise_for_status()
            
class AsyncClient:
    def __init__(self, app: Any | None = None, base_url: str = "http://test", timeout: float | None = None) -> None:
        self._app = app
        self._base_url = base_url
        self._client: Optional[ClientHelper] = None
        self._timeout = timeout
        self._lock = _AsyncLock()

    async def __aenter__(self) -> "AsyncClient":
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

    async def get(self, url: str, **kwargs: Any) -> Response:
        assert self._client is not None
        async with self._lock:
            resp = await asyncio.to_thread(self._client.get, url, **kwargs)
        return Response(resp)

    async def post(self, url: str, **kwargs: Any) -> Response:
        assert self._client is not None
        async with self._lock:
            resp = await asyncio.to_thread(self._client.post, url, **kwargs)
        return Response(resp)

    async def delete(self, url: str, **kwargs: Any) -> Response:
        assert self._client is not None
        async with self._lock:
            resp = await asyncio.to_thread(self._client.delete, url, **kwargs)
        return Response(resp)
