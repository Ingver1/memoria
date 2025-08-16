"""Minimal httpx stubs for tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Self


class Response:
    def __init__(self, status_code: int = 200, json_data: Any | None = None) -> None:
        self.status_code = status_code
        self._json = json_data or {}

    def json(self) -> Any:  # pragma: no cover
        return self._json


class Request:
    """Very small request representation used by CLI tests."""

    def __init__(
        self,
        method: str = "",
        url: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.method = method
        self.url = SimpleNamespace(path=url, params={}, __str__=lambda self_: url)
        self.content = content
        self.headers = headers or {}


class MockTransport:
    def __init__(self, handler: Callable[[Request], Response]) -> None:
        self.handler = handler

    def __call__(self, request: Request) -> Response:  # pragma: no cover - simple dispatch
        return self.handler(request)


class ASGITransport:
    def __init__(self, app: Any) -> None:  # pragma: no cover - stub
        self.app = app

    def __call__(self, request: Request) -> Response:  # pragma: no cover - stub
        return Response()


class AsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._transport = kwargs.get("transport")
        self._base_url = kwargs.get("base_url", "")

    async def __aenter__(self) -> Self:  # pragma: no cover
        return self

    async def __aexit__(self, *exc: object) -> None:  # pragma: no cover
        return None

    async def get(self, *args: Any, **kwargs: Any) -> Response:  # pragma: no cover
        if self._transport:
            req = Request("GET", args[0] if args else "", headers=kwargs.get("headers"))
            return self._transport(req)
        return Response()

    async def post(self, *args: Any, **kwargs: Any) -> Response:  # pragma: no cover
        if self._transport:
            content = kwargs.get("content")
            if content is None and "json" in kwargs:
                content = json.dumps(kwargs["json"]).encode()
            content = content or b""
            req = Request(
                "POST",
                args[0] if args else "",
                content=content,
                headers=kwargs.get("headers"),
            )
            return self._transport(req)
        return Response()

    async def request(self, method: str, url: str, **kwargs: Any) -> Response:  # pragma: no cover
        if self._transport:
            content = kwargs.get("content")
            if content is None and "json" in kwargs:
                content = json.dumps(kwargs["json"]).encode()
            content = content or b""
            req = Request(method, url, content=content, headers=kwargs.get("headers"))
            return self._transport(req)
        return Response()


class Timeout:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
        pass


class Client:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
        pass

    def post(self, *args: Any, **kwargs: Any) -> Response:  # pragma: no cover
        return Response()
