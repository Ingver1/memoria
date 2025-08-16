"""Stub Response class."""

from __future__ import annotations

import json
from typing import Any


class Response:
    """Minimal ASGI response stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.status_code = kwargs.get("status_code", 200)
        self.body = kwargs.get("content")
        self.headers = kwargs.get("headers", {})

    async def __call__(
        self, scope: Any, receive: Any, send: Any
    ) -> None:  # pragma: no cover - simple ASGI
        return None


class JSONResponse(Response):
    def __init__(
        self, content: Any = None, status_code: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(
            content=json.dumps(content) if content is not None else None,
            status_code=status_code,
            headers=headers,
        )
        self._content = content

    def json(self) -> Any:  # pragma: no cover - convenience
        return self._content
