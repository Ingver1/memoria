"""Stub Request class."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class Request:
    """Very small subset of Starlette's :class:`Request`."""

    def __init__(self, scope: Any | None = None) -> None:
        scope = scope or {}
        self.scope = scope
        raw_headers = scope.get("headers", [])
        self.headers = {k.decode(): v.decode() for k, v in raw_headers}
        client = scope.get("client") or ("", 0)
        self.client = SimpleNamespace(host=client[0])
        path = scope.get("path", "")
        self.url = SimpleNamespace(path=path, params={})
