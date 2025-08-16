"""CORS middleware stub."""

from __future__ import annotations

from typing import Any


class CORSMiddleware:
    def __init__(self, app: Any, *args: Any, **kwargs: Any) -> None:
        self.app = app
