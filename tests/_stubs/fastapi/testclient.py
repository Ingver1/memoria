"""Test client stub for FastAPI."""

from __future__ import annotations

from typing import Any, Self

__all__ = ["TestClient"]


class TestClient:
    def __init__(self, app: Any, *args: Any, **kwargs: Any) -> None:
        self.app = app

    def __enter__(self) -> Self:  # pragma: no cover - context helper
        return self

    def __exit__(self, *exc: object) -> None:  # pragma: no cover - context helper
        return None

    def get(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return None

    def post(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return None

    def delete(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return None


TestClient.__test__ = False

ClientHelper = TestClient
