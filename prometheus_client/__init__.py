from __future__ import annotations

import types
from contextlib import AbstractContextManager
from typing import Any

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

class _Metric:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._value = 0.0

    def inc(self, amount: float | int = 1) -> None:
        self._value += float(amount)

    def set(self, value: float | int) -> None:
        self._value = float(value)

    def time(self) -> AbstractContextManager[None]:
        class _Timer(AbstractContextManager[None]):
            def __enter__(self_inner) -> None:
                return None

            def __exit__(
                self_inner,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: types.TracebackType | None,
            ) -> None:
                return None

        return _Timer()

class Counter(_Metric):
    def labels(self, *args: Any, **kwargs: Any) -> "Counter":
        return self


class Gauge(_Metric):
    def labels(self, *args: Any, **kwargs: Any) -> "Gauge":
        return self


class Histogram(_Metric):
    def labels(self, *args: Any, **kwargs: Any) -> "Histogram":
        return self

def generate_latest() -> bytes:
    return b"# HELP dummy Dummy metric\n# TYPE dummy counter\n"


def make_asgi_app() -> Any:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            return
        body = generate_latest()
        headers = [
            (b"content-type", CONTENT_TYPE_LATEST.encode()),
            (b"content-length", str(len(body)).encode()),
        ]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    return app
