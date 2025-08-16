"""AsyncIO scheduler stub used in tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class AsyncIOScheduler:
    """Very small subset of :class:`apscheduler.AsyncIOScheduler`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._jobs: list[SimpleNamespace] = []

    def add_job(self, func: Any, *args: Any, **kwargs: Any) -> SimpleNamespace:
        job = SimpleNamespace(func=func, args=kwargs.get("args", []), kwargs=kwargs)
        self._jobs.append(job)
        return job

    def get_jobs(self) -> list[SimpleNamespace]:  # pragma: no cover - simple accessor
        return list(self._jobs)

    def start(self) -> None:  # pragma: no cover - no-op
        return None

    async def shutdown(self) -> None:  # pragma: no cover - simple async cleanup
        return None
