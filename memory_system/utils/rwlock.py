"""Simple reader/writer locks used across the project."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from threading import Condition


class AsyncRWLock:
    """Asynchronous reader/writer lock."""

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer = False

    @asynccontextmanager
    async def reader_lock(self) -> AsyncGenerator[None]:
        async with self._cond:
            await self._cond.wait_for(lambda: not self._writer)
            self._readers += 1
        try:
            yield
        finally:
            async with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @asynccontextmanager
    async def writer_lock(self) -> AsyncGenerator[None]:
        async with self._cond:
            await self._cond.wait_for(lambda: not self._writer and self._readers == 0)
            self._writer = True
        try:
            yield
        finally:
            async with self._cond:
                self._writer = False
                self._cond.notify_all()


class RWLock:
    """Thread based reader/writer lock."""

    def __init__(self) -> None:
        self._cond = Condition()
        self._readers = 0
        self._writer = False

    @contextmanager
    def reader_lock(self) -> Generator[None]:
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def writer_lock(self) -> Generator[None]:
        with self._cond:
            while self._writer or self._readers:
                self._cond.wait()
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


__all__ = ["AsyncRWLock", "RWLock"]
