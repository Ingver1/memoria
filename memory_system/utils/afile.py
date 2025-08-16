"""
Asynchronous file helper.

This module exposes :func:`open` which returns an object that can be used with
``async with`` to read from or write to a file without blocking the event
loop. The returned object supports ``read``, ``readline``, ``write`` and
asynchronous iteration via ``__aiter__``/``__anext__``. When the
`aiofiles <https://github.com/Tinche/aiofiles>`_ package is installed it is
used directly. Otherwise a small fallback implementation delegates file
operations to a thread using :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import builtins
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self, TextIO, cast

if TYPE_CHECKING:
    from types import TracebackType

log = logging.getLogger(__name__)

try:
    import aiofiles as _aio
except ImportError:

    class _AsyncFile:
        def __init__(self, path: str, mode: str = "r", encoding: str | None = None) -> None:
            self._path = path
            self._mode = mode
            self._enc = encoding
            self._fh: TextIO | None = None

        async def __aenter__(self) -> Self:
            open_func = cast(
                "Callable[[str, str, int, str | None], TextIO]",
                builtins.open,
            )
            # ``asyncio.to_thread`` stubs in some Python versions do not accept
            # keyword arguments, so pass ``encoding`` positionally after the
            # ``buffering`` argument which defaults to ``-1``.
            self._fh = await asyncio.to_thread(open_func, self._path, self._mode, -1, self._enc)
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            if self._fh is not None:
                await asyncio.to_thread(self._fh.close)
                self._fh = None

        def __aiter__(self) -> Self:
            return self

        async def __anext__(self) -> str:
            if self._fh is None:
                msg = "I/O operation on closed file."
                raise ValueError(msg)
            line = await asyncio.to_thread(self._fh.readline)
            if line:
                return line
            raise StopAsyncIteration

        async def read(self, size: int = -1) -> str:
            if self._fh is None:
                msg = "I/O operation on closed file."
                raise ValueError(msg)
            return await asyncio.to_thread(self._fh.read, size)

        async def readline(self) -> str:
            if self._fh is None:
                msg = "I/O operation on closed file."
                raise ValueError(msg)
            return await asyncio.to_thread(self._fh.readline)

        async def write(self, data: str) -> int:
            if self._fh is None:
                msg = "I/O operation on closed file."
                raise ValueError(msg)
            return await asyncio.to_thread(self._fh.write, data)

    def open(path: str, mode: str = "r", encoding: str | None = None) -> Any:
        """Open a file asynchronously using a thread fallback."""
        return _AsyncFile(path, mode, encoding)

except Exception:  # pragma: no cover - difficult to trigger
    log.exception("Unexpected error importing aiofiles")
    raise
else:

    def open(path: str, mode: str = "r", encoding: str | None = None) -> Any:
        """Proxy to aiofiles.open when available."""
        # Forward to real aiofiles if available.  ``cast`` avoids strict stub
        # mismatches between aiofiles versions while keeping runtime behaviour.
        return cast(Any, _aio.open)(path, mode=mode, encoding=encoding)
