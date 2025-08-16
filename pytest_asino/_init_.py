"""Minimal pytest-asyncio compatibility plugin used in tests.

This module provides a tiny subset of the real plugin sufficient for running
``async def`` tests in this repository.  It is intentionally lightweight and
has no external dependencies.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
from typing import Any

import pytest
from pytest import fixture as _fixture


@pytest.hookimpl
def pytest_runtest_setup(item: pytest.Item) -> None:  # pragma: no cover - test plugin
    """Ensure the event_loop fixture is requested for coroutine tests."""
    if inspect.iscoroutinefunction(getattr(item, "obj", None)) and "event_loop" not in getattr(
        item, "fixturenames", ()
    ):
        item.fixturenames.append("event_loop")


@pytest.hookimpl
def pytest_pyfunc_call(pyfuncitem: pytest.Item) -> bool | None:  # pragma: no cover
    """Run ``async def`` tests using the provided event loop."""
    testfunc = getattr(pyfuncitem, "obj", None)
    if inspect.iscoroutinefunction(testfunc):
        loop: asyncio.AbstractEventLoop = pyfuncitem.funcargs["event_loop"]
        argnames = getattr(pyfuncitem, "_fixtureinfo", None)
        names: list[str] = getattr(argnames, "argnames", []) if argnames else []
        kwargs: dict[str, Any] = {name: pyfuncitem.funcargs[name] for name in names}
        loop.run_until_complete(testfunc(**kwargs))
        return True
    return None


@_fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:  # pragma: no cover - test plugin
    """Provide a default event loop for tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)


def pytest_addoption(parser: pytest.Parser) -> None:  # pragma: no cover
    """Declare the asyncio_mode ini option for compatibility."""
    parser.addini("asyncio_mode", help="asyncio mode (stubbed)", default="auto")


__all__ = [
    "event_loop",
    "fixture",
    "pytest_addoption",
    "pytest_pyfunc_call",
    "pytest_runtest_setup",
]


def fixture(func=None, *fargs, **fkwargs):  # pragma: no cover - test plugin
    """Drop-in replacement for :func:`pytest.fixture` with async support."""

    def decorator(f):
        if inspect.isasyncgenfunction(f):

            @_fixture(*fargs, **fkwargs)
            @functools.wraps(f)
            def wrapper(*a, **kw):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                agen = f(*a, **kw)
                try:
                    result = loop.run_until_complete(agen.__anext__())
                    yield result
                finally:
                    with contextlib.suppress(StopAsyncIteration):
                        loop.run_until_complete(agen.__anext__())
                    loop.run_until_complete(agen.aclose())

            return wrapper

        if inspect.iscoroutinefunction(f):

            @_fixture(*fargs, **fkwargs)
            @functools.wraps(f)
            def wrapper(*a, **kw):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(f(*a, **kw))

            return wrapper

        return _fixture(*fargs, **fkwargs)(f)

    if func is not None and callable(func):
        return decorator(func)

    return decorator
