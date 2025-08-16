"""Minimal pytest-asyncio plugin for running coroutine tests."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect

import pytest
from pytest import fixture as _fixture


@pytest.hookimpl
def pytest_runtest_setup(item: pytest.Item) -> None:  # pragma: no cover - test plugin
    """Ensure the event_loop fixture is requested for coroutine tests."""
    if (
        inspect.iscoroutinefunction(getattr(item, "obj", None))
        and "event_loop" not in item.fixturenames
    ):
        item.fixturenames.append("event_loop")


@pytest.hookimpl
def pytest_pyfunc_call(pyfuncitem: pytest.Item) -> bool | None:  # pragma: no cover - test plugin
    """Run ``async def`` tests using the provided event loop."""
    testfunc = pyfuncitem.obj
    if inspect.iscoroutinefunction(testfunc):
        loop: asyncio.AbstractEventLoop = pyfuncitem.funcargs["event_loop"]
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        loop.run_until_complete(testfunc(**kwargs))
        return True
    return None


@_fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:  # pragma: no cover - test plugin
    """Provide a default event loop for tests without creating new loops.

    In restricted sandboxes, creating a new event loop may attempt to open
    socket pairs and fail. Reuse a pre-existing loop via the project helper.
    """
    from memory_system.utils.loop import get_loop
    import pytest as _pytest

    try:
        loop = get_loop()
    except RuntimeError:
        _pytest.skip("No usable event loop available in sandbox")
    # Do not close the shared loop in this sandboxed environment; let the
    # interpreter handle shutdown to avoid self-pipe errors from closing.
    yield loop


def pytest_addoption(parser: pytest.Parser) -> None:  # pragma: no cover - test plugin
    """Declare the asyncio_mode ini option for compatibility."""
    parser.addini("asyncio_mode", help="asyncio mode (stubbed)", default="auto")


__all__ = [
    "event_loop",
    "fixture",
    "pytest_addoption",
    "pytest_pyfunc_call",
    "pytest_runtest_setup",
]


# ---------------------------------------------------------------------------
# ``fixture`` implementation
# ---------------------------------------------------------------------------


def fixture(func=None, *fargs, **fkwargs):  # pragma: no cover - test plugin
    """Drop-in replacement for :func:`pytest.fixture` with async support."""

    def decorator(f):
        if inspect.isasyncgenfunction(f):

            @_fixture(*fargs, **fkwargs)
            @functools.wraps(f)
            def wrapper(*a, **kw):
                from memory_system.utils.loop import get_loop
                import pytest as _pytest

                try:
                    loop = get_loop()
                except RuntimeError:
                    _pytest.skip("No usable event loop available in sandbox")
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
                from memory_system.utils.loop import get_loop
                import pytest as _pytest

                try:
                    loop = get_loop()
                except RuntimeError:
                    _pytest.skip("No usable event loop available in sandbox")
                return loop.run_until_complete(f(*a, **kw))

            return wrapper

        return _fixture(*fargs, **fkwargs)(f)

    if func is not None and callable(func):
        return decorator(func)

    return decorator
