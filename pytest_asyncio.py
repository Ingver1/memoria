import asyncio
import inspect
from typing import Any, cast

import pytest

# Export a fixture decorator compatible with the standard ``pytest-asyncio``
# plugin so tests can use ``@pytest_asyncio.fixture`` regardless of the
# environment.
fixture = pytest.fixture

__all__ = [
    "pytest_configure",
    "pytest_pycollect_makeitem",
    "pytest_pyfunc_call",
    "pytest_fixture_setup",
    "pytest_unconfigure",
]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "asyncio: mark async test")


# Single shared event loop for the entire test session.  We also register it as
# the default loop so that synchronous tests can create ``asyncio`` primitives
# like :class:`asyncio.Future` without errors.
LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makeitem(
    collector: "pytest.Collector", name: str, obj: object
) -> "pytest.Function | list[pytest.Function] | None":
    """Collect async test functions with or without the asyncio mark."""
    if isinstance(obj, (staticmethod, classmethod)):
        obj = cast(object, obj.__func__)

    if not callable(obj) or not collector.funcnamefilter(name):
        return None

    has_mark = any(getattr(mark, "name", "") == "asyncio" for mark in getattr(obj, "pytestmark", []))
    if inspect.iscoroutinefunction(obj) or has_mark:
        if isinstance(collector, pytest.Class):
            # Ensure class methods are collected with a bound instance
            return list(collector._genfunctions(name, obj))
        return pytest.Function.from_parent(collector, name=name, callobj=obj)
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: "pytest.Function") -> bool | None:
    """Run async tests and wrappers in the session event loop."""
    testfunc = pyfuncitem.obj

    # Only handle coroutine functions or tests marked with ``asyncio`` which may
    # return an awaitable (e.g. Hypothesis wrappers).
    if not inspect.iscoroutinefunction(testfunc) and pyfuncitem.get_closest_marker("asyncio") is None:
        return None

    asyncio.set_event_loop(LOOP)
    fixtureinfo = getattr(pyfuncitem, "_fixtureinfo", None)
    if fixtureinfo is not None:
        argnames = getattr(fixtureinfo, "argnames", list(pyfuncitem.funcargs))
    else:
        argnames = list(pyfuncitem.funcargs)
    kwargs = {name: pyfuncitem.funcargs[name] for name in argnames if name in pyfuncitem.funcargs}

    instance = getattr(pyfuncitem, "instance", None)
    if instance is not None:
        testfunc = testfunc.__get__(instance, type(instance))

    result = testfunc(**kwargs)
    if inspect.isawaitable(result):
        LOOP.run_until_complete(result)
    return True


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef: "pytest.FixtureDef[Any]", request: "pytest.FixtureRequest") -> object:
    func = fixturedef.func
    if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
        asyncio.set_event_loop(LOOP)
        argnames = getattr(fixturedef, "argnames", list(inspect.signature(func).parameters))
        params = {name: request.getfixturevalue(name) for name in argnames if name in request.fixturenames}
        if inspect.isasyncgenfunction(func):
            agen = func(**params)
            # __anext__ returns Awaitable, so it's safe
            value = LOOP.run_until_complete(agen.__anext__())

            def finalizer() -> None:
                try:
                    LOOP.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    pass

            request.addfinalizer(finalizer)
            fixturedef.cached_result = (value, 0, None)
            return value
        # Ensure func(**params) is Awaitable
        awaitable = func(**params)
        if not inspect.isawaitable(awaitable):
            raise TypeError("Fixture function did not return an awaitable object")
        result = LOOP.run_until_complete(awaitable)
        fixturedef.cached_result = (result, 0, None)
        return result
    return None  # Ensure all code paths return


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: "pytest.Config") -> None:
    """Close the shared event loop when the test session ends."""
    try:
        LOOP.close()
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())
