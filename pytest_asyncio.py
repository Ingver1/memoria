import asyncio
import inspect

import pytest

# Export a fixture decorator compatible with the standard ``pytest-asyncio``
# plugin so tests can use ``@pytest_asyncio.fixture`` regardless of the
# environment.
fixture = pytest.fixture

__all__ = [
    "pytest_configure",
    "pytest_pyfunc_call",
    "pytest_fixture_setup",
    "pytest_unconfigure",
]


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark async test")


# Single shared event loop for the entire test session.  We also register it as
# the default loop so that synchronous tests can create ``asyncio`` primitives
# like :class:`asyncio.Future` without errors.
LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)

@pytest.hookimpl(tryfirst=True)


def pytest_pyfunc_call(pyfuncitem):
    testfunc = pyfuncitem.obj
    if inspect.iscoroutinefunction(testfunc):
        asyncio.set_event_loop(LOOP)
        fixtureinfo = getattr(pyfuncitem, "_fixtureinfo", None)
        if fixtureinfo is not None:
            argnames = getattr(fixtureinfo, "argnames", list(pyfuncitem.funcargs))
        else:
            argnames = list(pyfuncitem.funcargs)
        kwargs = {name: pyfuncitem.funcargs[name] for name in argnames if name in pyfuncitem.funcargs}
        LOOP.run_until_complete(testfunc(**kwargs))
        return True


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    func = fixturedef.func
    if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
        asyncio.set_event_loop(LOOP)
        argnames = getattr(fixturedef, "argnames", list(inspect.signature(func).parameters))
        params = {name: request.getfixturevalue(name) for name in argnames if name in request.fixturenames}
        if inspect.isasyncgenfunction(func):
            agen = func(**params)
            value = LOOP.run_until_complete(agen.__anext__())

            def finalizer() -> None:
                try:
                    LOOP.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    pass

            request.addfinalizer(finalizer)
            fixturedef.cached_result = (value, 0, None)
            return value
        result = LOOP.run_until_complete(func(**params))
        fixturedef.cached_result = (result, 0, None)
        return result


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Close the shared event loop when the test session ends."""
    try:
        LOOP.close()
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())
      
