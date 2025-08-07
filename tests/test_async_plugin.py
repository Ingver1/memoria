import asyncio


def test_sync_returns_coroutine() -> None:
    """A synchronous test can execute async code via ``asyncio.run``."""

    async def inner() -> str:
        return "ok"

    async def run() -> None:
        result = await inner()
        assert result == "ok"

    asyncio.run(run())


def test_async_function_collection() -> None:
    """Ensure asynchronous functions can still be executed."""
    asyncio.run(asyncio.sleep(0))
