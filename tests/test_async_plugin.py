import asyncio
from typing import Coroutine

import pytest


@pytest.mark.asyncio
def test_sync_returns_coroutine() -> Coroutine[None, None, None]:
    async def inner() -> str:
        return "ok"

    async def run() -> None:
        result = await inner()
        assert result == "ok"

    return run()


async def test_async_function_collection() -> None:
    """Async ``def`` tests should be collected and executed."""
    await asyncio.sleep(0)
