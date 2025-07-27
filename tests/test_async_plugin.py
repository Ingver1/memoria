import pytest


@pytest.mark.asyncio
def test_sync_returns_coroutine():
    async def inner():
        return "ok"

    async def run():
        result = await inner()
        assert result == "ok"

    return run()
