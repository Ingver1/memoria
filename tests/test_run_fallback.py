import asyncio

import pytest

from memory_system.cli import run


@pytest.mark.asyncio
async def test_run_fallback_inside_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run`` falls back when ``asyncio.run`` fails due to an active loop."""

    async def _coro() -> str:
        return "ok"

    class DummyLoop:
        def __init__(self) -> None:
            self.called = False

        def run_until_complete(self, c):  # type: ignore[no-untyped-def]
            self.called = True
            c.close()
            return "ok"

    dummy = DummyLoop()
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: dummy)

    assert run(_coro()) == "ok"
    assert dummy.called
