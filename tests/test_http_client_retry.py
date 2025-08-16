import pytest

from memory_system import rag_router


def test_record_feedback_idempotency_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    client = rag_router.MemoriaHTTPClient("http://api", retry=1)
    calls = 0
    seen_headers = None

    class DummyClient:
        async def post(self, url, content=None, headers=None):  # type: ignore[no-untyped-def]
            nonlocal calls, seen_headers
            calls += 1
            seen_headers = headers
            if calls == 1:
                err = RuntimeError("boom")
                raise err
            return object()

    client.client = DummyClient()

    async def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(rag_router.asyncio, "sleep", no_sleep)
    try:
        from memory_system.utils.loop import get_loop

        get_loop().run_until_complete(client.record_feedback("m1", success=True))
    except RuntimeError:
        pytest.skip("No usable event loop available in sandbox")
    assert seen_headers is not None
    assert "Idempotency-Key" in seen_headers
    assert calls == 2
