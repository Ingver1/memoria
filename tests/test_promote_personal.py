import pytest
import pytest_asyncio

from memory_system.unified_memory import Memory, promote_personal


class DummyStore:
    def __init__(self) -> None:
        self.memories: list[Memory] = []

    async def search_memory(self, *, query, k=5, **kwargs):
        return [
            m
            for m in self.memories
            if query in m.text or m.metadata.get("card", {}).get("title") == query
        ]

    async def add_memory(self, memory: Memory) -> None:
        self.memories.append(memory)

    async def get(self, memory_id: str) -> Memory:
        for m in self.memories:
            if m.memory_id == memory_id:
                return m
        raise KeyError(memory_id)

    async def upsert_scores(self, scores):
        return None


@pytest_asyncio.fixture
async def store():
    return DummyStore()


@pytest.mark.asyncio
async def test_promote_personal_saves_card(store):
    mem = await promote_personal(
        "Title",
        "A unique claim",
        ["https://example.com"],
        "CC0",
        store=store,
    )
    loaded = await store.get(mem.memory_id)
    assert loaded.metadata["card"]["title"] == "Title"
    assert loaded.metadata["card"]["claim"] == "A unique claim"
    assert loaded.metadata["card"]["license"] == "CC0"


@pytest.mark.asyncio
async def test_promote_personal_rejects_pii(store):
    with pytest.raises(ValueError):
        await promote_personal(
            "T",
            "Contact me at test@example.com",
            ["https://example.com"],
            "CC0",
            store=store,
        )


@pytest.mark.asyncio
async def test_promote_personal_enforces_novelty(store):
    await promote_personal(
        "T1",
        "Novel claim",
        ["https://source"],
        "CC0",
        store=store,
    )
    with pytest.raises(ValueError):
        await promote_personal(
            "T2",
            "Novel claim",
            ["https://source2"],
            "CC0",
            store=store,
        )


@pytest.mark.asyncio
async def test_promote_personal_checks_license(store):
    with pytest.raises(ValueError):
        await promote_personal(
            "T",
            "Claim",
            ["https://source"],
            "proprietary",
            store=store,
        )


@pytest.mark.asyncio
async def test_promote_personal_requires_http_sources(store):
    with pytest.raises(ValueError):
        await promote_personal(
            "T",
            "Claim",
            ["not-a-url"],
            "CC0",
            store=store,
        )


@pytest.mark.asyncio
async def test_promote_personal_quarantines_conflicting_sources(store, monkeypatch):
    from memory_system import unified_memory as um

    monkeypatch.setattr(um, "CONTRADICTION_THRESHOLD", 1)
    await promote_personal(
        "T",
        "Claim1",
        ["https://src"],
        "CC0",
        store=store,
    )
    mem = await promote_personal(
        "T",
        "Claim2",
        ["https://src"],
        "CC0",
        store=store,
    )
    assert mem.metadata.get("quarantined") is True
    assert um.SOURCE_CONTRADICTIONS["https://src"] >= 1


@pytest.mark.asyncio
async def test_promote_personal_stale_source_degrades_trust(store, monkeypatch):
    from types import SimpleNamespace

    import httpx as httpx_stub

    from memory_system import unified_memory as um

    transport = httpx_stub.MockTransport(lambda request: httpx_stub.Response(404))
    dummy_httpx = SimpleNamespace(
        AsyncClient=lambda *a, **k: httpx_stub.AsyncClient(transport=transport)
    )
    monkeypatch.setattr(um, "require_httpx", lambda: dummy_httpx)

    mem = await promote_personal(
        "T2",
        "Claim",
        ["https://bad"],
        "CC0",
        store=store,
    )
    assert mem.ttl_seconds == um.STALE_TTL_SECONDS
    assert mem.metadata["card"]["trust_score"] < 1.0
    assert mem.metadata["card"]["validation"]["https://bad"]["ok"] is False
