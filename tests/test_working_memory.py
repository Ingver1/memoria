import datetime as dt
from types import SimpleNamespace

import pytest

from memory_system.unified_memory import Memory
from memory_system.working_memory import TaskCompletionConfig, WorkingMemory


class DummyStore:
    def __init__(self) -> None:
        self.memories: list[Memory] = []

    async def add_memory(self, memory: Memory) -> None:  # pragma: no cover - simple stub
        self.memories.append(memory)

    async def upsert_scores(self, scores):  # pragma: no cover - simple stub
        pass


@pytest.mark.asyncio
async def test_central_executive_push_pop() -> None:
    wm = WorkingMemory()
    wm.push_task("task1")
    wm.push_task("task2")
    assert wm.central_executive.current_task() == "task2"
    store = DummyStore()
    config = TaskCompletionConfig(
        lesson="remember to hydrate",
        valence=0.2,
        importance=0.3,
        success_score=0.9,
    )
    popped = await wm.pop_task(store=store, config=config)
    assert popped == "task2"
    assert wm.central_executive.current_task() == "task1"
    lesson_mem = next((m for m in store.memories if m.memory_type == "lesson"), None)
    assert lesson_mem is not None
    assert "remember to hydrate" in lesson_mem.text
    assert lesson_mem.valence == 0.2
    assert lesson_mem.importance == 0.3
    assert lesson_mem.success_score == 0.9


@pytest.mark.asyncio
async def test_serialize_state_persists_components() -> None:
    wm = WorkingMemory()
    wm.push_task("write tests")
    wm.write_scratchpad("temp note")
    wm.add_visual_reference("img://ref1")
    store = DummyStore()
    memories = await wm.serialize_state(store=store, persist=True)
    assert len(memories) == 3
    texts = [m.text for m in store.memories]
    assert any("Outstanding task" in t for t in texts)
    assert any("Scratchpad" in t for t in texts)
    assert any("Visual references" in t for t in texts)


@pytest.mark.asyncio
async def test_serialize_state_skip_persistence() -> None:
    wm = WorkingMemory()
    wm.push_task("skip me")
    store = DummyStore()
    memories = await wm.serialize_state(store=store)
    assert memories == []
    assert store.memories == []


@pytest.mark.asyncio
async def test_build_prompt_includes_context(monkeypatch) -> None:
    wm = WorkingMemory()
    wm.write_scratchpad("remember this")

    async def fake_search(**kwargs):
        return [Memory(memory_id="1", text="retrieved context", created_at=dt.datetime.now(dt.UTC))]

    monkeypatch.setattr("memory_system.working_memory.search", fake_search)
    prompt = await wm.build_prompt("What now?")
    assert "retrieved context" in prompt
    assert "remember this" in prompt
    assert prompt.strip().endswith("What now?")


def test_eviction_based_on_priority(monkeypatch) -> None:
    dummy_settings = SimpleNamespace(working_memory=SimpleNamespace(budget=2))
    monkeypatch.setattr("memory_system.working_memory.get_settings", lambda: dummy_settings)
    wm = WorkingMemory()
    now = dt.datetime.now(dt.UTC)
    wm.add_memory("a", win_rate=0.2, created_at=now - dt.timedelta(seconds=5))
    wm.add_memory("b", win_rate=0.9, created_at=now - dt.timedelta(seconds=10))
    wm.add_memory("c", win_rate=0.5, created_at=now - dt.timedelta(seconds=1))
    assert len(wm.items) == 2
    texts = {item.text for item in wm.items}
    assert texts == {"b", "c"}
