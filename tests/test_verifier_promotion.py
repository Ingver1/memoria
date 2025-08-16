import datetime as dt

import pytest

from memory_system.unified_memory import Memory, update_trust_scores


class DummyStore:
    def __init__(self, mems):
        self.memories = {m.memory_id: m for m in mems}

    async def update_memory(self, memory_id, *, metadata=None, **kwargs):
        mem = self.memories[memory_id]
        if metadata:
            if mem.metadata is None:
                mem.metadata = {}
            mem.metadata.update(metadata)
        return mem

    async def upsert_scores(self, scores):
        pass


@pytest.mark.asyncio
async def test_code_verifier_controls_promotion(tmp_path) -> None:
    pass_dir = tmp_path / "pass"
    fail_dir = tmp_path / "fail"
    pass_dir.mkdir()
    fail_dir.mkdir()
    (pass_dir / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    (fail_dir / "test_bad.py").write_text("def test_bad():\n    assert False\n")

    now = dt.datetime.now(dt.UTC)
    mem_ok = Memory(
        memory_id="1",
        text="print('ok')",
        created_at=now,
        metadata={
            "trust_score": 0.0,
            "shadow": True,
            "verifier": {"type": "code", "test_suite_path": str(pass_dir), "build_hash": "x"},
        },
    )
    mem_bad = Memory(
        memory_id="2",
        text="print('bad')",
        created_at=now,
        metadata={
            "trust_score": 0.0,
            "shadow": True,
            "verifier": {"type": "code", "test_suite_path": str(fail_dir), "build_hash": "x"},
        },
    )

    st1 = DummyStore([mem_ok])
    st2 = DummyStore([mem_bad])

    async def reason(task, mems):
        return 4.0 if mems else 0.0

    await update_trust_scores("task", [mem_ok], reasoner=reason, store=st1)
    await update_trust_scores("task", [mem_bad], reasoner=reason, store=st2)

    assert st1.memories["1"].metadata.get("shadow") is False
    assert st2.memories["2"].metadata.get("shadow") is True
