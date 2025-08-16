import pytest

from memory_system.skills import (
    memory_to_skill,
    retrieve_skills,
    select_skill_ucb1,
    store_skill,
    update_skill_success,
)


@pytest.mark.asyncio
async def test_store_retrieve_and_update_skill(store):
    steps = [{"action": "compute", "value": 1}]
    params = {"n": 1}
    result = {"out": 2}
    mem = await store_skill(steps, params, result, store=store)

    candidates = await retrieve_skills("compute", k=1, retriever="sparse", store=store)
    assert candidates and candidates[0].memory_id == mem.memory_id
    skill = memory_to_skill(candidates[0])
    assert skill.steps == steps

    chosen = select_skill_ucb1(candidates)
    assert chosen is not None and chosen.memory_id == mem.memory_id

    updated = await update_skill_success(chosen, reward=0.0, store=store)
    assert updated.success_score == 0.0
