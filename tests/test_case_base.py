import pytest

from memory_system.agent.planner import generate_plans
from memory_system.core.store import Case, SQLiteMemoryStore
from memory_system.unified_memory import add_lesson
from memory_system.utils.cases import adapt_case


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "cases.db"
    return SQLiteMemoryStore(f"file:{path}?mode=rwc")


@pytest.mark.asyncio
async def test_search_cases(store: SQLiteMemoryStore) -> None:
    case1 = Case.new(
        problem="sort numbers", plan="sort numbers with quicksort", outcome="ok", evaluation=1.0
    )
    case2 = Case.new(problem="find max", plan="iterate to find max", outcome="ok", evaluation=0.5)
    await store.add_case(case1)
    await store.add_case(case2)

    res = await store.search_cases("sort")
    assert res
    assert res[0].problem == "sort numbers"


@pytest.mark.asyncio
async def test_adapt_case() -> None:
    c = Case.new(problem="add numbers", plan="first add numbers together")
    adapted = adapt_case(c, "sum integers")
    assert adapted.problem == "sum integers"
    assert "sum integers" in adapted.plan


@pytest.mark.asyncio
async def test_planner_generates_plans(store: SQLiteMemoryStore) -> None:
    c = Case.new(problem="boil water", plan="boil water using a kettle")
    await store.add_case(c)
    plans, lessons = await generate_plans("boil water", store=store)
    assert any("boil water" in p for p in plans)
    assert isinstance(lessons, list)


@pytest.mark.asyncio
async def test_planner_surfaces_lessons(store: SQLiteMemoryStore) -> None:
    await add_lesson("boil water carefully", store=store)
    _, lessons = await generate_plans("boil water", store=store)
    assert any("boil water carefully" in l for l in lessons)
