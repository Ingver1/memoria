from __future__ import annotations

"""Simple agent planner leveraging case-based reasoning."""


from memory_system.core.store import get_store
from memory_system.unified_memory import MemoryStoreProtocol, search
from memory_system.utils.cases import adapt_case


async def generate_plans(
    problem: str,
    *,
    k: int = 3,
    store: MemoryStoreProtocol | None = None,
) -> tuple[list[str], list[str]]:
    """
    Return ``k`` plan proposals and relevant lessons for ``problem``.

    Existing cases are retrieved from the long-term store using
    :meth:`search_cases`.  Each retrieved case is adapted to the new problem
    via :func:`adapt_case`.  In addition, top-k relevant lessons are surfaced
    from long-term memory to guide future actions.
    """
    st: MemoryStoreProtocol = store or await get_store()
    cases = await st.search_cases(problem, k=k)
    plans = [adapt_case(c, problem).plan for c in cases]
    lesson_mems = await search(
        problem,
        k=k,
        metadata_filter={"memory_type": "lesson"},
        store=st,
        context={"store_id": id(st)},
    )
    lessons = [m.text for m in lesson_mems]
    return plans, lessons
