import pytest

from memory_system.settings import UnifiedSettings

try:  # pragma: no cover - optional dependency chain
    from memory_system.core.scheduler import start_background_tasks
except ModuleNotFoundError as exc:  # cryptography missing
    start_background_tasks = None  # type: ignore[assignment]
    _missing_reason = str(exc)


class DummyStore:
    meta_store = object()
    vector_store = object()


@pytest.mark.asyncio
async def test_background_scheduler_has_jobs():
    settings = UnifiedSettings()
    store = DummyStore()
    if start_background_tasks is None:  # cryptography missing
        pytest.skip(_missing_reason)
    try:
        scheduler = start_background_tasks(store, settings)
    except RuntimeError as exc:  # APScheduler missing
        pytest.skip(str(exc))
    try:
        jobs = scheduler.get_jobs()
        assert len(jobs) == 2
    finally:
        await scheduler.shutdown()
