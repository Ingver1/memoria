"""Background scheduling utilities for maintenance tasks."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

from memory_system.core.hierarchical_summarizer import HierarchicalSummarizer
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.interfaces import VectorIndexMaintenance
from memory_system.core.maintenance import (
    check_and_heal_index_drift,
    forget_old_memories,
)
from memory_system.settings import UnifiedSettings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from memory_system.core.enhanced_store import EnhancedMemoryStore

log = logging.getLogger(__name__)


async def _run_hierarchy(store: EnhancedMemoryStore, settings: UnifiedSettings) -> None:
    """Build summary levels for stored memories."""
    try:
        index = cast("FaissHNSWIndex", getattr(store.vector_store, "index", store.vector_store))
        summarizer = HierarchicalSummarizer(
            store._store,
            index,
            threshold=settings.maintenance.summarize_threshold,
            strategy=settings.summary_strategy,
        )
        level = 0
        while True:
            created = await summarizer.build_level(level)
            if not created:
                break
            level += 1
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("hierarchical summarisation failed: %s", exc)


async def _run_forgetting(store: EnhancedMemoryStore, settings: UnifiedSettings) -> None:
    """Prune low-value memories based on decay scores."""
    try:
        index = cast("FaissHNSWIndex", getattr(store.vector_store, "index", store.vector_store))
        await forget_old_memories(
            store._store,
            index,
            min_total=settings.maintenance.forget_min_total,
            retain_fraction=settings.maintenance.forget_retain_fraction,
            ttl=settings.maintenance.forget_ttl_seconds,
            low_trust=settings.maintenance.forget_low_trust_threshold,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("forgetting task failed: %s", exc)


async def _run_revision(store: EnhancedMemoryStore, settings: UnifiedSettings) -> None:
    """Ensure vector index matches records in the database."""
    try:
        index = cast(
            "VectorIndexMaintenance", getattr(store.vector_store, "index", store.vector_store)
        )
        await check_and_heal_index_drift(
            store._store,
            index,
            drift_threshold=settings.maintenance.audit_drift_threshold,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("revision task failed: %s", exc)


def start_background_tasks(
    store: EnhancedMemoryStore, settings: UnifiedSettings
) -> AsyncIOScheduler:
    """Start APScheduler with maintenance jobs and return the scheduler."""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("apscheduler is required for background tasks") from exc

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _run_hierarchy,
        "interval",
        seconds=settings.maintenance.summarize_interval_seconds,
        args=[store, settings],
        max_instances=1,
    )
    scheduler.add_job(
        _run_forgetting,
        "interval",
        seconds=settings.maintenance.forget_interval_seconds,
        args=[store, settings],
        max_instances=1,
    )
    scheduler.add_job(
        _run_revision,
        "interval",
        seconds=settings.maintenance.audit_interval_seconds,
        args=[store, settings],
        max_instances=1,
    )
    scheduler.start()
    return scheduler


@asynccontextmanager
async def scheduler_lifespan(
    store: EnhancedMemoryStore, settings: UnifiedSettings
) -> AsyncIterator[AsyncIOScheduler]:
    """Async context manager that ensures scheduler shutdown."""
    scheduler = start_background_tasks(store, settings)
    try:
        yield scheduler
    finally:
        await scheduler.shutdown()
