"""Simplified memory management routes used in tests."""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import cast

from fastapi import APIRouter, HTTPException, Query, Request, status

from memory_system import unified_memory
from memory_system.api.schemas import MemoryCreate, MemoryQuery, MemoryRead
from memory_system.core.store import Memory, SQLiteMemoryStore, get_memory_store, get_store
from memory_system.utils.security import EnhancedPIIFilter

log = logging.getLogger(__name__)
router = APIRouter(tags=["Memory Management"])


async def _store(request: Request) -> SQLiteMemoryStore:
    """Return app-bound store or fall back to process singleton."""
    try:
        return get_memory_store(request)
    except AttributeError:
        return await get_store()


@router.post("/", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreate,
    request: Request,
) -> MemoryRead:
    store = await _store(request)
    pii_filter = EnhancedPIIFilter()
    clean_text, _found, _types = pii_filter.redact(payload.text)
    mem = Memory(
        id=str(uuid.uuid4()),
        text=clean_text,
        metadata={
            "tags": payload.tags,
            "role": payload.role,
            "user_id": payload.user_id,
            "modality": payload.modality,
            "language": payload.language,
        },
    )
    await store.add(mem)
    log.info("Created memory %s", mem.id)
    mem_read = MemoryRead.model_validate(asdict(mem))
    return cast(MemoryRead, mem_read)


@router.get("/", response_model=list[MemoryRead])
async def list_memories(
    request: Request,
    user_id: str | None = Query(None),
) -> list[MemoryRead]:
    store = await _store(request)
    records = await store.search(metadata_filters={"user_id": user_id} if user_id else None)
    payload = [MemoryRead.model_validate(asdict(r)) for r in records]
    return payload


@router.post("/search", response_model=list[MemoryRead])
async def search_memories(
    query: MemoryQuery,
    request: Request,
) -> list[MemoryRead]:
    if not query.query:
        raise HTTPException(status_code=422, detail="Query must not be empty")
    store = await _store(request)
    meta = dict(query.metadata_filter or {})
    meta.setdefault("modality", query.modality)
    if query.language is not None:
        meta.setdefault("language", query.language)
    results = await store.search(
        text_query=query.query,
        metadata_filters=meta,
        limit=query.top_k,
    )
    payload = [MemoryRead.model_validate(asdict(r)) for r in results]
    return payload


@router.get("/best", response_model=list[MemoryRead])
async def best_memories(
    request: Request,
    limit: int = Query(5, ge=1, le=50),
) -> list[MemoryRead]:
    """Return the most important memories ranked by score."""
    store = await _store(request)
    records = await unified_memory.list_best(n=limit, store=store)
    payload = [MemoryRead.model_validate(asdict(r)) for r in records]
    return payload
