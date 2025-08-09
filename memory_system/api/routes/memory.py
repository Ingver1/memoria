"""Simplified memory management routes used in tests."""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import Any, Optional, cast

from fastapi import APIRouter, HTTPException, Query, Request, status

from memory_system import unified_memory
from memory_system.api.schemas import (
    MemoryCreate,
    MemoryQuery,
    MemoryRead,
    MemoryReinforce,
    MemoryUpdate,
)
from memory_system.core.store import Memory, SQLiteMemoryStore, get_memory_store, get_store
from memory_system.unified_memory import ListBestWeights
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


@router.patch("/{memory_id}", response_model=MemoryRead)
async def update_memory(memory_id: str, payload: MemoryUpdate, request: Request) -> MemoryRead:
    """Patch existing memory fields and metadata."""
    store = await _store(request)
    metadata: dict[str, Any] = {}
    if payload.role is not None:
        metadata["role"] = payload.role
    if payload.tags is not None:
        metadata["tags"] = payload.tags
    updated = await unified_memory.update(
        memory_id,
        text=payload.text,
        metadata=metadata or None,
        importance=payload.importance,
        importance_delta=payload.importance_delta,
        valence=payload.valence,
        valence_delta=payload.valence_delta,
        emotional_intensity=payload.arousal,
        emotional_intensity_delta=payload.arousal_delta,
        store=store,
    )
    mem_read = MemoryRead.model_validate(asdict(updated))
    return cast(MemoryRead, mem_read)


@router.post("/{memory_id}/reinforce", response_model=MemoryRead)
async def reinforce_memory(memory_id: str, payload: MemoryReinforce, request: Request) -> MemoryRead:
    """Reinforce a memory's scoring attributes."""
    store = await _store(request)
    updated = await unified_memory.reinforce(
        memory_id,
        amount=payload.importance_delta,
        valence_delta=payload.valence_delta,
        intensity_delta=payload.arousal_delta,
        store=store,
    )
    mem_read = MemoryRead.model_validate(asdict(updated))
    return cast(MemoryRead, mem_read)


@router.get("/best", response_model=list[MemoryRead])
async def best_memories(
    request: Request,
    n: int = Query(50, ge=1, le=500, alias="limit"),
    level: int | None = Query(None, ge=0),
    user_id: str | None = Query(None),
    importance: Optional[float] = Query(None, ge=0.0, description="Weight for importance when ranking"),
    arousal: Optional[float] = Query(
        None, ge=0.0, description="Weight for emotional intensity (arousal)"
    ),  # alias для emotional_intensity
    valence_pos: Optional[float] = Query(None, ge=0.0, description="Weight for positive valence"),
    valence_neg: Optional[float] = Query(None, ge=0.0, description="Weight for negative valence"),
):
    store = await _store(request)
    meta = {"user_id": user_id} if user_id else None

    weights = None
    if any(v is not None for v in (importance, arousal, valence_pos, valence_neg)):
        weights = ListBestWeights(
            importance=importance or 1.0,
            emotional_intensity=arousal or 1.0,
            valence_pos=valence_pos or 1.0,
            valence_neg=valence_neg or 0.5,
        )

    records = await unified_memory.list_best(
        n=n,
        store=store,
        level=level,
        metadata_filter=meta,
        weights=weights,
    )
    payload = [MemoryRead.model_validate(asdict(r)) for r in records]
    return payload
