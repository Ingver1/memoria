"""Simplified memory management routes used in tests."""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, List

from fastapi import APIRouter, HTTPException, Query, Request, status
from memory_system.api.schemas import MemoryCreate, MemoryQuery, MemoryRead, MemorySearchResult
from memory_system.core.store import Memory, SQLiteMemoryStore, get_memory_store
from memory_system.utils.security import EnhancedPIIFilter
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)
router = APIRouter(tags=["Memory Management"])


async def _store(request: Request) -> SQLiteMemoryStore:
    return get_memory_store(request)


@router.post("/", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreate,
    request: Request,
) -> JSONResponse:
    store = await _store(request)
    pii_filter = EnhancedPIIFilter()
    clean_text, _found, _types = pii_filter.redact(payload.text)
    mem = Memory(
        id=str(uuid.uuid4()),
        text=clean_text,
        metadata={"tags": payload.tags, "role": payload.role, "user_id": payload.user_id},
    )
    await store.add(mem)
    log.info("Created memory %s", mem.id)
    mem_read = MemoryRead.model_validate(asdict(mem))
    return JSONResponse(mem_read.model_dump(), status_code=201)


@router.get("/", response_model=list[MemoryRead])
async def list_memories(
    request: Request,
    user_id: str | None = Query(None),
) -> JSONResponse:
    store = await _store(request)
    records = await store.search(metadata_filters={"user_id": user_id} if user_id else None)
    payload = [MemoryRead.model_validate(asdict(r)).model_dump() for r in records]
    return JSONResponse(payload)


@router.post("/search", response_model=list[MemorySearchResult])
async def search_memories(
    query: MemoryQuery,
    request: Request,
) -> JSONResponse:
    if not query.query:
        raise HTTPException(status_code=422, detail="Query must not be empty")
    store = await _store(request)
    results = await store.search(text_query=query.query, limit=query.top_k)
    payload = [MemorySearchResult.model_validate(asdict(r)).model_dump() for r in results]
    return JSONResponse(payload)


@router.get("/best", response_model=list[MemoryRead])
async def best_memories(request: Request, limit: int = Query(5, ge=1, le=50)) -> JSONResponse:
    store = await _store(request)
    records = await store.list_recent(n=limit)
    payload = [MemoryRead.model_validate(asdict(r)).model_dump() for r in records]
    return JSONResponse(payload)
