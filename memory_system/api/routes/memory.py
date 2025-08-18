"""Simplified memory management routes used in tests."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import math
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal, cast

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic import BaseModel, Field
else:  # pragma: no cover - runtime import
    try:
        import pydantic as _pyd_module
    except ImportError:  # pragma: no cover
        from memory_system.utils import pydantic_compat as _pyd_module
    BaseModel = _pyd_module.BaseModel
    Field = _pyd_module.Field

from memory_system.api.dependencies import get_vector_store
from memory_system.api.schemas import (
    MemoryAging,
    MemoryCreate,
    MemoryQuery,
    MemoryRead,
    MemoryReinforce,
    MemoryUpdate,
    SearchParams
)
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.maintenance import consolidate_store, forget_old_memories
from memory_system.core.memory_dynamics import MemoryDynamics
from memory_system.core.store import Memory, SQLiteMemoryStore, get_memory_store, get_store
from memory_system.memory_helpers import last_accessed
from memory_system.settings import get_settings
from memory_system.unified_memory import (
    ListBestWeights,
    MemoryStoreProtocol,
    list_best,
    reinforce,
    update,
)
from memory_system.utils.security import EnhancedPIIFilter

log = logging.getLogger(__name__)
router = APIRouter(tags=["Memory Management"])


COMMON_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"description": "Unauthorized"},
    429: {"description": "Too Many Requests"},
}

MEMORY_EXAMPLE = {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "text": "hello world",
    "role": "user",
    "tags": ["greeting"],
    "valence": 0.1,
    "emotional_intensity": 0.2,
    "importance": 0.3,
    "modality": "text",
    "language": "en",
    "memory_type": "episodic",
    "pinned": False,
    "ttl_seconds": 3600,
    "last_used": "2024-01-01T00:00:00Z",
    "success_score": 0.0,
    "decay": 0.0,
    "user_id": "user-123",
    "schema_type": "experience",
    "verifiability": 0.9,
    "source_hash": "abc123",
    "provenance": ["unit-test"],
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
}

MEMORY_CREATE_EXAMPLE = {
    "text": "hello world",
    "role": "user",
    "tags": ["greeting"],
    "user_id": "user-123",
    "modality": "text",
    "language": "en",
    "memory_type": "episodic",
    "pinned": False,
    "schema_type": "experience",
    "verifiability": 0.9,
    "source_hash": "abc123",
    "provenance": ["unit-test"],
}

MEMORY_QUERY_EXAMPLE = {
    "query": "hello",
    "top_k": 5,
    "metadata_filter": {"user_id": "user-123"},
    "modality": "text",
    "channel": "global",
}

MEMORY_UPDATE_EXAMPLE = {
    "text": "updated text",
    "importance_delta": 0.1,
    "verifiability": 0.8,
    "schema_type": "experience",
    "source_hash": "def456",
    "provenance": ["unit-test"],
}

MEMORY_REINFORCE_EXAMPLE = {
    "importance_delta": 0.2,
    "valence_delta": 0.1,
}

MEMORY_AGING_EXAMPLE = {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "age_days": 1.0,
    "retention": 0.9,
    "access_count": 3,
}

FORGET_EXAMPLE = {"policy": "low_trust", "threshold": 0.5}


CREATE_MEMORY_RESPONSES: dict[int | str, dict[str, Any]] = COMMON_RESPONSES | {
    201: {"content": {"application/json": {"example": MEMORY_EXAMPLE}}},
}

MEMORY_LIST_RESPONSES: dict[int | str, dict[str, Any]] = COMMON_RESPONSES | {
    200: {"content": {"application/json": {"example": [MEMORY_EXAMPLE]}}},
}

MEMORY_ITEM_RESPONSES: dict[int | str, dict[str, Any]] = COMMON_RESPONSES | {
    200: {"content": {"application/json": {"example": MEMORY_EXAMPLE}}},
}


class ForgetPayload(BaseModel):
    """Request body for forget endpoint."""

    policy: str | None = Field(default=None, description="Forgetting policy identifier")
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Threshold for the policy"
    )


async def _store(request: Request) -> SQLiteMemoryStore:
    """Return app-bound store or fall back to process singleton."""
    try:
        return cast("SQLiteMemoryStore", get_memory_store(request))
    except AttributeError:
        return await get_store()


@router.post(
    "/",
    response_model=MemoryRead,
    status_code=status.HTTP_201_CREATED,
    response_model_exclude_none=True,
    responses=CREATE_MEMORY_RESPONSES,
)
async def create_memory(
    request: Request,
    payload: MemoryCreate = Body(..., examples=[MEMORY_CREATE_EXAMPLE]),
) -> MemoryRead:
    store = await _store(request)
    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        cached = store.get_idempotent(idem_key)
        if cached is not None:
            return MemoryRead.model_validate(cached)
    settings = getattr(store, "settings", None)
    pii_filter = EnhancedPIIFilter()
    clean_text, found, _types = pii_filter.redact(payload.text)
    if found and getattr(getattr(settings, "security", None), "pii_log_details", False):
        stats = {k: v for k, v in pii_filter.get_stats().items() if v}
        total = sum(stats.values())
        log.info(
            "Detected %d PII items: %s",
            total,
            ", ".join(f"{k}={v}" for k, v in stats.items()),
        )
    meta: dict[str, Any] = {
        "tags": payload.tags,
        "role": payload.role,
        "user_id": payload.user_id,
        "modality": payload.modality,
        "language": payload.language,
        "lang": payload.lang,
        "lang_confidence": payload.lang_confidence,
    }
    if payload.metadata:
        meta.update(payload.metadata)
    if payload.schema_type is not None:
        meta["schema_type"] = payload.schema_type
    if payload.verifiability is not None:
        meta["verifiability"] = payload.verifiability
    if payload.source_hash is not None:
        meta["source_hash"] = payload.source_hash
    if payload.provenance is not None:
        meta["provenance"] = payload.provenance
    if payload.card is not None:
        card_dict = payload.card.model_dump(exclude_none=True)
        meta["card"] = card_dict
        if "success_count" in card_dict:
            meta.setdefault("success_count", card_dict["success_count"])
        if "trial_count" in card_dict:
            meta.setdefault("trial_count", card_dict["trial_count"])
    mem = Memory(
        id=str(uuid.uuid4()),
        text=clean_text,
        memory_type=payload.memory_type,
        pinned=payload.pinned,
        ttl_seconds=payload.ttl_seconds,
        last_used=payload.last_used or dt.datetime.now(dt.UTC),
        success_score=payload.success_score,
        decay=payload.decay,
        metadata=meta,
    )
    await store.add_memory(mem)
    log.info("Created memory %s", mem.id)
    mem_read = MemoryRead.model_validate(asdict(mem))
    if idem_key:
        store.set_idempotent(idem_key, mem_read.model_dump())
    return mem_read


@router.post(
    "/add",
    response_model=MemoryRead,
    status_code=status.HTTP_201_CREATED,
    response_model_exclude_none=True,
    responses=CREATE_MEMORY_RESPONSES,
)
async def add_memory(
    request: Request,
    payload: MemoryCreate = Body(..., examples=[MEMORY_CREATE_EXAMPLE]),
) -> MemoryRead:
    return await create_memory(request, payload)


@router.get(
    "/",
    response_model=list[MemoryRead],
    response_model_exclude_none=True,
    responses={
        **COMMON_RESPONSES,
        "200": {"content": {"application/json": {"example": [MEMORY_EXAMPLE]}}},
    },
)
async def list_memories(
    request: Request,
    user_id: str | None = Query(None, examples=["user-123"]),
) -> list[MemoryRead]:
    store = await _store(request)
    records = await store.search(metadata_filters={"user_id": user_id} if user_id else None)
    payload = [MemoryRead.model_validate(asdict(r)) for r in records]
    return payload


@router.post(
    "/search",
    response_model=list[MemoryRead],
    response_model_exclude_none=True,
    responses=MEMORY_LIST_RESPONSES,
)
async def search_memories(
    request: Request,
    query: MemoryQuery = Body(..., examples=[MEMORY_QUERY_EXAMPLE]),
    effort: Literal["low", "med", "high"] = Query("med"),
    limits: SearchParams = Depends(),
) -> list[MemoryRead]:
    if not query.query:
        raise HTTPException(status_code=422, detail="Query must not be empty")
    if query.channel != "global":
        # Validation is enforced at the schema level, but we double-check here
        # to ensure unsupported channels are rejected with a clear message.
        raise HTTPException(status_code=422, detail="Only 'global' channel is supported")
    settings = get_settings()
    budgets = getattr(settings.effort, effort)
    if (
        limits.max_cross_rerank_n is not None
        and limits.max_cross_rerank_n > budgets.max_cross_rerank_n
    ):
        log.warning(
            "cross rerank limit exceeded: requested %d allowed %d",
            limits.max_cross_rerank_n,
            budgets.max_cross_rerank_n,
        )
        raise HTTPException(status_code=429, detail="Cross rerank limit exceeded")
    max_k = limits.max_k or budgets.max_k
    max_context_tokens = limits.max_context_tokens or budgets.max_context_tokens
    timeout = limits.timeout or budgets.timeout_seconds
    if query.top_k > max_k:
        log.warning("k limit exceeded: requested %d allowed %d", query.top_k, max_k)
        raise HTTPException(status_code=429, detail="Requested k exceeds limit")
    store = await _store(request)
    meta = dict(query.metadata_filter or {})
    meta.setdefault("modality", query.modality)
    meta["lang"] = query.lang
    meta["lang_confidence"] = query.lang_confidence
    if query.language is not None:
        meta.setdefault("language", query.language)
    try:
        results = await asyncio.wait_for(
            store.search(
                text_query=query.query,
                metadata_filters=meta,
                limit=query.top_k,
            ),
            timeout=timeout,
        )
    except TimeoutError as err:
        log.warning("search timeout after %.2f s", timeout)
        raise HTTPException(status_code=429, detail="Search timeout") from err
    payload = [MemoryRead.model_validate(asdict(r)) for r in results]
    trimmed: list[MemoryRead] = []
    total = 0
    for item in payload:
        tok = len(item.text.split())
        if total + tok > max_context_tokens:
            log.warning("context token limit exceeded: %d > %d", total + tok, max_context_tokens)
            break
        total += tok
        trimmed.append(item)
    return trimmed


@router.patch(
    "/{memory_id}",
    response_model=MemoryRead,
    response_model_exclude_none=True,
    responses={
        **COMMON_RESPONSES,
        "200": {"content": {"application/json": {"example": MEMORY_EXAMPLE}}},
    },
)
async def update_memory(
    memory_id: str,
    request: Request,
    payload: MemoryUpdate = Body(..., examples=[MEMORY_UPDATE_EXAMPLE]),
) -> MemoryRead:
    """Patch existing memory fields and metadata."""
    store = await _store(request)
    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        cached = store.get_idempotent(idem_key)
        if cached is not None:
            return MemoryRead.model_validate(cached)
    metadata: dict[str, Any] = {}
    if payload.role is not None:
        metadata["role"] = payload.role
    if payload.tags is not None:
        metadata["tags"] = payload.tags
    if payload.schema_type is not None:
        metadata["schema_type"] = payload.schema_type
    if payload.verifiability is not None:
        metadata["verifiability"] = payload.verifiability
    if payload.source_hash is not None:
        metadata["source_hash"] = payload.source_hash
    if payload.provenance is not None:
        metadata["provenance"] = payload.provenance
    updated = await update(
        memory_id,
        text=payload.text,
        metadata=metadata or None,
        importance=payload.importance,
        importance_delta=payload.importance_delta,
        valence=payload.valence,
        valence_delta=payload.valence_delta,
        emotional_intensity=payload.emotional_intensity,
        emotional_intensity_delta=payload.emotional_intensity_delta,
        memory_type=payload.memory_type,
        pinned=payload.pinned,
        ttl_seconds=payload.ttl_seconds,
        last_used=payload.last_used,
        success_score=payload.success_score,
        decay=payload.decay,
        store=cast("MemoryStoreProtocol", store),
    )
    mem_read = MemoryRead.model_validate(asdict(updated))
    if idem_key:
        store.set_idempotent(idem_key, mem_read.model_dump())
    return mem_read


@router.post(
    "/{memory_id}/pin",
    response_model=MemoryRead,
    response_model_exclude_none=True,
    responses=MEMORY_ITEM_RESPONSES,
)
async def pin_memory(memory_id: str, request: Request) -> MemoryRead:
    store = await _store(request)
    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        cached = store.get_idempotent(idem_key)
        if cached is not None:
            return MemoryRead.model_validate(cached)
    updated = await update(memory_id, pinned=True, store=cast("MemoryStoreProtocol", store))
    mem_read = MemoryRead.model_validate(asdict(updated))
    if idem_key:
        store.set_idempotent(idem_key, mem_read.model_dump())
    return mem_read


@router.delete(
    "/{memory_id}/pin",
    response_model=MemoryRead,
    response_model_exclude_none=True,
    responses={
        **COMMON_RESPONSES,
        "200": {"content": {"application/json": {"example": MEMORY_EXAMPLE}}},
    },
)
async def unpin_memory(memory_id: str, request: Request) -> MemoryRead:
    store = await _store(request)
    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        cached = store.get_idempotent(idem_key)
        if cached is not None:
            return MemoryRead.model_validate(cached)
    updated = await update(memory_id, pinned=False, store=cast("MemoryStoreProtocol", store))
    mem_read = MemoryRead.model_validate(asdict(updated))
    if idem_key:
        store.set_idempotent(idem_key, mem_read.model_dump())
    return mem_read


@router.post(
    "/{memory_id}/reinforce",
    response_model=MemoryRead,
    response_model_exclude_none=True,
    responses=MEMORY_ITEM_RESPONSES,
)
async def reinforce_memory(
    memory_id: str,
    request: Request,
    payload: MemoryReinforce = Body(
        ...,
        examples=[MEMORY_REINFORCE_EXAMPLE],
    ),
) -> MemoryRead:
    """Reinforce a memory's scoring attributes."""
    store = await _store(request)
    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        cached = store.get_idempotent(idem_key)
        if cached is not None:
            return MemoryRead.model_validate(cached)
    updated = await reinforce(
        memory_id,
        amount=payload.importance_delta,
        valence_delta=payload.valence_delta,
        intensity_delta=payload.emotional_intensity_delta,
        store=cast("MemoryStoreProtocol", store),
    )
    mem_read = MemoryRead.model_validate(asdict(updated))
    if idem_key:
        store.set_idempotent(idem_key, mem_read.model_dump())
    return mem_read


@router.get(
    "/best",
    response_model=list[MemoryRead],
    response_model_exclude_none=True,
    responses={
        **COMMON_RESPONSES,
        "200": {"content": {"application/json": {"example": [MEMORY_EXAMPLE]}}},
    },
)
async def best_memories(
    request: Request,
    n: int = Query(50, ge=1, le=500, alias="limit"),
    level: int | None = Query(None, ge=0),
    user_id: str | None = Query(None, examples=["user-123"]),
    importance: float | None = Query(
        None, ge=0.0, description="Weight for importance when ranking"
    ),
    emotional_intensity: float | None = Query(
        None,
        ge=0.0,
        alias="arousal",
        description="Weight for emotional intensity (arousal)",
    ),
    valence_pos: float | None = Query(None, ge=0.0, description="Weight for positive valence"),
    valence_neg: float | None = Query(None, ge=0.0, description="Weight for negative valence"),
    score_parts: bool = Query(
        False,
        description="Include score component breakdown (dev only)",
    ),
) -> list[MemoryRead]:
    store = await _store(request)
    meta = {"user_id": user_id} if user_id else None

    weights: ListBestWeights | None = None
    if any(v is not None for v in (importance, emotional_intensity, valence_pos, valence_neg)):
        weights = ListBestWeights(
            importance=importance or 1.0,
            emotional_intensity=emotional_intensity or 1.0,
            valence_pos=valence_pos or 1.0,
            valence_neg=valence_neg or 0.5,
        )

    records = await list_best(
        n=n,
        store=cast("MemoryStoreProtocol", store),
        level=level,
        metadata_filter=meta,
        weights=weights,
    )

    include_parts = False
    if score_parts:
        try:
            from memory_system.settings import get_settings

            include_parts = get_settings().profile == "development"
        except ImportError:  # pragma: no cover - settings optional
            include_parts = False

    dyn = MemoryDynamics(weights=weights or ListBestWeights()) if include_parts else None
    payload: list[MemoryRead] = []
    for r in records:
        data = asdict(r)
        data["id"] = data.pop("memory_id")
        if include_parts and dyn is not None:
            store_mem = Memory(
                id=r.memory_id,
                text=r.text,
                created_at=r.created_at,
                importance=r.importance,
                valence=r.valence,
                emotional_intensity=r.emotional_intensity,
                metadata=r.metadata,
                level=0,
                episode_id=r.episode_id,
                modality=r.modality,
                connections=r.connections,
            )
            score, parts = cast(
                "tuple[float, dict[str, float]]",
                dyn.score(store_mem, return_parts=True),
            )
            parts["total"] = score
            data["score_parts"] = parts
        model = MemoryRead.model_validate(data)
        payload.append(model)
    return payload


@router.get(
    "/aging",
    summary="Memory aging data",
    response_model=list[MemoryAging],
    response_model_exclude_none=True,
    responses={
        **COMMON_RESPONSES,
        "200": {"content": {"application/json": {"example": [MEMORY_AGING_EXAMPLE]}}},
    },
)
async def aging(
    request: Request,
    limit: int = Query(100, ge=1, le=1000, examples=[100]),
) -> list[MemoryAging]:
    """Return age and retention metrics for recent memories."""
    store = await _store(request)
    records = await store.list_recent(n=limit)
    now = dt.datetime.now(dt.UTC)
    dyn = MemoryDynamics()
    payload: list[MemoryAging] = []
    for r in records:
        last = last_accessed(r)
        age_days = (now - last).total_seconds() / 86_400.0
        freq = 0
        if r.metadata:
            try:
                freq = int(r.metadata.get("access_count", 0))
            except (ValueError, TypeError):
                freq = 0
        imp = min(1.0, r.importance * (1.0 + math.log1p(freq)))
        retention = dyn.decay(
            importance=imp,
            valence=r.valence,
            emotional_intensity=r.emotional_intensity,
            age_days=age_days,
        )
        payload.append(
            MemoryAging(
                id=r.id,
                age_days=age_days,
                retention=retention,
                access_count=freq,
            )
        )
    return payload


@router.post(
    "/forget",
    summary="Forget low-value or low-trust memories",
    response_model=dict[str, int],
    responses=COMMON_RESPONSES,
)
async def forget(
    request: Request,
    payload: ForgetPayload = Body(
        default_factory=ForgetPayload,
        examples=[FORGET_EXAMPLE],
    ),
) -> dict[str, int]:
    """Trigger forgetting of memories by decay score or policy."""
    store = await _store(request)
    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        cached = store.get_idempotent(idem_key)
        if cached is not None:
            return cached
    vector_store = get_vector_store(request)
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store unavailable")
    index = cast("FaissHNSWIndex", getattr(vector_store, "index", vector_store))
    kwargs: dict[str, Any] = {}
    if payload.policy == "low_trust":
        kwargs["low_trust"] = payload.threshold
    deleted = await forget_old_memories(store, index, **kwargs)
    result = {"deleted": deleted}
    if idem_key:
        store.set_idempotent(idem_key, result)
    return result


@router.post(
    "/consolidate",
    summary="Consolidate similar memories",
    response_model=dict[str, int],
    responses=COMMON_RESPONSES,
)
async def consolidate(request: Request) -> dict[str, int]:
    """Cluster and summarise memories into new entries."""
    store = await _store(request)
    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        cached = store.get_idempotent(idem_key)
        if cached is not None:
            return cached
    vector_store = get_vector_store(request)
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store unavailable")
    index = cast("FaissHNSWIndex", getattr(vector_store, "index", vector_store))
    created = await consolidate_store(store, index, strategy="contribution")
    result = {"created": len(created)}
    if idem_key:
        store.set_idempotent(idem_key, result)
    return result
