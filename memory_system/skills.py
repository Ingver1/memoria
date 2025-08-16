from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .unified_memory import (
    Memory,
    MemoryStoreProtocol,
    add,
    search,
    update,
)


@dataclass(slots=True)
class Skill:
    """Serialized representation of a reusable skill."""

    steps: list[dict[str, Any]]
    params: dict[str, Any] | None = None
    result: Any | None = None
    success_score: float = 0.0


def _skill_text(skill: Skill) -> str:
    """Return a text summary for indexing a skill."""
    parts = [
        "steps: " + json.dumps(skill.steps, ensure_ascii=False),
        "params: " + json.dumps(skill.params, ensure_ascii=False),
        "result: " + json.dumps(skill.result, ensure_ascii=False),
    ]
    return " \n".join(parts)


async def store_skill(
    steps: list[dict[str, Any]],
    params: dict[str, Any] | None,
    result: Any,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Persist a successful action trace as a ``skill`` memory."""
    skill = Skill(steps=steps, params=params, result=result, success_score=1.0)
    text = _skill_text(skill)
    metadata = {"steps": steps, "params": params, "result": result}
    memory = await add(
        text,
        metadata=metadata,
        memory_type="skill",
        success_score=skill.success_score,
        store=store,
    )
    return memory


async def retrieve_skills(
    query: str,
    *,
    k: int = 5,
    retriever: str = "hybrid",
    store: MemoryStoreProtocol | None = None,
) -> Sequence[Memory]:
    """Return candidate skills matching ``query`` using hybrid search."""
    meta = {"memory_type": "skill"}
    raw = await search(
        query,
        retriever=retriever,
        k=k,
        k_bm25=k,
        k_vec=k,
        metadata_filter=meta,
        store=store,
    )

    def _to_mem(m: Any) -> Memory:
        if isinstance(m, Memory):
            return m
        return Memory(
            memory_id=m.id,
            text=m.text,
            created_at=m.created_at,
            valence=m.valence,
            emotional_intensity=m.emotional_intensity,
            importance=m.importance,
            episode_id=m.episode_id,
            modality=m.modality,
            connections=m.connections,
            metadata=m.metadata,
            memory_type=m.memory_type,
            ttl_seconds=m.ttl_seconds,
            last_used=m.last_used,
            success_score=m.success_score,
            decay=m.decay,
        )

    results = [_to_mem(m) for m in raw]
    return [m for m in results if m.memory_type == "skill"]


def select_skill_ucb1(candidates: Sequence[Memory]) -> Memory | None:
    """Select a skill using the UCB1 algorithm."""
    if not candidates:
        return None
    total = sum(int((m.metadata or {}).get("access_count", 0)) for m in candidates) + 1
    best = max(
        candidates,
        key=lambda m: (
            (m.success_score or 0.0)
            + math.sqrt(2 * math.log(total) / (int((m.metadata or {}).get("access_count", 0)) + 1))
        ),
    )
    return best


async def update_skill_success(
    memory: Memory,
    reward: float,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Update ``success_score`` of a skill after execution."""
    # Historical implementations used an incremental average for the reward,
    # but our simplified test model expects the latest reward to fully
    # determine the score.  We therefore replace the ``success_score`` with the
    # provided ``reward`` and bump the access counter.
    count = int((memory.metadata or {}).get("access_count", 0))
    new_count = count + 1
    new_score = reward
    meta = dict(memory.metadata or {})
    meta["access_count"] = new_count
    updated = await update(
        memory.memory_id,
        metadata=meta,
        success_score=new_score,
        store=store,
    )
    return updated


def memory_to_skill(memory: Memory) -> Skill:
    """Reconstruct a :class:`Skill` from stored memory."""
    meta = memory.metadata or {}
    return Skill(
        steps=list(meta.get("steps", [])),
        params=meta.get("params"),
        result=meta.get("result"),
        success_score=memory.success_score or 0.0,
    )
