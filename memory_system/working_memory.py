"""
Simple working memory structures for agents.

The module provides lightweight in-memory components that model the classic
Baddeley & Hitch working memory: a central executive that manages a list of
ongoing tasks, a phonological loop for transient textual information and a
visuospatial sketchpad for UI or image references.

Each structure exposes small helper APIs designed to be easily consumed by
LLM based agents.  Completed state can be serialised into the long-term
``memory_system`` via :func:`serialize_state` so that important intermediate
steps are not lost once the agent moves on.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Awaitable, Callable, MutableMapping, Sequence

from memory_system.settings import get_settings

from .unified_memory import Memory, add, add_lesson, search, update_trust_scores

# ---------------------------------------------------------------------------
# Working memory components
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CentralExecutive:
    """Manage a stack of tasks and a focus pointer."""

    tasks: list[str] = field(default_factory=list)
    focus: int | None = None

    def push_task(self, task: str) -> None:
        """Push *task* onto the stack and focus it."""
        self.tasks.append(task)
        self.focus = len(self.tasks) - 1

    def pop_task(self) -> str | None:
        """Pop and return the current task."""
        if not self.tasks:
            self.focus = None
            return None
        task = self.tasks.pop()
        self.focus = len(self.tasks) - 1 if self.tasks else None
        return task

    def current_task(self) -> str | None:
        """Return the task currently in focus."""
        if self.focus is None:
            return None
        if 0 <= self.focus < len(self.tasks):
            return self.tasks[self.focus]
        return None

    def focus_next(self) -> None:
        """Advance focus to the next task (circular)."""
        if not self.tasks:
            self.focus = None
            return
        if self.focus is None:
            self.focus = 0
        else:
            self.focus = (self.focus + 1) % len(self.tasks)

    def focus_previous(self) -> None:
        """Move focus to the previous task (circular)."""
        if not self.tasks:
            self.focus = None
            return
        if self.focus is None:
            self.focus = 0
        else:
            self.focus = (self.focus - 1) % len(self.tasks)


@dataclass(slots=True)
class PhonologicalLoop:
    """Transient textual scratchpad."""

    buffer: list[str] = field(default_factory=list)

    def write(self, text: str) -> None:
        """Append ``text`` to the scratchpad."""
        self.buffer.append(text)

    def read(self) -> str:
        """Return the entire scratchpad contents."""
        return "\n".join(self.buffer)

    def clear(self) -> None:
        """Clear the scratchpad."""
        self.buffer.clear()


@dataclass(slots=True)
class VisuospatialSketchpad:
    """Store lightweight references to visual/UI artifacts."""

    references: list[str] = field(default_factory=list)

    def add_reference(self, ref: str) -> None:
        """Store a new visual reference."""
        self.references.append(ref)

    def list_references(self) -> list[str]:
        """Return all stored visual references."""
        return list(self.references)

    def clear(self) -> None:
        """Remove all visual references."""
        self.references.clear()


@dataclass(slots=True)
class WorkingMemoryItem:
    """Transient memory item stored with win rate metadata."""

    text: str
    win_rate: float
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))


@dataclass(slots=True)
class TaskCompletionConfig:
    """Configuration for persisting completed tasks to long-term memory."""

    lesson: str | None = None
    valence: float = 0.0
    importance: float = 0.0
    success_score: float = 0.0
    top_memories: Sequence[Memory] | None = None
    reasoner: Callable[[str, Sequence[Memory]], Awaitable[float]] | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkingMemory:
    """Container bundling all working-memory components."""

    central_executive: CentralExecutive = field(default_factory=CentralExecutive)
    phonological_loop: PhonologicalLoop = field(default_factory=PhonologicalLoop)
    visuospatial_sketchpad: VisuospatialSketchpad = field(default_factory=VisuospatialSketchpad)
    items: list[WorkingMemoryItem] = field(default_factory=list)

    # -- Convenience wrappers -------------------------------------------------

    def push_task(self, task: str) -> None:
        """Push *task* onto the task stack."""
        self.central_executive.push_task(task)

    async def pop_task(
        self,
        *,
        store: Any | None = None,  # noqa: ANN401
        config: TaskCompletionConfig | None = None,
    ) -> str | None:
        """Pop current task and persist it and an optional lesson to long-term memory."""
        task = self.central_executive.pop_task()
        if task is not None:
            await add(
                f"Completed task: {task}",
                memory_type="working",
                store=store,
            )
            cfg = config or TaskCompletionConfig()
            if cfg.lesson:
                await add_lesson(
                    cfg.lesson,
                    valence=cfg.valence,
                    importance=cfg.importance,
                    success_score=cfg.success_score,
                    store=store,
                )
            if cfg.top_memories and cfg.reasoner:
                await update_trust_scores(
                    task,
                    cfg.top_memories,
                    cfg.reasoner,
                    store=store,
                )
        return task

    def write_scratchpad(self, text: str) -> None:
        """Write text to the phonological loop."""
        self.phonological_loop.write(text)

    def add_visual_reference(self, ref: str) -> None:
        """Store a visual reference in the sketchpad."""
        self.visuospatial_sketchpad.add_reference(ref)

    def add_memory(
        self,
        text: str,
        win_rate: float,
        *,
        created_at: dt.datetime | None = None,
    ) -> None:
        """Add a new working-memory item."""
        item = WorkingMemoryItem(
            text=text,
            win_rate=win_rate,
            created_at=created_at or dt.datetime.now(dt.UTC),
        )
        self.items.append(item)
        self._evict()

    def _evict(self) -> None:
        budget = get_settings().working_memory.budget
        if len(self.items) <= budget:
            return
        now = dt.datetime.now(dt.UTC)

        def priority(it: WorkingMemoryItem) -> float:
            age = (now - it.created_at).total_seconds()
            recency = 1 / (age + 1)
            return recency * it.win_rate

        self.items.sort(key=priority, reverse=True)
        del self.items[budget:]

    # -- Serialisation --------------------------------------------------------

    async def serialize_state(
        self,
        *,
        store: Any | None = None,  # noqa: ANN401
        metadata: MutableMapping[str, Any] | None = None,
        persist: bool = False,
    ) -> Sequence[Memory]:
        """
        Serialise remaining working-memory state to long-term memory.

        Returns a sequence of :class:`~memory_system.unified_memory.Memory`
        objects representing the persisted entries.
        """
        persisted: list[Memory] = []

        if not persist:
            self.central_executive.tasks.clear()
            self.central_executive.focus = None
            self.phonological_loop.clear()
            self.visuospatial_sketchpad.clear()
            self.items.clear()
            return persisted

        # Persist outstanding tasks
        for task in list(self.central_executive.tasks):
            mem = await add(
                f"Outstanding task: {task}",
                memory_type="working",
                metadata=metadata,
                store=store,
            )
            persisted.append(mem)
        self.central_executive.tasks.clear()
        self.central_executive.focus = None

        # Persist scratchpad
        scratch = self.phonological_loop.read()
        if scratch:
            mem = await add(
                f"Scratchpad:\n{scratch}",
                memory_type="working",
                metadata=metadata,
                store=store,
            )
            persisted.append(mem)
        self.phonological_loop.clear()

        # Persist visual references
        visuals = self.visuospatial_sketchpad.list_references()
        if visuals:
            mem = await add(
                "Visual references: " + ", ".join(visuals),
                memory_type="working",
                metadata=metadata,
                store=store,
            )
            persisted.append(mem)
        self.visuospatial_sketchpad.clear()

        for item in list(self.items):
            meta = dict(metadata or {})
            meta.update({"created_at": item.created_at, "win_rate": item.win_rate})
            mem = await add(
                item.text,
                memory_type="working",
                metadata=meta,
                store=store,
            )
            persisted.append(mem)
        self.items.clear()

        return persisted

    # -- Agent integration ----------------------------------------------------

    async def build_prompt(
        self,
        user_message: str,
        *,
        store: Any | None = None,  # noqa: ANN401
        k: int = 5,
        metadata_filter: MutableMapping[str, Any] | None = None,
    ) -> str:
        """Return a prompt seeded with retrieved long-term context."""
        memories = await search(
            query=user_message,
            k=k,
            metadata_filter=metadata_filter,
            store=store,
        )
        parts = [m.text for m in memories]

        scratch = self.phonological_loop.read()
        if scratch:
            parts.append(f"Scratchpad:\n{scratch}")
        visuals = self.visuospatial_sketchpad.list_references()
        if visuals:
            parts.append("Visual references: " + ", ".join(visuals))

        parts.append(user_message)
        return "\n\n".join(parts)


__all__ = [
    "CentralExecutive",
    "PhonologicalLoop",
    "VisuospatialSketchpad",
    "WorkingMemory",
]
