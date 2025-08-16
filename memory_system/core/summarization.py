"""Summarization strategies for memory consolidation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memory_system.core.store import Memory


@runtime_checkable
class SummaryStrategy(Protocol):
    """Callable protocol for summarization strategies."""

    def __call__(self, memories: Sequence[Memory | None]) -> str:
        """Return a summary string for the given memories."""
        ...


def head2tail(memories: Sequence[Memory | None]) -> str:
    """Join up to two most important texts with an ellipsis."""
    if not memories:
        return ""
    valid = [m for m in memories if m is not None]
    if not valid:
        return ""
    top = sorted(
        valid,
        key=lambda m: (m.importance, m.created_at),
        reverse=True,
    )[:2]
    return " … ".join(_extract_text(m) for m in top)


def concat(memories: Sequence[Memory | None]) -> str:
    """Concatenate all memory texts with spaces."""
    return " ".join(_extract_text(m) for m in memories if m is not None)


def _extract_text(m: Memory | None) -> str:
    """Return condensed lesson if available, else the raw text."""
    if m is None:
        return ""
    meta = getattr(m, "metadata", None)
    card = getattr(m, "card", None)
    if card is None and meta and isinstance(meta, Mapping):
        card = meta.get("card")
    if isinstance(card, Mapping):
        lesson = card.get("lesson")
        if isinstance(lesson, str) and lesson.strip():
            return lesson.strip()
    if (
        card is not None
        and hasattr(card, "lesson")
        and isinstance(card.lesson, str)
        and card.lesson
    ):
        return card.lesson.strip()
    return m.text.strip()


STRATEGIES: dict[str, SummaryStrategy] = {
    "head2tail": head2tail,
    "concat": concat,
}
