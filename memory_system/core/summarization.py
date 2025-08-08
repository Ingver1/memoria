"""Summarization strategies for memory consolidation."""

from __future__ import annotations

from typing import Dict, Protocol, Sequence

from memory_system.core.store import Memory


class SummaryStrategy(Protocol):
    """Callable protocol for summarization strategies."""

    def __call__(self, memories: Sequence[Memory]) -> str:
        """Return a summary string for the given memories."""
        ...


def head2tail(memories: Sequence[Memory]) -> str:
    """Join up to two most important texts with an ellipsis."""
    if not memories:
        return ""
    top = sorted(
        memories,
        key=lambda m: (m.importance, m.created_at),
        reverse=True,
    )[:2]
    return " … ".join(m.text.strip() for m in top)


def concat(memories: Sequence[Memory]) -> str:
    """Concatenate all memory texts with spaces."""
    return " ".join(m.text.strip() for m in memories)


STRATEGIES: Dict[str, SummaryStrategy] = {
    "head2tail": head2tail,
    "concat": concat,
}
