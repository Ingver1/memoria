"""Synthetic dialogue benchmark for the Unified Memory System."""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING

import typer

from memory_system import EnhancedMemoryStore, UnifiedSettings

if TYPE_CHECKING:
    from collections.abc import Iterator

app = typer.Typer(help="Benchmark synthetic dialogues for recall and forgetting rate.")


def _dialogue(turns: int) -> Iterator[tuple[str, str]]:
    """Generate a deterministic user/assistant dialogue."""
    for i in range(turns):
        yield "user", f"user message {i}"
        yield "assistant", f"assistant reply {i}"


async def _run(
    turns: int,
    *,
    vector_dim: int,
    retain_fraction: float,
    seed: int,
) -> tuple[float, float]:
    """Insert synthetic messages and measure recall/forgetting."""
    settings = UnifiedSettings.for_testing()
    settings.model.vector_dim = vector_dim
    rng = random.Random(seed)  # noqa: S311 - deterministic for reproducible tests
    async with EnhancedMemoryStore(settings) as store:
        memories: list[tuple[str, list[float]]] = []
        for role, text in _dialogue(turns):
            vec = [rng.random() for _ in range(vector_dim)]
            mem = await store.add_memory(text=text, role=role, embedding=vec)
            memories.append((mem.id, vec))

        hits = 0
        for mem_id, vec in memories:
            res = await store.semantic_search(vector=vec, k=1)
            if res and res[0].id == mem_id:
                hits += 1
        recall = hits / len(memories) if memories else 0.0

        deleted = await store.forget_memories(min_total=0, retain_fraction=retain_fraction)
        forgetting_rate = deleted / len(memories) if memories else 0.0

    return recall, forgetting_rate


@app.command()
def dialogue(
    turns: int = typer.Option(50, help="Number of dialogue turns."),
    vector_dim: int = typer.Option(32, help="Embedding dimension."),
    retain_fraction: float = typer.Option(
        0.8, help="Fraction of memories to retain during forgetting."
    ),
    seed: int = typer.Option(42, help="PRNG seed."),
) -> None:
    """Run the synthetic dialogue benchmark."""
    recall, forgetting_rate = asyncio.run(
        _run(
            turns,
            vector_dim=vector_dim,
            retain_fraction=retain_fraction,
            seed=seed,
        )
    )
    typer.echo(f"Recall: {recall:.2%}")
    typer.echo(f"Forgetting rate: {forgetting_rate:.2%}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()
