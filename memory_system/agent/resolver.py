from __future__ import annotations

"""Simple resolver that evaluates multiple answer variants."""

import logging
from collections.abc import Awaitable, Callable

from memory_system.core.store import get_store
from memory_system.unified_memory import ExperienceCard, MemoryStoreProtocol, add

logger = logging.getLogger(__name__)


async def resolve(
    question: str,
    *,
    n: int = 3,
    generator: Callable[[str, int], Awaitable[tuple[str, str]]],
    verifier: Callable[[str], Awaitable[tuple[bool, str | None]]],
    store: MemoryStoreProtocol | None = None,
) -> tuple[str, str, list[str]]:
    """
    Return first verified answer generated for ``question``.

    The ``generator`` coroutine is invoked up to ``n`` times producing a tuple
    of ``(answer, chain)``.  Each ``answer`` is checked with ``verifier``
    which returns ``(ok, reason)``.  The first passing variant is returned and
    its reasoning chain stored in long‑term memory via
    :func:`memory_system.unified_memory.add`.

    Args:
        question: Problem to solve.
        n: Number of variants to try. Defaults to ``3``.
        generator: Callable returning ``(answer, chain)``.
        verifier: Callable returning ``(ok, reason)`` where ``reason`` describes
            failure or ``None`` on success.
        store: Optional memory store used for persistence.

    Returns:
        A tuple ``(answer, chain, antipatterns)`` where ``answer`` and ``chain``
        correspond to the winning variant and ``antipatterns`` contains failure
        reasons from rejected variants.

    Raises:
        ValueError: If no candidate passes verification.

    """
    st: MemoryStoreProtocol
    if store is None:
        st = await get_store()
    else:
        st = store
    antipatterns: list[str] = []
    for i in range(n):
        answer, chain = await generator(question, i)
        logger.debug("variant.generated", extra={"index": i, "chain": chain})
        ok, reason = await verifier(answer)
        if ok:
            logger.info("variant.accepted", extra={"index": i})
            card = ExperienceCard(
                situation=question,
                approach=chain,
                result=answer,
            )
            await add(
                chain,
                card=card,
                metadata={"question": question, "answer": answer},
                store=st,
            )
            return answer, chain, antipatterns
        logger.warning("variant.rejected", extra={"index": i, "reason": reason, "chain": chain})
        antipatterns.append(reason or "unspecified")
        fail_card = ExperienceCard(
            situation=question,
            approach=chain,
            result=answer,
            antipattern=reason,
        )
        await add(
            chain,
            card=fail_card,
            metadata={"question": question, "answer": answer, "failure": reason},
            memory_type="lesson",
            store=st,
        )
    raise ValueError("all generated answers failed verification")
