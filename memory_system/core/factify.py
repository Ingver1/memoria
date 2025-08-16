"""
Minimal fact extraction placeholder.

The real implementation would call an external service to extract
structured facts.  For now we simply echo the input as the canonical
claim and return empty structures for triples and evidence.
"""

from __future__ import annotations

from typing import Any


def factify(text: str) -> dict[str, Any]:
    """
    Extract factual structure from ``text``.

    Parameters
    ----------
    text:
        The input text from which to extract facts.

    Returns
    -------
    dict
        A mapping containing ``canonical_claim`` (str), ``triples`` (list)
        and ``evidence`` (list of ``{url, status, retrieved_at}`` dicts).

    """
    return {
        "canonical_claim": text.strip(),
        "triples": [],
        "evidence": [],
    }
