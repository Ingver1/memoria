from __future__ import annotations

"""Synthetic dialogue helpers for benchmarks and tests.

This module provides small deterministic chat transcripts used across the
long‑term memory benchmark suite.  The transcripts intentionally contain
personally identifiable information (PII) so that security checks and PII
filters can be exercised in a repeatable manner.
"""

from dataclasses import dataclass


@dataclass
class Turn:
    """
    Single dialogue turn.

    Attributes
    ----------
    speaker:
        Name of the speaker (e.g. ``"user"`` or ``"assistant"``).
    text:
        Utterance text for the turn.

    """

    speaker: str
    text: str


PII_EMAIL = "test@example.com"
"""Email address embedded in :func:`basic_dialogue` for PII tests."""


def basic_dialogue() -> list[Turn]:
    """
    Return a short conversation containing PII.

    The exchange is deterministic and tiny, making it suitable for tests
    and micro‑benchmarks that require predictable text with PII elements.
    """
    return [
        Turn("user", "Hello there!"),
        Turn("assistant", "Hi, how can I help?"),
        Turn("user", f"My email is {PII_EMAIL}"),
        Turn("assistant", f"Thanks, I'll remember {PII_EMAIL}"),
    ]


# Eagerly evaluate a default instance for convenience in tests.
DEFAULT_DIALOGUE: list[Turn] = basic_dialogue()
