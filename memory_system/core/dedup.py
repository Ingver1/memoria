"""Near-duplicate detection utilities."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

__all__ = ["hamming_distance", "is_near_duplicate", "simhash"]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _token_hash(token: str) -> int:
    """Return a stable 64-bit hash for *token*."""
    # hashlib.blake2b provides a fast, deterministic hash.  Using an 8 byte
    # digest gives us a 64-bit integer which we convert using big endian to
    # keep the bit ordering consistent across platforms.
    return int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def simhash(text: str) -> int:
    """Return a 64-bit SimHash for *text*."""
    bits = [0] * 64
    for token in _tokens(text):
        h = _token_hash(token)
        for i in range(64):
            bit = 1 << i
            bits[i] += 1 if h & bit else -1
    fingerprint = 0
    for i, v in enumerate(bits):
        if v > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two SimHashes."""
    x = a ^ b
    dist = 0
    while x:
        dist += 1
        x &= x - 1
    return dist


def is_near_duplicate(text: str, existing: Iterable[str], *, threshold: int = 3) -> bool:
    """Return True if *text* is a near-duplicate of any *existing* texts."""
    target = simhash(text)
    return any(hamming_distance(target, simhash(other)) <= threshold for other in existing)
