from __future__ import annotations

import base64
import binascii
from enum import Enum
from pathlib import Path

MAGIC_HEADER = b"# MEMORIA_DUMP_v1"
FERNET_OVERHEAD = 1 + 8 + 16 + 32  # version + timestamp + IV + HMAC


class EncState(Enum):
    ENCRYPTED = "encrypted"
    PLAINTEXT = "plaintext"
    UNKNOWN = "unknown"


def verify_encryption(
    path: str | Path, *, scheme: str = "fernet", sample_lines: int = 64
) -> EncState:
    """
    Heuristically validate that a text-ish artifact contains encrypted content.

    - For ``'fernet'`` expects urlsafe-base64 and version byte ``0x80`` after decode.
    - Reads only a small sample; safe for big files.
    - Returns ``UNKNOWN`` for binary formats we do not intend to encrypt.

    NOTE: This is an audit helper for tests/benchmarks, not a cryptographic proof.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return EncState.UNKNOWN

    # We only know how to sanity-check text/line-oriented outputs.
    # Skip obvious binaries (e.g., FAISS index files).
    if p.suffix.lower() in {".faiss", ".idx", ".bin"}:
        return EncState.UNKNOWN

    if scheme != "fernet":
        return EncState.UNKNOWN

    checked = 0
    with p.open("rb") as fh:
        header = fh.readline().rstrip(b"\n")
        if header != MAGIC_HEADER:
            return EncState.PLAINTEXT
        for raw_line in fh:
            if checked >= sample_lines:
                break
            line = raw_line.strip()
            if not line:
                continue
            # must be urlsafe base64 and decode to bytes starting with 0x80 (Fernet version)
            try:
                decoded = base64.urlsafe_b64decode(line)
            except (binascii.Error, ValueError):
                return EncState.PLAINTEXT
            if len(decoded) <= FERNET_OVERHEAD or decoded[0] != 0x80:
                return EncState.PLAINTEXT
            checked += 1

    # If we had at least one non-empty line and all matched Fernet shape
    return EncState.ENCRYPTED if checked > 0 else EncState.UNKNOWN
