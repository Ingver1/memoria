from __future__ import annotations

import base64
from collections.abc import Iterable
from pathlib import Path

from memory_system.utils.security import EnhancedPIIFilter

from .metrics import MAGIC_HEADER, EncState, verify_encryption


def ensure_no_pii(results: Iterable[str]) -> bool:
    """Return ``True`` if ``results`` contain no PII according to the filter."""
    pii = EnhancedPIIFilter()
    return all(not pii.detect(text) for text in results)


def run_report(index_path: str | Path, search_results: Iterable[str]) -> dict[str, object]:
    """
    Run security checks and return a small summary report.

    Parameters
    ----------
    index_path: str | Path
        Location of the vector index/metadata file.
    search_results: Iterable[str]
        Text snippets returned by a search operation.

    Returns
    -------
    dict[str, object]
        Dictionary with ``encryption`` state and ``pii_safe`` flag.

    """
    enc_state = verify_encryption(index_path)
    pii_ok = ensure_no_pii(search_results)
    return {"encryption": enc_state.value, "pii_safe": pii_ok}


def _fake_token() -> bytes:
    """Generate a minimal Fernet-like token for tests."""
    return b"\x80" + b"\x00" * 8 + b"\x00" * 16 + b"\x01" + b"\x00" * 32


def test_verify_encryption_and_header(tmp_path: Path) -> None:
    token_line = base64.urlsafe_b64encode(_fake_token())
    dump = MAGIC_HEADER + b"\n" + token_line + b"\n"
    path = tmp_path / "dump.txt"
    path.write_bytes(dump)
    report = run_report(path, ["safe text"])
    assert report == {"encryption": EncState.ENCRYPTED.value, "pii_safe": True}


def test_verify_encryption_missing_header(tmp_path: Path) -> None:
    token_line = base64.urlsafe_b64encode(_fake_token())
    path = tmp_path / "plain.txt"
    path.write_bytes(token_line + b"\n")
    assert verify_encryption(path) == EncState.PLAINTEXT


def test_verify_encryption_invalid_length(tmp_path: Path) -> None:
    short_token = base64.urlsafe_b64encode(b"\x80short")
    path = tmp_path / "short.txt"
    path.write_bytes(MAGIC_HEADER + b"\n" + short_token + b"\n")
    assert verify_encryption(path) == EncState.PLAINTEXT
