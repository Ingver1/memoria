from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from memory_system.utils.security import EnhancedPIIFilter

from .metrics import verify_encryption


def ensure_no_pii(results: Iterable[str]) -> bool:
    """Return ``True`` if ``results`` contain no PII according to the filter."""
    pii = EnhancedPIIFilter()
    for text in results:
        if pii.detect(text):
            return False
    return True


def run_report(
    index_path: str | Path, search_results: Iterable[str]
) -> dict[str, object]:
    """Run security checks and return a small summary report.

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
