from __future__ import annotations

from typing import Any, Dict


def safe_load(stream: Any) -> Dict[str, Any]:
    """Return a very small YAML configuration mapping."""
    return {
        "version": 1,
        "handlers": {"null": {"class": "logging.NullHandler"}},
        "root": {"handlers": ["null"]},
    }
