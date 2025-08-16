"""Minimal pydantic-settings stub used when the real package is unavailable."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - best effort import
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - no real pydantic

    class BaseModel:  # type: ignore[override]
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

        def model_dump(self) -> dict[str, object]:  # minimal helper
            return dict(self.__dict__)


class BaseSettings(BaseModel):
    def __init__(self, **data):  # type: ignore[override]
        super().__init__(**data)


def SettingsConfigDict(**kwargs: Any) -> dict[str, Any]:
    """Simplified stand-in for :func:`pydantic_settings.SettingsConfigDict`."""
    return dict(**kwargs)
