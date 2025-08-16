"""Event loop utilities for Unified Memory System."""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop

# Keep a module-level fallback loop reference captured at import time.
# sitecustomize/usercustomize set a default loop early; we reuse it to avoid
# creating new event loops in restricted sandboxes where creating a loop may
# attempt to open socket pairs and fail under seccomp.
try:  # Best-effort: do not create a loop if none is set yet
    _FALLBACK_LOOP: AbstractEventLoop | None = asyncio.get_event_loop()
except Exception:  # pragma: no cover - defensive in environments without a loop
    _FALLBACK_LOOP = None


def get_loop() -> AbstractEventLoop:
    """
    Return a loop even from sync context.

    - If a running loop exists -> return it.
    - Else return the current (set) loop or create a new one (main thread).
    """
    # Prefer an already running loop when available.
    try:
        return asyncio.get_running_loop()
    except Exception:
        pass

    # Fallback to a current (but not running) loop if set.
    try:
        return asyncio.get_event_loop()
    except Exception:
        # If no current loop is set, reuse the captured fallback when
        # available, restoring it as the current loop.
        if _FALLBACK_LOOP is not None and not _FALLBACK_LOOP.is_closed():
            asyncio.set_event_loop(_FALLBACK_LOOP)
            return _FALLBACK_LOOP

    # As a last resort, attempt to create a new loop. If this fails (e.g. in
    # restricted sandboxes forbidding socketpair), raise a consistent
    # RuntimeError so callers can treat it as "no running loop".
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("no usable event loop available") from exc
