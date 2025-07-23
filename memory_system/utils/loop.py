"""Event loop utilities for Unified Memory System."""

from __future__ import annotations

import asyncio


def get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Return the running event loop or create a new one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
