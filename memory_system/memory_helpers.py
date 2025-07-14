"""Compatibility wrapper exposing high-level memory helper functions."""
from __future__ import annotations

import unified_memory as _u

Memory = _u.Memory
MemoryStoreProtocol = _u.MemoryStoreProtocol

add = _u.add
search = _u.search
delete = _u.delete
update = _u.update
list_recent = _u.list_recent
list_best = _u.list_best
set_default_store = _u.set_default_store
get_default_store = _u.get_default_store

__all__ = [
    "Memory",
    "MemoryStoreProtocol",
    "add",
    "search",
    "delete",
    "update",
    "list_recent",
    "list_best",
    "set_default_store",
    "get_default_store",
]
