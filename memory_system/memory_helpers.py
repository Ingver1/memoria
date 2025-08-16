"""Compatibility wrapper exposing high-level memory helper functions."""

from __future__ import annotations

from . import unified_memory as _u

Memory = _u.Memory
MemoryStoreProtocol = _u.MemoryStoreProtocol
ListBestWeights = _u.ListBestWeights
PersonalCard = _u.PersonalCard

add = _u.add
search = _u.search
delete = _u.delete
update = _u.update
update_trust_scores = _u.update_trust_scores
last_accessed = _u.last_accessed
list_recent = _u.list_recent
list_best = _u.list_best
promote_personal = _u.promote_personal
set_default_store = _u.set_default_store
get_default_store = _u.get_default_store

__all__ = [
    "ListBestWeights",
    "Memory",
    "MemoryStoreProtocol",
    "PersonalCard",
    "add",
    "delete",
    "get_default_store",
    "last_accessed",
    "list_best",
    "list_recent",
    "promote_personal",
    "search",
    "set_default_store",
    "update",
    "update_trust_scores",
]
