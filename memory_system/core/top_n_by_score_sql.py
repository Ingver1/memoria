"""Helper to construct SQL for weighted top-N queries."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import MutableMapping, Sequence
from typing import Any

from memory_system.unified_memory import ListBestWeights

_RECENCY_TAU = 86_400.0
_METADATA_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_metadata_key(key: str) -> None:
    """
    Ensure metadata keys are simple alphanumeric identifiers.

    Parameters
    ----------
    key:
        Metadata key to validate.

    Raises
    ------
    ValueError
        If the key contains potentially unsafe characters.

    """
    if not _METADATA_KEY_RE.fullmatch(key):
        raise ValueError(f"Invalid metadata filter key: {key}")


def build_top_n_by_score_sql(
    n: int,
    weights: ListBestWeights,
    *,
    level: int | None = None,
    metadata_filter: MutableMapping[str, Any] | None = None,
    ids: Sequence[str] | None = None,
) -> tuple[str, Sequence[Any]]:
    """Return SQL and params to fetch *n* memories ranked by weighted score."""
    now = dt.datetime.now(dt.UTC).isoformat()
    clauses: list[str] = [
        "m.valid_from <= ?",
        "m.valid_to >= ?",
        "m.tx_from <= ?",
        "m.tx_to >= ?",
    ]
    params: list[Any] = [now, now, now, now]
    if level is not None:
        clauses.append("m.level = ?")
        params.append(level)
    if metadata_filter:
        for key, val in metadata_filter.items():
            validate_metadata_key(key)
            if key in {"episode_id", "modality"}:
                clauses.append(f"m.{key} = ?")
                params.append(val)
            else:
                clauses.append("json_extract(m.metadata, ?) = ?")
                params.extend([f"$.{key}", val])
    if ids:
        placeholders = ", ".join(["?"] * len(ids))
        clauses.append(f"m.id IN ({placeholders})")
        params.extend(ids)
    score_expr = (
        "(m.importance * ?) + (m.emotional_intensity * ?) + "
        "(CASE WHEN m.valence >= 0 THEN m.valence * ? ELSE m.valence * ? END) + "
        "(? * exp(-(strftime('%s','now') - strftime('%s', COALESCE(json_extract(m.metadata, '$.last_accessed'), m.created_at)))/?)) + "
        "(? * log(1 + COALESCE(json_extract(m.metadata, '$.ema_access'), 0)))"
    )
    sql = (
        "SELECT m.id, m.text, m.created_at, m.valid_from, m.valid_to, m.tx_from, m.tx_to, m.importance, m.valence, "
        "m.emotional_intensity, m.level, m.episode_id, m.modality, m.connections, m.metadata, "
        "m.memory_type, m.pinned, m.ttl_seconds, m.last_used, m.success_score, m.decay "
        "FROM memories m"
    )
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += f" ORDER BY {score_expr} DESC LIMIT ?"
    params.extend(
        [
            weights.importance,
            weights.emotional_intensity,
            weights.valence_pos,
            weights.valence_neg,
            weights.recency,
            _RECENCY_TAU,
            weights.frequency,
            n,
        ]
    )
    return sql, params
