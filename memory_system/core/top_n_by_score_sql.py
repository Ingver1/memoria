"""Helper to construct SQL for weighted top-N queries."""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

from memory_system.unified_memory import ListBestWeights


def build_top_n_by_score_sql(
    n: int,
    weights: ListBestWeights,
    *,
    level: int | None = None,
    metadata_filter: MutableMapping[str, Any] | None = None,
    ids: Sequence[str] | None = None,
    min_score: float | None = None,
) -> tuple[str, Sequence[Any]]:
    """Return SQL and params to fetch *n* memories ranked by weighted score."""
    clauses: list[str] = []
    params: list[Any] = []
    if level is not None:
        clauses.append("m.level = ?")
        params.append(level)
    if metadata_filter:
        for key, val in metadata_filter.items():
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
        "(CASE WHEN m.valence >= 0 THEN m.valence * ? ELSE m.valence * ? END)"
    )
    sql = (
        "SELECT m.id, m.text, m.created_at, m.importance, m.valence, "
        "m.emotional_intensity, m.level, m.episode_id, m.modality, m.connections, m.metadata "
        "FROM memories m"
    )
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if min_score is not None:
        cond = f"{score_expr} >= ?"
        sql += (" AND " if clauses else " WHERE ") + cond
        params.extend(
            [
                weights.importance,
                weights.emotional_intensity,
                weights.valence_pos,
                weights.valence_neg,
                min_score,
            ]
        )
    sql += f" ORDER BY {score_expr} DESC LIMIT ?"
    params.extend(
        [
            weights.importance,
            weights.emotional_intensity,
            weights.valence_pos,
            weights.valence_neg,
            n,
        ]
    )
    return sql, params
