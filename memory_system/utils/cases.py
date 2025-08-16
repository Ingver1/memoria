from __future__ import annotations

"""Utility helpers for case-based reasoning."""

from memory_system.core.store import Case


def adapt_case(case: Case, new_problem: str) -> Case:
    """
    Return a new :class:`Case` with plan adapted to ``new_problem``.

    The adaptation strategy is intentionally simple: occurrences of the
    original problem string inside the plan are replaced with ``new_problem``.
    Other fields are copied verbatim.
    """
    plan = case.plan.replace(case.problem, new_problem) if case.problem else case.plan
    return Case.new(
        problem=new_problem,
        plan=plan,
        outcome=case.outcome,
        evaluation=case.evaluation,
        embedding=case.embedding,
    )
