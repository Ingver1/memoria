"""
Scenario helpers for ``ltm_bench``.

The :mod:`ltm_bench.scenario` package collects small synthetic dialogues and
other fixtures used by benchmarks and tests.  They provide deterministic
inputs so results are reproducible across runs.
"""

from .synthetic import DEFAULT_DIALOGUE, PII_EMAIL, Turn, basic_dialogue

__all__ = [
    "DEFAULT_DIALOGUE",
    "PII_EMAIL",
    "Turn",
    "basic_dialogue",
]
