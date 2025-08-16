"""Tests for synthetic dialogue helpers used in benchmarks."""

from ltm_bench.scenario import DEFAULT_DIALOGUE, PII_EMAIL, Turn, basic_dialogue


def test_default_dialogue_contains_pii() -> None:
    """The canned dialogue should expose the example email for PII tests."""
    assert any(PII_EMAIL in turn.text for turn in DEFAULT_DIALOGUE)
    assert all(isinstance(turn, Turn) for turn in DEFAULT_DIALOGUE)


def test_basic_dialogue_reproducible() -> None:
    """``basic_dialogue`` should return a fresh but equivalent transcript."""
    d1 = basic_dialogue()
    d2 = basic_dialogue()
    assert d1 is not d2  # new list each call
    assert d1 == d2
