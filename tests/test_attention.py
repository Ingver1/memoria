import datetime as dt

import pytest

from memory_system.core.keyword_attention import (
    attention_weights,
    order_by_attention,
    score_embeddings,
    score_token_overlap,
)
from memory_system.unified_memory import Memory, _get_cache


def _mem(text: str) -> Memory:
    now = dt.datetime.now(dt.UTC)
    return Memory("id" + text, text, now)


def test_punctuation_and_stopwords() -> None:
    memories = [_mem("hello world"), _mem("foo bar")]
    weights = attention_weights(memories, "Hello, and world!")
    assert weights[0] > weights[1]


def test_stopword_only_query() -> None:
    memories = [_mem("foo"), _mem("bar")]
    weights = attention_weights(memories, "and the or but")
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(0.5)


def test_stopwords_in_memory_do_not_dominate() -> None:
    memories = [_mem("keyword match"), _mem("and the or but")]
    weights = attention_weights(memories, "keyword and the")
    assert weights[0] > weights[1]


def test_metadata_weight_changes_ranking() -> None:
    now = dt.datetime.now(dt.UTC)
    mem_low = Memory("id1", "foo bar", now, metadata={"weight": 0.1})
    mem_high = Memory("id2", "foo bar", now, metadata={"weight": 5.0})
    weights = attention_weights([mem_low, mem_high], "foo", weight_key="weight")
    assert weights[1] > weights[0]


def test_attention_similarity_caching() -> None:
    cache = _get_cache()
    cache.clear()
    mem = _mem("foo bar")
    query = "foo"

    attention_weights([mem], query)
    first = cache.get_stats()

    attention_weights([mem], query)
    second = cache.get_stats()

    assert first["hit_rate"] == 0.0
    assert second["hit_rate"] > first["hit_rate"]
    cache.clear()


def test_ordering_with_token_overlap() -> None:
    memories = [_mem("hello world"), _mem("foo bar")]
    ordered = order_by_attention(memories, "hello", scoring=score_token_overlap)
    assert ordered[0].text == "hello world"


def test_ordering_with_embeddings() -> None:
    memories = [_mem("hello world"), _mem("foo bar")]
    ordered = order_by_attention(memories, "hello world", scoring=score_embeddings)
    assert ordered[0].text == "hello world"
