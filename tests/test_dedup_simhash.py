from memory_system.core.dedup import is_near_duplicate, simhash


def test_simhash_is_stable():
    # Specific value derived from deterministic blake2b hashing
    assert simhash("hello world") == 13723176454590477


def test_is_near_duplicate_simple():
    assert is_near_duplicate("Hello world!", ["other", "hello world"]) is True
    assert is_near_duplicate("unique text", ["completely different"]) is False
