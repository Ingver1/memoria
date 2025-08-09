import logging

import pytest

np = pytest.importorskip("numpy")

from embedder import _model, embed


def test_embed_logs_batch_size_and_zero_for_empty_string(caplog):
    dim = _model.get_sentence_embedding_dimension()
    with caplog.at_level(logging.INFO):
        vec = embed("")
    assert np.array_equal(vec, np.zeros(dim, dtype=np.float32))
    assert "Embedding batch size 1" in caplog.text


def test_embed_all_empty_raises():
    with pytest.raises(ValueError):
        embed(["", ""])


def test_embed_empty_list_raises():
    with pytest.raises(ValueError):
        embed([])
