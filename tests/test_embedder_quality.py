import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


COSINE_THRESHOLD = 0.8


def test_embedder_similarity_quality() -> None:
    from embedder import embed

    data_path = Path(__file__).parent / "data" / "embedding_pairs.json"
    with data_path.open(encoding="utf-8") as fh:
        pairs = json.load(fh)

    similar_scores: list[float] = []
    different_scores: list[float] = []

    for item in pairs:
        s1, s2 = item["pair"]
        vecs = embed([s1, s2])
        score = float(np.dot(vecs[0], vecs[1]))
        if item["label"] == "similar":
            similar_scores.append(score)
        else:
            different_scores.append(score)

    assert similar_scores, "No similar pairs provided"
    assert different_scores, "No different pairs provided"

    min_similar = min(similar_scores)
    max_different = max(different_scores)

    assert min_similar > COSINE_THRESHOLD
    assert min_similar > max_different
