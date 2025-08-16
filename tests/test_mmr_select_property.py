import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings, strategies as st

from memory_system.rag_router import mmr_select

pytestmark = [pytest.mark.property, pytest.mark.needs_hypothesis]


def _vec() -> st.SearchStrategy[NDArray[np.float32]]:
    return (
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=3,
        )
        .map(np.array)
        .map(lambda v: v.astype(np.float32))
    )


@given(base=_vec(), other=_vec())
@settings(max_examples=50)
def test_mmr_select_filters_duplicates(
    base: NDArray[np.float32], other: NDArray[np.float32]
) -> None:
    assume(np.linalg.norm(base) > 0)
    assume(np.linalg.norm(other) > 0)
    base = base / np.linalg.norm(base)
    other = other / np.linalg.norm(other)
    assume(abs(np.dot(base, other)) < 0.99)

    docs = np.stack([base, base, other])
    selected = mmr_select(base, docs, k=2)
    assert len(selected) == 2
    # NumPy stub does not support fancy indexing; fetch elements individually
    v0, v1 = docs[selected[0]], docs[selected[1]]
    sim = float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    assert sim < 0.99


@given(base=_vec(), docs=st.lists(_vec(), min_size=3, max_size=6))
@settings(max_examples=25)
def test_mmr_select_returns_unique_indices(
    base: NDArray[np.float32], docs: list[NDArray[np.float32]]
) -> None:
    assume(np.linalg.norm(base) > 0)
    for d in docs:
        assume(np.linalg.norm(d) > 0)
    base = base / np.linalg.norm(base)
    docs_arr = np.stack([d / np.linalg.norm(d) for d in docs])
    k = min(3, len(docs_arr))
    selected = mmr_select(base, docs_arr, k=k)
    assert len(selected) == k
    assert len(set(selected)) == len(selected)
    assert all(0 <= idx < len(docs_arr) for idx in selected)
