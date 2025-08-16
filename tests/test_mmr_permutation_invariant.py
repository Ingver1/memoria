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


@given(
    query=_vec(),
    docs=st.lists(_vec(), min_size=1, max_size=6),
    k_raw=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=25)
def test_mmr_selection_invariant_under_permutation(
    query: NDArray[np.float32], docs: list[NDArray[np.float32]], k_raw: int
) -> None:
    assume(np.linalg.norm(query) > 0)
    for d in docs:
        assume(np.linalg.norm(d) > 0)
    docs_arr = np.stack([d / np.linalg.norm(d) for d in docs])
    query = query / np.linalg.norm(query)
    k = min(k_raw, len(docs_arr))
    sel1 = mmr_select(query, docs_arr, k)
    sel1_vecs = sorted(tuple(docs_arr[i]) for i in sel1)
    docs_rev = docs_arr[::-1]
    sel2 = mmr_select(query, docs_rev, k)
    sel2_vecs = sorted(tuple(docs_rev[i]) for i in sel2)
    assert sel1_vecs == sel2_vecs
