import pytest

from memory_system.core.index import (
    _QUERY_CNT,
    _QUERY_ERR,
    _VEC_ADDED,
    _VEC_DELETED,
)
from memory_system.utils.metrics import (
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    MEM_CREATED_TOTAL,
    MEM_DELETED_TOTAL,
    MET_ERRORS_TOTAL,
    MET_POOL_EXHAUSTED,
)


def _get_value(metric):
    if hasattr(metric, "value"):
        return metric.value
    try:
        return metric._value.get()
    except AttributeError:
        return metric._value


@pytest.mark.parametrize(
    "metric,label_args",
    [
        (MET_ERRORS_TOTAL, ("test", "component")),
        (MET_POOL_EXHAUSTED, ()),
        (MEM_CREATED_TOTAL, ("text",)),
        (MEM_DELETED_TOTAL, ("text",)),
        (CACHE_HITS_TOTAL, ()),
        (CACHE_MISSES_TOTAL, ()),
        (_VEC_ADDED, ()),
        (_VEC_DELETED, ()),
        (_QUERY_CNT, ()),
        (_QUERY_ERR, ()),
    ],
)
def test_counter_monotonic(metric, label_args):
    m = metric.labels(*label_args) if label_args else metric
    start = _get_value(m)
    m.inc()
    assert _get_value(m) == start + 1
    with pytest.raises(ValueError):
        m.inc(-1)
    assert _get_value(m) == start + 1
