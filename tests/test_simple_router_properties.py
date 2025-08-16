import pytest

pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings, strategies as st

from memory_system.router import Channel, SimpleRouter

pytestmark = [pytest.mark.property, pytest.mark.needs_hypothesis]


@given(st.text(min_size=0, max_size=SimpleRouter.MIN_QUERY_LEN - 1))
@settings(max_examples=25)
def test_short_queries_disable_retrieval(query: str) -> None:
    router = SimpleRouter()
    decision = router.decide(query)
    assert decision.use_retrieval is False
    assert decision.target_channels == []
    assert decision.is_global_query is True


@given(prefix=st.sampled_from(["no memory:", "nomem:"]), tail=st.text())
@settings(max_examples=25)
def test_opt_out_prefix_disables_retrieval(prefix: str, tail: str) -> None:
    router = SimpleRouter()
    decision = router.decide(prefix + tail)
    assert decision.use_retrieval is False
    assert decision.target_channels == []
    assert decision.is_global_query is True


@given(trigger=st.sampled_from(SimpleRouter.PERSONAL_TRIGGERS), tail=st.text(min_size=3))
@settings(max_examples=25)
def test_personal_triggers_select_personal_channel(trigger: str, tail: str) -> None:
    q = (trigger + tail).lower()
    assume(not any(t in q for t in SimpleRouter.GLOBAL_TRIGGERS))
    router = SimpleRouter()
    decision = router.decide(q)
    assert decision.use_retrieval is True
    assert decision.is_global_query is False
    assert decision.target_channels == [Channel.PERSONAL]


@given(trigger=st.sampled_from(SimpleRouter.GLOBAL_TRIGGERS), tail=st.text(min_size=3))
@settings(max_examples=25)
def test_global_triggers_select_global_channel(trigger: str, tail: str) -> None:
    q = (trigger + tail).lower()
    assume(not any(t in q for t in SimpleRouter.PERSONAL_TRIGGERS))
    router = SimpleRouter()
    decision = router.decide(q)
    assert decision.use_retrieval is True
    assert decision.is_global_query is True
    assert Channel.GLOBAL in decision.target_channels
