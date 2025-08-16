from memory_system.router import SimpleRouter


def test_short_query_disabled():
    decision = SimpleRouter.decide("hi")
    assert decision.targets == ()


def test_no_memory_prefix():
    decision = SimpleRouter.decide("no memory: skip search")
    assert decision.targets == ()


def test_personal_trigger():
    decision = SimpleRouter.decide("remember my birthday")
    assert decision.targets == ("personal",)


def test_project_trigger():
    decision = SimpleRouter.decide("error in repo")
    assert decision.targets == ("project",)


def test_both_triggers():
    decision = SimpleRouter.decide("remember to fix the repo")
    assert decision.targets == ("personal", "project")


def test_force_project_context():
    decision = SimpleRouter.decide("remember me", force_project_context=True)
    assert decision.targets == ("personal", "project")
