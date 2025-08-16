import importlib
import sys
import types
from contextlib import contextmanager

import pytest


@contextmanager
def no_grad():
    yield


dummy_torch = types.SimpleNamespace(
    cuda=types.SimpleNamespace(is_available=lambda: False),
    no_grad=no_grad,
)


class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, *_, **__):  # pragma: no cover - simple stub
        return cls()

    def __call__(self, *_, **__):  # pragma: no cover - simple stub
        class DummyInputs(dict):
            def to(self, *_):  # pragma: no cover - simple stub
                return self

        return DummyInputs()


class DummyModel:
    @classmethod
    def from_pretrained(cls, *_, **__):  # pragma: no cover - simple stub
        return cls()

    def eval(self):  # pragma: no cover - simple stub
        return self

    def to(self, *_):  # pragma: no cover - simple stub
        return self

    def __call__(self, **_):  # pragma: no cover - simple stub
        class DummyLogits:
            def squeeze(self):  # pragma: no cover - simple stub
                class DummyVal:
                    def item(self) -> float:  # pragma: no cover - simple stub
                        return 0.0

                return DummyVal()

        return types.SimpleNamespace(logits=DummyLogits())


class DummyPeft:
    @staticmethod
    def from_pretrained(model, *_):  # pragma: no cover - simple stub
        return model


def test_preference_reranker_missing_torch(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForSequenceClassification=object,
            AutoTokenizer=object,
        ),
    )
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=object))
    monkeypatch.delitem(sys.modules, "memory_system.adapter.reranker", raising=False)
    with pytest.raises(ModuleNotFoundError, match="torch is required"):
        importlib.import_module("memory_system.adapter.reranker")


def test_preference_reranker_with_torch(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=DummyTokenizer,
            AutoModelForSequenceClassification=DummyModel,
        ),
    )
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=DummyPeft))
    monkeypatch.delitem(sys.modules, "memory_system.adapter.reranker", raising=False)
    rr = importlib.import_module("memory_system.adapter.reranker")
    reranker = rr.PreferenceReranker()
    assert reranker.score("q", "d") == 0.0
