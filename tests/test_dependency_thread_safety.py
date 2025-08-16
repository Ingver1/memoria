from __future__ import annotations

import threading
from types import SimpleNamespace

from memory_system.utils import dependencies


def _run_concurrent(require_func, attr_name, module_name, monkeypatch, extra_attrs=None) -> None:
    counter = 0
    counter_lock = threading.Lock()

    def fake_import(name: str):
        assert name == module_name
        nonlocal counter
        with counter_lock:
            counter += 1
        module = SimpleNamespace()
        if extra_attrs:
            for key, value in extra_attrs.items():
                setattr(module, key, value)
        return module

    monkeypatch.setattr(dependencies, "import_module", fake_import)
    setattr(dependencies, attr_name, None)

    start = threading.Event()

    def worker() -> None:
        start.wait()
        require_func()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    assert counter == 1


def test_require_dependencies_thread_safe(monkeypatch) -> None:
    _run_concurrent(dependencies.require_faiss, "_faiss", "faiss", monkeypatch)
    _run_concurrent(dependencies.require_httpx, "_httpx", "httpx", monkeypatch)
    _run_concurrent(
        dependencies.require_sentence_transformers,
        "_st",
        "memory_system._vendor.sentence_transformers",
        monkeypatch,
        {"SentenceTransformer": object},
    )
