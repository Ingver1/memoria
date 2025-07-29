# Unified Memory System (UMS) 🧠

*Scalable, encrypted, and fully-tested vector-memory backend for LLM agents.*

---

## 🚀 Why UMS?

UMS is a plug-and-play semantic memory service for LLM agents. It’s small, fast, secure, and production-ready. If you want a memory backend that “just works” and is easy to audit, UMS is for you.

---

## ✨ Highlights (July 2025)

| Feature               | UMS (v1.0)           | MemOS (Preview)         |
|-----------------------|----------------------|-------------------------|
| Install size          | ~180 MB Docker       | 2+ GB multi-service     |
| Storage backend       | SQLite + FAISS       | Custom “MemCube”        |
| Encryption-at-rest    | ✅ SQLCipher         | ❌ Coming Q4 2025      |
| Test coverage         | 100% unit/integration| Unknown                 |
| Deployment            | pip, Docker, serverless | k8s Operator (beta)  |
| Hardware requirements | 1 CPU / 1 GB RAM     | 8 CPUs / 16 GB RAM      |
| License               | Apache-2.0           | Clause-7 research       |
| Status                | Stable / Prod        | Preview / Research      |

<sub>Sources: arXiv 2507.03724, VentureBeat 08-Jul-2025, MemTensor/MemOS release notes.</sub>

---

## 🔑 Key Features

- **Async FastAPI + FAISS HNSW**: blazing fast semantic search
- **SQLCipher encryption**: secure at rest
- **API-token auth & rate limits**: secure in transit
- **Prometheus metrics & health checks**: easy monitoring
- **100% test coverage**: unit, property, fuzz, performance
- **Pluggable key management**: local JSON keyring with a stub for AWS KMS
- **Simple install**: `pip install ai-memory` or Docker one-liner

---

## 🏁 Quick Start

```bash
pip install ai-memory        # production
pip install ai-memory[dev]   # +tests & tooling
pip install -e .[dev]       # editable for local dev
uvicorn memory_system.api.app:create_app --reload
```

**Minimal client:**
```python
from memory_system import EnhancedMemoryStore, UnifiedSettings
import numpy as np, asyncio

settings = UnifiedSettings.for_testing()
store = EnhancedMemoryStore(settings)

async def demo():
    await store.add_memory(text="Hello world!", embedding=np.random.rand(settings.model.vector_dim).tolist())
    hits = await store.semantic_search(vector=np.random.rand(settings.model.vector_dim).tolist(), k=3)
    print(hits)

asyncio.run(demo())
```

---

## 🧪 Testing

Tests rely on the bundled plugins under `memoria/_pytest_plugins/`.  
`sitecustomize.py` sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` so that pytest only
loads these stub plugins.  `conftest.py` loads them via
`pytest_plugins = ("memoria._pytest_plugins",)`. Always run tests from the
repository root so `conftest.py` can discover the plugins:

```bash
pytest -q -m "not perf"
```

### Matrix

| Suite      | Command                                      | Avg time |
|------------|----------------------------------------------|----------|
| Smoke      | pytest -q -m "not perf"                      | 8 s      |
| Property   | pytest -q -m property                        | 8 s      |
| Perf/Bench | pytest -q tests/test_performance.py --benchmark-only | 30 s |
| Load       | locust -f load_tests/locustfile.py            | user     |
| API fuzz   | pytest tests/test_api_fuzz.py                 | 4 s      |

---

## ⚙️ Configuration

| Env var                  | Default                | Description           |
|--------------------------|------------------------|-----------------------|
| AI_DATABASE__URL         | sqlite:///./data/memory.db | DB path / DSN     |
| AI_SECURITY__ENCRYPT_AT_REST | false              | Enable SQLCipher      |
| AI_MODEL__VECTOR_DIM     | 384                    | Embedding dimension   |
| AI_PERF__MAX_WORKERS     | 4                      | Async workers         |
| AI_MONITORING__ENABLE_METRICS | false             | Expose /metrics       |

Copy `.env.example` → `.env` and tweak values.

---

## 🛡 Security Model

- **Disk**: AES-256-GCM via SQLCipher (pysqlcipher3)
- **Transit**: HTTPS/TLS recommended; API-token checked on every request
- **Fault tolerance**: If FAISS index is missing/corrupted, `/health` returns 503 and write paths are blocked until recovery

---

## 🗺 Roadmap

1. Hot-swap backends: Qdrant & DuckDB extensions
2. Hierarchical summarisation: automatic memory compaction
3. Streaming ingestion: SSE / WebSocket pipeline

Pull Requests welcome!

---

## 📜 License

Apache License 2.0 – free for commercial & research use.

© 2025 Evgeny Leshchenko, with assistance from ChatGPT & Claude.
