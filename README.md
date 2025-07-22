# Unified Memory System (UMS) 🧠

*Scalable, encrypted, and fully-tested vector-memory backend for LLM agents.*

---

## ✨ What’s New (July 2025)

> **MemOS vs UMS**  
> A flurry of excitement followed the public preview of **MemOS 1.0 “Stellar”** – the self-described *“memory operating system”* for large language models.<br>
> UMS takes a different stance: instead of a full OS-level scheduler, it delivers a slim, auditable service that you can embed today.

|                        | **UMS (v1.0)** | **MemOS (1.0 Preview\*)** |
|------------------------|----------------|---------------------------|
| Install size           | ~180 MB Docker | 2+ GB multi-service stack |
| Storage backend        | SQLite + FAISS | Custom “MemCube” shards   |
| Encryption-at-rest     | ✅ SQLCipher   | ❌ roadmap Q4 2025        |
| Test coverage (CI)     | 100 % unit/integration | unknown |
| Deployment            | `pip`, Docker, serverless | k8s Operator (beta) |
| Hardware requirements  | 1 CPU / 1 GB RAM | 8 CPUs / 16 GB RAM (rec.) |
| License                | Apache-2.0     | Clause-7 research license |
| Status                 | Stable / Prod  | Preview / Research        |

<sub>\*Sources: arXiv 2507.03724, VentureBeat 08-Jul-2025, MemTensor/MemOS release notes.</sub>

UMS = **small core, big confidence**.  
If you need a plug-and-play semantic memory today, keep reading.

---

## 🔑 Key Features

| Area | Highlights |
|------|-----------|
| **Architecture** | Async FastAPI + FAISS HNSW with dynamic `ef_search` tuning |
| **Security** | SQLCipher encryption, API-token auth, rate limits |
| **Observability** | `/metrics` (Prometheus), `/health` (deep), structured logs |
| **Quality** | 100 % unit + property-based + fuzz + performance tests |
| **CI/CD** | Fast smoke suite on each push, full perf suite nightly |
| **Ease of use** | `pip install memoria1` or one-liner Docker run |

---

## 🚀 Quick Start

```bash
pip install memoria1        # production
pip install memoria1[dev]   # +tests & tooling
pip install -e .[dev]   # editable install for local development
# or, in a cloned repo, pip install -r requirements_dev.txt for perf tests
uvicorn memory_system.api.app:create_app --reload

Minimal client

from memoria1 import EnhancedMemoryStore, UnifiedSettings
import numpy as np, asyncio

settings = UnifiedSettings.for_testing()
store = EnhancedMemoryStore(settings)

async def demo():
    await store.add_memory(text="Hello world!", embedding=np.random.rand(settings.model.vector_dim).tolist())
    hits = await store.semantic_search(vector=np.random.rand(settings.model.vector_dim).tolist(), k=3)
    print(hits)

asyncio.run(demo())

### Offline embedding

Create `embedder.py` and add:
```python
from sentence_transformers import SentenceTransformer
import numpy as np
import inspect

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str | list[str]) -> np.ndarray:
    if isinstance(text, str):
        text = [text]
    if "normalize_embeddings" in inspect.signature(_model.encode).parameters:
        vecs = _model.encode(text, normalize_embeddings=True)
    else:
        vecs = _model.encode(text)
        vecs = np.asarray(vecs, dtype=np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs[0] if len(vecs) == 1 else vecs
```

Use it with Memoria:
```python
from memoria1 import EnhancedMemoryStore, UnifiedSettings
from embedder import embed

settings = UnifiedSettings.for_testing()
settings.embed_func = embed
store = EnhancedMemoryStore(settings)
```

Now embeddings are generated locally without any external API.         

---

🧪 Testing Matrix

Suite   Command Avg time

Smoke   pytest -q -m "not perf" 8 s
Property  pytest -q -m property  8 s
Perf / Bench    pytest -q tests/test_performance.py --benchmark-only       30 s
Load (Locust)   locust -f load_tests/locustfile.py      user-defined
API fuzz        pytest tests/test_api_fuzz.py   4 s


The smoke suite runs on every push; perf + load run nightly via GitHub Actions cron.


---

⚙️ Configuration

Env var	Default	Description

AI_DATABASE__URL	sqlite:///./data/memory.db	DB path / DSN
AI_SECURITY__ENCRYPT_AT_REST	false	Enable SQLCipher
AI_MODEL__VECTOR_DIM	384	Embedding dimension
AI_PERF__MAX_WORKERS	4	Async workers
AI_MONITORING__ENABLE_METRICS	false	Expose /metrics


Copy .env.example → .env and tweak values.


---

🛡  Security Model

Disk – AES-256-GCM via SQLCipher (pysqlcipher3).

Transit – HTTPS/TLS recommended; API-token checked on every request.

Fault tolerance – if FAISS index is missing/corrupted, /health returns 503 and write paths are blocked until recovery.



---

🗺  Roadmap

1. Hot-swap backends – Qdrant & DuckDB extensions.


2. Hierarchical summarisation – automatic memory compaction.


3. Streaming ingestion – SSE / WebSocket pipeline.



Pull Requests welcome!


---

📜 License

Apache License 2.0 – free for commercial & research use.

© 2025 Evgeny Leshchenko, with assistance from ChatGPT & Claude.
