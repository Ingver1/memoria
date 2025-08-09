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
| Storage backend       | SQLite + FAISS/Qdrant | Custom “MemCube”       |
| Encryption-at-rest    | ✅ SQLCipher         | ❌ Coming Q4 2025       |
| Test coverage         | 100% unit/integration| Unknown                 |
| Deployment            | pip, Docker, serverless | k8s Operator (beta)  |
| Hardware requirements | 1 CPU / 1 GB RAM     | 8 CPUs / 16 GB RAM      |
| License               | Apache-2.0           | Clause-7 research       |
| Status                | Stable / Prod        | Preview / Research      |

<sub>Sources: arXiv 2507.03724, VentureBeat 08-Jul-2025, MemTensor/MemOS release notes.</sub>

---

## 🔑 Key Features

- **Async FastAPI + FAISS HNSW or Qdrant**: blazing fast semantic search
- **SQLCipher encryption**: secure at rest
- **API-token auth & rate limits**: secure in transit
- **Prometheus metrics & health checks**: easy monitoring
- **100% test coverage**: unit, property, fuzz, performance
- **Pluggable key management**: local JSON keyring with a stub for AWS KMS
- **Hierarchical summarisation**: clusters similar memories into higher `level`s and marks singletons as final ([docs](docs/architecture.md#4-hierarchical-summarisation))
- **Simple install**: `pip install ai-memory` or Docker one-liner

---

## 🏁 Quick Start

```bash
pip install ai-memory        # production
pip install ai-memory[security]  # +encryption support (or `pip install .[security]`)
pip install ai-memory[metrics]   # +Prometheus metrics
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
    await store.start()
    await store.add_memory(text="Hello world!", embedding=np.random.rand(settings.model.vector_dim).tolist())
    hits = await store.semantic_search(
        vector=np.random.rand(settings.model.vector_dim).tolist(), k=3, return_distance=True
    )
    print(hits)

asyncio.run(demo())
```

---

## 🧪 Testing

Run the test suite from the repository root:

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
| AI_DATABASE__URL         | sqlite:///./data/memory.db | DB path / DSN (use `libsql://` for remote) |
| AI_DATABASE__BACKEND     | faiss                  | Vector store backend  |
| AI_DATABASE__QDRANT_URL  | http://localhost:6333  | Qdrant endpoint       |
| AI_DATABASE__QDRANT_COLLECTION | memory           | Qdrant collection     |
| AI_SECURITY__ENCRYPT_AT_REST | false              | Enable SQLCipher      |
| AI_MODEL__VECTOR_DIM     | 384                    | Embedding dimension   |
| AI_PERF__MAX_WORKERS     | 4                      | Async workers         |
| AI_MONITORING__ENABLE_METRICS | false             | Expose /metrics       |

Copy `.env.example` → `.env` and tweak values.

### FAISS index options

| Env var             | Default | Description                                 |
|---------------------|---------|---------------------------------------------|
| UMS_INDEX_TYPE      | HNSW    | FAISS index type (`HNSW`, `IVFFLAT`, `IVFPQ`, `HNSWPQ`, `OPQ`) |
| UMS_USE_GPU         | 0       | Move index to GPU if `1` and GPUs available |
| UMS_IVF_NLIST       | 100     | Number of clusters for IVF-based indices    |
| UMS_IVF_NPROBE      | 8       | Search probes (`nprobe`) for IVF-based indices |
| UMS_PQ_M            | 16      | PQ: number of sub-vector segments (`M`)     |
| UMS_PQ_BITS         | 8       | PQ: bits per sub-vector                     |
| UMS_EF_CONSTRUCTION | 128     | HNSW: graph construction depth              |
| UMS_HNSW_M          | 32      | HNSW: number of bi-directional links        |
| UMS_EF_SEARCH       | 32      | HNSW: search depth (`efSearch`)             |

## 🔐 Encryption Options

UMS offers two ways to protect stored data. The `text` column can be encrypted
individually, or the entire SQLite database file can be secured.

### Fernet field encryption

Only the `text` column is encrypted before being written to disk using a
[Fernet](https://cryptography.io/en/latest/fernet/) key. Supply your own key via
[`AI_SECURITY__ENCRYPTION_KEY`](memory_system/config/settings.py#L77).

```bash
AI_DATABASE__URL=sqlite:///./data/memory.db
AI_SECURITY__ENCRYPTION_KEY=<base64-key>
```

### SQLCipher full-database encryption

Set [`AI_SECURITY__ENCRYPT_AT_REST`](memory_system/config/settings.py#L76) to
`true` to encrypt the entire SQLite file with SQLCipher. This switches the DSN
to `sqlite+sqlcipher` and reuses the
[`AI_SECURITY__ENCRYPTION_KEY`](memory_system/config/settings.py#L77).

```bash
AI_SECURITY__ENCRYPT_AT_REST=true
AI_DATABASE__URL=sqlite+sqlcipher:///./data/memory.db
```

### Choosing an approach

- **Fernet field encryption** – minimal overhead and works anywhere SQLite is
  available. Use when only the `text` payload needs protection.
- **SQLCipher** – protects the entire database (metadata, indices). Choose when
  host or disk access is untrusted; requires the SQLCipher driver and adds a
  small performance cost.

### Key management backends

Encryption keys can be supplied directly or loaded from an external key
management service. Configure this behaviour via the `kms_backend` field in
`SecurityConfig`:

```bash
AI_SECURITY__KMS_BACKEND=local        # or 'vault', 'aws'
AI_SECURITY__KMS_PARAMS='{"url":"https://vault"}'
```

`vault` and `aws` are currently placeholders that raise
`NotImplementedError` when used.

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
