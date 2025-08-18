# Unified Memory System (UMS) 🧠

*Scalable, encrypted, and fully-tested vector-memory backend for LLM agents.*

> **Note:** This project is a work in progress and is not yet functional.

---

### Security
Found a vulnerability? See [SECURITY.md](./SECURITY.md).

## 🚀 Why UMS?

UMS is a plug-and-play semantic memory service for LLM agents. It’s small, fast, secure, and production-ready. If you want a memory backend that “just works” and is easy to audit, UMS is for you.

---

## 🔑 Key Features

- **Async FastAPI + FAISS HNSW or Qdrant**: blazing fast semantic search
- **SQLCipher encryption**: secure at rest
- **API-token auth & rate limits**: secure in transit
- **Prometheus metrics & health checks**: easy monitoring
- **100% test coverage**: unit, property, fuzz, performance
- **Pluggable key management**: local JSON keyring, Vault or AWS KMS
- **Hierarchical summarisation**: clusters similar memories into higher `level`s and marks singletons as final ([docs](docs/architecture.md#4-hierarchical-summarisation)) 
 - **Optional cross-encoder reranker**: `pip install sentence-transformers` and enable with `AI_RANKING__USE_CROSS_ENCODER=1` or `use_cross_encoder=True`. The default models are `bge-m3` for embeddings and `cross-encoder/ms-marco-MiniLM-L6-v2` for reranking, but any Sentence Transformers bi-encoder / cross-encoder pair can be used.
- **Simple install**: `pip install ai-memory[lite]` or Docker one-liner
- **Dual-channel retrieval**: personal memories are searched locally while global knowledge is fetched remotely

---

## 🏁 Quick Start

```bash
pip install ai-memory[lite]          # minimal core
pip install ai-memory[api,faiss]     # API server with FAISS
pip install ai-memory[full]          # all optional features
pip install -e .[dev]                # editable for local dev
uvicorn memory_system.api.app:create_app --factory --reload
```

### Secure preset

Start the server with SQLCipher and managed keys:

```bash
AI_ENV=production AI_SECURITY__KMS_BACKEND=local python -m scripts.secure_entry
```

`create_app` uses FastAPI's lifespan to initialise an `EnhancedMemoryStore`, which
is available on `app.state.memory_store` during requests.

**Minimal client:**
```python
from memory_system import EnhancedMemoryStore, UnifiedSettings
import numpy as np, asyncio

settings = UnifiedSettings.for_testing()
settings.summary_strategy = "concat"  # simpler summaries
settings.dynamics.decay_rate = 15.0    # faster forgetting

async def demo():
    async with EnhancedMemoryStore(settings) as store:
        await store.add_memory(
            text="Hello world!",
            embedding=np.random.rand(settings.model.vector_dim).tolist(),
            role="user",
            tags=["greeting"],
        )
        hits = await store.semantic_search(
            vector=np.random.rand(settings.model.vector_dim).tolist(),
            k=3,
            metadata_filter={"role": "user"},
            level=0,
            return_distance=True,
        )
        print(hits)

asyncio.run(demo())
```

### Dual-channel design

UMS distinguishes between two memory channels:

- **Global** memories live in the remote service and can be shared across agents.
- **Personal** memories are kept in a local vector store and queries against them never leave your device.

This setup lets applications blend shared knowledge with private context without compromising privacy.

## 🛡️ What we collect and why

UMS keeps privacy front and centre. Two flags in `SecurityConfig` control all
data sharing:

- `privacy_mode` — `strict` (default) disables any operation that could leak
  identifiers, while `standard` allows features like `/memory/mark_access`.
- `telemetry_level` — `aggregate` (default) sends only anonymised metrics;
  `none` disables telemetry entirely.

Even when telemetry is enabled, metrics are aggregated and exclude memory IDs or
other personally identifiable information.

## Dependencies

The base install is intentionally slim. Optional features are grouped into extras:

| Extra      | Enables                                  | Test markers                     |
|------------|------------------------------------------|----------------------------------|
| `lite`     | Core memory store                        | –                                |
| `api`      | FastAPI server & telemetry               | `needs_fastapi`, `needs_httpx`   |
| `faiss`    | Local FAISS vector index                 | `needs_faiss`                    |
| `qdrant`   | Remote Qdrant vector index               | –                                |
| `numpy`    | Vector operations                        | `needs_numpy`                    |
| `cli`      | Typer/Rich command line                  | `needs_httpx`                    |
| `http`     | HTTP client for API tests                | `needs_httpx`                    |
| `security` | Encryption & SQLCipher                   | `needs_crypto`                   |
| `metrics`  | Prometheus instrumentation               | `needs_fastapi`, `needs_httpx`   |
| `bench`    | Benchmark tooling                        | `perf`, `bench`                  |
| `full`     | All of the above                         | `needs_fastapi`, `needs_httpx`,<br>`needs_numpy`, `needs_crypto`, `perf` |

Development tooling lives under the `dev` extra. For example:

```bash
pip install -e .[dev]       # install test tooling
pip install ai-memory[bench,cli] # add benchmarking support
```

## 📊 Benchmarking

UMS includes small utilities to exercise performance and retention.

### Synthetic dialogue recall

Generate a deterministic conversation and report recall and forgetting rate:

```bash
mem-bench dialogue --turns 50
```

### Search throughput

Measure search performance across retrieval modes:

```bash
python -m scripts.bench_search
```

The module deterministically builds 1k/10k dummy corpora and reports
median, 95th‑percentile and QPS for vector vs. hybrid search with optional
MMR (λ=0.7) and cross‑encoder reranking (e.g. `bge-m3` +
`cross-encoder/ms-marco-MiniLM-L6-v2`). Cross reranking only reorders the top
20 candidates.

## 🤖 Using with LLM agents

Initialize the memory store with embedding dimensions that match your model. For
single‑modal models this is a single `vector_dim`; for multi‑modal models supply
`vector_dims`.

```python
from memory_system import EnhancedMemoryStore, UnifiedSettings, ModelConfig
import numpy as np, asyncio

# single modality
settings = UnifiedSettings(model=ModelConfig(vector_dim=384))

# multi-modal example:
# settings = UnifiedSettings(
#     model=ModelConfig(modalities=["text", "image"], vector_dims={"text": 384, "image": 3})
# )

async def agent_demo():
    async with EnhancedMemoryStore(settings) as store:
        await store.add_memory(
            text="hello",
            embedding=np.random.rand(settings.model.vector_dim).tolist(),
        )
        hits = await store.semantic_search(
            vector=np.random.rand(settings.model.vector_dim).tolist(),
            k=3,
        )
        print(hits)

asyncio.run(agent_demo())
```

> **Note**: When using the bundled `embedder` module to generate embeddings
> locally, call `embedder.close_models()` when your service shuts down to free
> model resources.

Expose these FastAPI endpoints to connect your agent over HTTP:

- `POST /memory/add`
- `POST /memory/search`
- `POST /memory/consolidate`
- `POST /memory/forget`

### Memory maintenance

Both maintenance operations are accessible from the Python client:

```python
async def maintenance_demo():
    await store.consolidate_memories(threshold=0.9)
    removed = await store.forget_memories(min_total=100, retain_fraction=0.8)
    print(f"Forgot {removed} memories")

asyncio.run(maintenance_demo())
```

These mirror the `/memory/consolidate` and `/memory/forget` HTTP endpoints.

---

## 🧪 Testing

Run the test suite from the repository root:

```bash
PYTHONPATH=scripts pytest -q -m "not perf"
```

### Matrix

| Suite      | Command                                      | Avg time |
|------------|----------------------------------------------|----------|
| Smoke      | `PYTHONPATH=scripts pytest -q -m "not perf"` | 8 s      |
| Property   | `PYTHONPATH=scripts pytest -q -m property`   | 8 s      |
| Perf/Bench | `PYTHONPATH=scripts pytest -q tests/test_performance.py --benchmark-only` | 30 s |
| Load       | locust -f load_tests/locustfile.py            | user     |
| API fuzz   | `PYTHONPATH=scripts pytest tests/test_api_fuzz.py` | 4 s      |

### Troubleshooting tests

If running the test suite fails because pytest auto-loads unrelated
plugins from your environment, disable plugin discovery:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=scripts pytest -q -m "not perf"
```

With auto-loading disabled, required plugins need to be specified
explicitly via `-p` (for example, CI loads coverage support with
`-p pytest_cov`).

---

## ⚙️ Configuration

| Env var                  | Default                | Description           |
|--------------------------|------------------------|-----------------------|
| AI_DATABASE__URL         | sqlite:///./data/memory.db | DB path / DSN (e.g. `libsql://<url>?auth_token=...`) |
| AI_DATABASE__BACKEND     | faiss                  | Vector store backend  |
| AI_DATABASE__QDRANT_URL  | http://localhost:6333  | Qdrant endpoint       |
| AI_DATABASE__QDRANT_COLLECTION | memory           | Qdrant collection     |
| AI_DATABASE__WAL_CHECKPOINT_INTERVAL | 60.0        | Seconds between WAL checkpoints |
| AI_DATABASE__WAL_CHECKPOINT_WRITES   | 1000        | Writes before forcing a checkpoint |
| AI_SECURITY__ENCRYPT_AT_REST | false              | Enable SQLCipher      |
| AI_MODEL__VECTOR_DIM     | 384                    | Embedding dimension   |
| AI_PERF__MAX_WORKERS     | 4                      | Async workers         |
| AI_MONITORING__ENABLE_METRICS | false             | Expose /metrics       |
| AI_SUMMARY_STRATEGY      | head2tail              | Hierarchical summary strategy |
| AI_DYNAMICS__DECAY_RATE  | 30.0                   | Intensity decay rate (days) |
| AI_DYNAMICS__DECAY_WEIGHTS__IMPORTANCE | 0.4 | Decay weight for importance |
| AI_DYNAMICS__DECAY_WEIGHTS__EMOTIONAL_INTENSITY | 0.3 | Decay weight for emotional intensity |
| AI_DYNAMICS__DECAY_WEIGHTS__VALENCE | 0.3 | Decay weight for valence |

Copy `.env.example` → `.env` and tweak values. For step-by-step instructions on
running a Qdrant vector store and wiring it to the memory system, see
[docs/qdrant_setup.md](docs/qdrant_setup.md).

Remote libSQL backends use DSNs of the form `libsql://<url>?auth_token=...`.
When such a DSN is configured, WAL files are managed server-side and the local
`AI_DATABASE__WAL_*` settings are ignored.

### FAISS index options

| Setting | Default | Description |
|---------|---------|-------------|
| `faiss.index_type` | `HNSW` | FAISS index type (`HNSW`, `IVF`, `IVFPQ`, `HNSWPQ`, `OPQ`) |
| `faiss.use_gpu` | `false` | Move index to GPU if GPUs are available |
| `faiss.M` | `32` | HNSW: number of bi-directional links |
| `faiss.ef_construction` | `200` | HNSW: graph construction depth |
| `faiss.ef_search` | `100` | HNSW: search depth (`efSearch`) or IVF `nprobe` |
| `faiss.nlist` | `100` | Number of clusters for IVF-based indices |
| `faiss.nprobe` | `8` | Search probes for IVF-based indices |
| `faiss.pq_m` | `16` | PQ: number of sub-vector segments (`M`) |
| `faiss.pq_bits` | `8` | PQ: bits per sub-vector |
| `faiss.autotune` | `false` | Automatically tune HNSW parameters on first insert |
| `faiss.dataset_size` | – | Expected dataset size for heuristic tuning |
| `faiss.nlist_scale` | `4.0` | Multiplier used when deriving `nlist` from dataset size |
| `faiss.ef_search_scale` | `0.5` | Multiplier used when deriving `ef_search` |
| `faiss.ef_construction_scale` | `2.0` | Multiplier used when deriving `ef_construction` |
| `faiss.pq_m_div` | `24` | Divides vector dimension when deriving `pq_m` |

## 🔐 Encryption Options

UMS offers two ways to protect stored data. The `text` column can be encrypted
individually, or the entire SQLite database file can be secured.

### Fernet field encryption

This feature requires the [`cryptography`](https://pypi.org/project/cryptography/)
package. Only the `text` column is encrypted before being written to disk using a
[Fernet](https://cryptography.io/en/latest/fernet/) key. Supply your own key via
[`AI_SECURITY__ENCRYPTION_KEY`](memory_system/settings/security.py#L41).

```bash
AI_DATABASE__URL=sqlite:///./data/memory.db
AI_SECURITY__ENCRYPTION_KEY=<base64-key>
```

### SQLCipher full-database encryption

Set [`AI_SECURITY__ENCRYPT_AT_REST`](memory_system/settings/security.py#L40) to
`true` to encrypt the entire SQLite file with SQLCipher. This switches the DSN
to `sqlite+sqlcipher` and reuses the
[`AI_SECURITY__ENCRYPTION_KEY`](memory_system/settings/security.py#L41).

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
AI_SECURITY__KMS_BACKEND=aws
AI_SECURITY__KMS_KEY_ID=arn:aws:kms:...:key/123
AI_SECURITY__KMS_PARAMS='{"region_name":"us-east-1"}'
# AWS credentials can be provided via environment variables such as
# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
```

### Fernet key validation and rotation

- **Validate key format** – Fernet keys must be URL‑safe base64 strings that
  decode to exactly 32 bytes. Use
  `memory_system.utils.security.validate_fernet_key` or attempt to construct a
  `Fernet` object to confirm the key is valid.
- **Rotate regularly** – generate a fresh key with
  `cryptography.fernet.Fernet.generate_key()` on a schedule (for example every
  30 days) and update your keyring or environment variable.
- **Maintain a key pool** – keep prior keys available for decryption while new
  writes use the latest key. Retire old keys only after all data encrypted with
  them has been re‑encrypted or expired.

---

## 🛡 Security Model

- **Disk**: AES-256-GCM via SQLCipher (sqlcipher3-binary)
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
