# memoria — System Architecture

<details>
<summary>Mermaid diagram — end-to-end flow (click to expand)</summary>

```mermaid
graph TD
  subgraph Ingestion & Vectorisation
    A[Incoming content (text, PDF, webhook)] -->|clean + chunk| B[Embedder (OpenAI, Instructor)]
    B -->|vector| C((Vector))
  end

  %% Storage layer
  C --> D[FAISS HNSW Index]
  D -->|top-k IDs| E[Rank & Filter]
  E --> F[(SQLite JSON1 metadata)]

  %% Retrieval
  F -->|payload| G[[Client / LLM prompt builder]]
```

</details>

---

## 1. Data Ingestion 🚚
| Step         | Module                                 | Notes                                               |
|--------------|----------------------------------------|-----------------------------------------------------|
| Clean & Chunk| memory_system.ingest.cleaner           | Language-agnostic, ~2k-token windows                |
| Embed        | memory_system.vector.embedder          | OpenAI by default, swap in Instructor/HuggingFace   |
| Persist      | memory_system.core.vector_store.AsyncFaissHNSWStore | Raw vector + PK, metadata separate |

All CPU-bound FAISS calls run in the default thread pool for snappy event loop.

---

## 2. Storage & Index 📦
| Layer     | Implementation         | Why                                               |
|-----------|------------------------|---------------------------------------------------|
| Vector    | FAISS HNSW (m=32)      | Good recall/latency for ≤5M vectors on single box |
| Metadata  | SQLite JSON1           | Lightweight, enables rich queries                 |
| Back-ups  | Background replicate() | Compressed snapshot every N minutes               |

**Compaction:**
- Removes tombstoned vectors
- Shrinks .index blob with FAISS in-place merge
- VACUUMs SQLite and ANALYZEs indices

---

## 3. Retrieval 🔎
1. Semantic search: vector → top-k IDs
2. Re-rank & filter: JSON1 predicates + optional LLM re-ranking
3. Return: Memory dataclass (text, score, metadata)

---

## 4. API Surface 🌐
- REST (`/memory/add`, `/search`)
- Events (`/sse`) — memory-updated stream
- CLI (`ai-mem add/search`)
- gRPC (optional) — high-throughput internal bus

All endpoints use FastAPI + OpenTelemetryMiddleware for full span tracing.

---

## 5. Observability 👀
| Component | Tech                                       |
|-----------|--------------------------------------------|
| Metrics   | prometheus_client (latency, faiss_queries) |
| Tracing   | OTEL SDK → OTLP HTTP → Tempo/Jaeger        |
| Logs      | logging.yaml → plaintext/JSON (LOG_JSON=1) |

---

## 6. Deployment 🚀
- Single-container image (`python:3.12-slim-bookworm`, ≈120 MB)
- Health probes:
  - `/health/live` → liveness (always 200 unless Uvicorn crashes)
  - `/health/ready` → store ping + FAISS integrity
- Docker Compose example spins API + OTEL collector + Grafana/Loki in three commands

---

## Extending
- Swap FAISS for Qdrant: implement `AbstractVectorStore`
- Plug PostgreSQL metadata backend: use psycopg3 + JSONB
- Add web-crawler ingest: subclass `BaseIngestor`

---

© 2025 Ingver / AI-memory Team — Licensed under Apache-2.0
