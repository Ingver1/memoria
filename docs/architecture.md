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
| Clean & Chunk| memory_system.utils.security.EnhancedPIIFilter | Language-agnostic, ~2k-token windows (future `memory_system.ingest` module) |
| Embed        | memory_system.core.embedding.EmbeddingService  | OpenAI by default, swap in Instructor/HuggingFace   |
| Persist      | memory_system.core.vector_store.AsyncFaissHNSWStore / `QdrantVectorStore` | Raw vector + PK, metadata separate |

All CPU-bound FAISS calls run in the default thread pool for snappy event loop.

---

## 2. Storage & Index 📦
| Layer     | Implementation         | Why                                               |
|-----------|------------------------|---------------------------------------------------|
| Vector    | FAISS HNSW (m=32) / Qdrant | Good recall/latency or managed service        |
| Metadata  | SQLite JSON1           | Lightweight, enables rich queries                 |
| Back-ups  | Background replicate() | Compressed snapshot every N minutes               |

**Persistence:**
- The FAISS index is saved to disk after each memory is added and reloaded on start if the index file exists, ensuring vector search state survives restarts. Qdrant persists remotely.

**Compaction:**
- Removes tombstoned vectors
- Shrinks .index blob with FAISS in-place merge
- VACUUMs SQLite and ANALYZEs indices

**Adaptive recall tuning:**
- Background task issues control queries and measures recall
- `ef_search` increases when recall < 0.9 SLA, decreases when safely above

---

## 3. Retrieval 🔎
1. Semantic search: vector → top-k IDs
2. Re-rank & filter: JSON1 predicates + optional LLM re-ranking
3. Return: Memory dataclass (text, score, metadata)

---

## 4. Hierarchical summarisation 🧩
Memories can be compacted into higher-level summaries to keep the store
manageable over time.

- Each memory carries a numeric `level`; raw inserts start at `0`.
- The `HierarchicalSummarizer` groups memories on a source level by cosine
  similarity and writes a new summary to `level + 1`.
- Singleton memories that do not meet the clustering threshold are tagged
  with `metadata['final'] = True` so they are skipped on future passes.
- Summaries keep track of their inputs via `metadata['source_ids']` and
  `metadata['cluster_size']`.

---

## 5. API Surface 🌐
- REST (`/memory/add`, `/search`)
- Events (`/sse`) — memory-updated stream
- CLI (`ai-mem add/search`)
- gRPC (optional) — high-throughput internal bus

All endpoints use FastAPI + OpenTelemetryMiddleware for full span tracing.

---

## 6. Observability 👀
| Component | Tech                                       |
|-----------|--------------------------------------------|
| Metrics   | prometheus_client (latency, faiss_queries) |
| Tracing   | OTEL SDK → OTLP HTTP → Tempo/Jaeger       |
| Logs      | logging.yaml → plaintext/JSON (LOG_JSON=1) |

Additional metrics:
- `ums_ann_query_latency_seconds` — histogram of FAISS search latency
- `ums_ann_index_size` — gauge of current index size

---

## 7. Deployment 🚀
- Single-container image (`python:3.12-slim-bookworm`, ≈120 MB)
- Health probes:
  - `/health/live` → liveness (always 200 unless Uvicorn crashes)
  - `/health/ready` → store ping + FAISS integrity
- Docker Compose example spins API + OTEL collector + Grafana/Loki in three commands

---

## Extending
- Swap vector backend: implement `AbstractVectorStore` (FAISS and Qdrant included)
- Plug PostgreSQL metadata backend: use psycopg3 + JSONB
- Add web-crawler ingest: subclass `BaseIngestor`

---

© 2025 Ingver / AI-memory Team — Licensed under Apache-2.0
