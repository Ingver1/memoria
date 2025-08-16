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

## Key Components 🧩
- **Embedder** – transforms cleaned text chunks into high‑dimensional vectors using pluggable models such as OpenAI or Instructor.
- **Vector Store** – FAISS HNSW index or Qdrant backend providing approximate nearest‑neighbour search.
- **Metadata Store** – SQLite JSON1 database holding rich metadata and relational queries.
- **API Layer** – FastAPI endpoints and optional gRPC interface exposing add/search operations.
- **Background Workers** – perform consolidation, backups and summarisation tasks.

## Data Flow 🔄
1. Content enters via REST, CLI or webhook.
2. `EnhancedPIIFilter` cleans and chunks text.
3. `EmbeddingService` generates vectors.
4. Vectors persist to FAISS/Qdrant while metadata is written to SQLite.
5. Retrieval queries the vector index, joins metadata and returns results to clients.

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

### Memory encryption & layout
Only the `text` field of each memory is encrypted at rest using
[Fernet](https://cryptography.io/en/latest/fernet/) from the
[`cryptography`](https://pypi.org/project/cryptography/) package. FAISS vectors
remain plaintext to keep similarity search fast, while metadata is stored
separately in SQLite.

```json
{
  "text": "gAAAAABk9...==",  
  "embedding": [0.12, -0.05, 0.32],
  "metadata": {"source": "chat", "user": "alice"}
}
```

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

See the [Memory maintenance](../README.md#memory-maintenance) section for code
examples on consolidation and forgetting.

---

## 5. API Surface 🌐
- REST (`/memory/add`, `/search`)
- Events (`/sse`) — memory-updated stream
- CLI (`ai-mem add/search`)
- gRPC (optional) — high-throughput internal bus

All write endpoints honour an `Idempotency-Key` header allowing safe retries.
The CLI exposes this through the `--idempotency-key` option. All endpoints use
FastAPI + OpenTelemetryMiddleware for full span tracing.

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
- `ums_consolidation_latency_seconds` — histogram of consolidation latency
- `ums_forget_latency_seconds` — histogram of forgetting latency

- `ums_memories_created_total` — counter of created memories by modality
- `ums_memories_deleted_total` — counter of deleted memories by modality

---

## Threat Model ⚠️

| Threat | Mitigation |
|--------|------------|
| Unauthorized access | Protect endpoints with authentication and encrypt text at rest via SQLCipher. |
| Data exfiltration | Isolate vectors and metadata; decrypt only on demand within a trusted environment. |
| Data poisoning | Validate inputs and monitor ingestion for anomalous patterns. |
| Denial of Service | Rate‑limit write APIs and cap payload sizes to prevent resource exhaustion. |

---

## 7. Deployment 🚀
- Single-container image (`python:3.12-slim-bookworm`, ≈120 MB)
- Health probes:
  - `/health/live` → liveness (always 200 unless Uvicorn crashes)
  - `/health/ready` → store ping + FAISS integrity
- Docker Compose example spins API + OTEL collector + Grafana/Loki in three commands

See [deployment guide](deployment.md) for setup and client connection steps. For LLM integration and backup procedures, refer to [LLM-agent integration](agent_integration.md) and [backup & restore](backup_restore.md).

---

## Extending
- Swap vector backend: implement `AbstractVectorStore` (FAISS and Qdrant included)
- Plug PostgreSQL metadata backend: use psycopg3 + JSONB
- Add web-crawler ingest: subclass `BaseIngestor`

---

© 2025 Ingver / AI-memory Team — Licensed under Apache-2.0
