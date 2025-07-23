# memoria — System Architecture

<details>
<summary>Mermaid diagram — end-to-end flow (click to expand)</summary>

```mermaid
graph TD
  subgraph  Ingestion & Vectorisation
    A[Incoming content<br />(text • PDF • web-hook)] -->|clean + chunk| B[Embedder<br />(OpenAI • Instructor)]
    B -->|1280-d vector| C((Vector))
  end

  %% Storage layer
  C --> D[FAISS HNSW Index]
  D -->|top-k IDs| E[Rank & Filter]
  E --> F[(SQLite JSON1<br />metadata)]

  %% Retrieval
  F -->|payload| G[[Client / LLM<br />prompt builder]]
```

</details>
---

1. Data Ingestion 🚚

Step	Module	Notes

Clean & Chunk	memory_system.ingest.cleaner	Language-agnostic heuristics, ~2 k-token windows
Embed	memory_system.vector.embedder	Default is OpenAI text-embedding-3-small; can swap in Instructor/HuggingFace
Persist	memory_system.core.vector_store.AsyncFaissHNSWStore	Stores raw vector + PK; metadata written separately


All CPU-bound FAISS calls run in the default thread pool so the event loop stays snappy.


---

2. Storage & Index 📦

Layer	Implementation	Why

Vector	FAISS HNSW (m=32)	Good recall/latency trade-off for ≤5 M vectors on single box
Metadata	SQLite JSON1	Lightweight; enables queries like source='email' AND importance>0.8
Back-ups	Background task replicate()	Writes compressed snapshot (.fz) every N minutes


Compaction

AsyncFaissHNSWStore launches compact() in a background coroutine that:

1. Removes tombstoned vectors


2. Shrinks the .index blob with FAISS’s in-place merge


3. VACUUMs SQLite and ANALYZEs indices


---

3. Retrieval 🔎

1. Semantic search: vector → top-k IDs


2. Re-rank & filter: JSON1 predicates + optional LLM re-ranking


3. Return: Memory dataclass (text, score, metadata)



Latency (bench, AMD 5950X):

k	P50	P95

5	22 ms	38 ms
20	48 ms	71 ms


---

4. API Surface 🌐

REST (/memory/add, /search)

Events (/sse) — memory-updated stream

CLI (ai-mem add/search)

gRPC (optional extra) — high-throughput internal bus


All endpoints wired through FastAPI + OpenTelemetryMiddleware for full span tracing.


---

5. Observability 👀

Component	Tech

Metrics	prometheus_client (latency_seconds, faiss_queries_total)
Tracing	OTEL SDK → OTLP HTTP → Tempo/Jaeger
Logs	logging.yaml → plaintext or JSON (LOG_JSON=1)


---

6. Deployment 🚀

Single-container image (python:3.12-slim-bookworm, ≈120 MB)

Health probes:

/health/live → liveness (always 200 unless Uvicorn crashes)

/health/ready → store ping + FAISS integrity


Docker-Compose example (docker-compose.yml) spins API + OTEL collector + Grafana/Loki in three commands:


---

Extending

Swap FAISS for Qdrant by implementing AbstractVectorStore.

Plug a PostgreSQL metadata backend via psycopg3 + JSONB (drop-in).

Add web-crawler ingest by subclassing BaseIngestor.


---

© 2025 Ingver / AI-memory Team — Licensed under Apache-2.0
