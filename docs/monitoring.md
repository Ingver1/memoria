# Monitoring and Alerts

Memoria exports Prometheus metrics that can be scraped by Grafana.
The file [`../monitoring/alerts.yaml`](../monitoring/alerts.yaml) defines
basic alert rules bound to these metrics.

## Multiprocess setup

When running under multiple Gunicorn or Uvicorn workers, set the
`PROMETHEUS_MULTIPROC_DIR` environment variable to a writable directory
before starting the server.  On startup Memoria will remove stale `*.db`
files from this directory and configure Prometheus' multiprocess
collector so that each worker contributes to a unified metrics view.

Example using four Uvicorn workers:

```bash
export PROMETHEUS_MULTIPROC_DIR=/tmp/metrics
uvicorn memory_system.api.app:app --workers 4
```

Or with Gunicorn:

```bash
export PROMETHEUS_MULTIPROC_DIR=/tmp/metrics
gunicorn -k uvicorn.workers.UvicornWorker -w 4 memory_system.api.app:app
```

Old metrics files from previous runs are cleaned up automatically at
startup, but you can also clear the directory manually if needed.

## Metrics overview

### Counters
- `ums_errors_total` — Total errors labelled by type and component.
- `ums_pool_exhausted_total` — Connection pool exhausted events.
- `ums_memories_created_total{modality}` and `ums_memories_deleted_total{modality}` — Memory lifecycle.
- `ums_cache_hits_total` / `ums_cache_misses_total` — Cache effectiveness.
- `ums_vectors_added_total` / `ums_vectors_deleted_total` / `ums_ann_queries_total` / `ums_ann_query_errors_total` — ANN index activity.
- `ums_quality_gate_drop_reason_total{reason}` — Memories dropped by the quality gate.

### Gauges
- `ums_cache_hit_rate` — Live cache hit ratio.
- `ums_embedding_queue_length` — Current embedding queue length.
- `ums_ann_index_size` — Number of vectors stored.
- `ums_system_cpu_percent` and `ums_system_mem_percent` — Host resource usage.
- `ums_process_uptime_seconds` — Process uptime.
- `ums_memory_accept_rate` — Share of memories that pass quality checks.
- `ums_lesson_promotion_rate` — Frequency of lessons promoted to long-term memory.
- `ums_retrieval_hit_at_k` — Fraction of retrievals with a relevant result within top *k*.

### Histograms
- Latency metrics such as `ums_db_query_latency_seconds`,
  `ums_search_latency_seconds`, `ums_embedding_latency_seconds`,
  `ums_consolidation_latency_seconds`, `ums_forget_latency_seconds`, and
  `ums_ann_query_latency_seconds`.
- `ums_contrib_score` — Distribution of contribution scores.

## Example: search latency

`ums_ann_query_latency_seconds` tracks FAISS search time.  The included
alert fires when the p95 latency stays above **200 ms** for five minutes:

```yaml
- alert: HighSearchLatency
  expr: histogram_quantile(0.95, sum(rate(ums_ann_query_latency_seconds_bucket[5m])) by (le)) > 0.2
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: ANN search p95 latency high (>200ms)
```

Additional rules in the same file watch for query errors using
`ums_ann_query_errors_total`.  Tailor thresholds to your deployment
needs.

## Example: spike in bad memories

`ums_quality_gate_drop_reason_total{reason}` captures memories rejected by the
quality gate. The following alert fires when more than **10** memories per
minute are being dropped for any reason:

```yaml
- alert: BadMemorySpike
  expr: sum(rate(ums_quality_gate_drop_reason_total[5m])) > 10
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: Surge in memories failing quality gate
```
