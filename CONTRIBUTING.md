# Contributing

## Performance test thresholds

Performance tests in `tests/test_performance.py` rely on runtime thresholds. To
accommodate different hardware these limits can be overridden via environment
variables when running the tests. The defaults (in milliseconds unless stated
otherwise) are:

- `MAX_EMBEDDING_TIME_MS` (100)
- `MAX_BATCH_PER_TEXT_MS` (50)
- `MAX_BATCH_TOTAL_MS` (5000)
- `MAX_CONCURRENT_PER_TASK_MS` (200)
- `MAX_CONCURRENT_TOTAL_MS` (10000)
- `EMBEDDING_CACHE_FACTOR` (0.5, multiplier)
- `MAX_EMBEDDING_MEMORY_MB` (100)
- `MAX_INDEX_AVG_SEARCH_MS` (5)
- `MAX_INDEX_MAX_SEARCH_MS` (20)
- `MAX_INDEX_BUILD_PER_VECTOR_MS` (10)
- `MAX_INDEX_CONCURRENT_AVG_MS` (50)
- `MAX_INDEX_CONCURRENT_TOTAL_MS` (10000)
- `MAX_INDEX_MEMORY_KB` (10)
- `MAX_WRITE_PER_VECTOR_MS` (30)
- `MAX_FLUSH_TIME_MS` (5000)
- `MAX_READ_AVG_MS` (1)
- `MAX_READ_MAX_MS` (20)
- `MAX_WRITE_CONCURRENT_TOTAL_MS` (10000)
- `MAX_READ_CONCURRENT_TOTAL_MS` (10000)
- `MAX_CACHE_GET_AVG_MS` (0.1)
- `MAX_CACHE_GET_MAX_MS` (1)
- `MAX_CACHE_PUT_AVG_MS` (0.1)
- `MAX_CACHE_PUT_MAX_MS` (5)
- `MIN_CACHE_OPS_PER_SEC` (1000)
- `MAX_CACHE_CONCURRENT_TOTAL_MS` (5000)
- `MAX_PII_DETECT_MS` (10)
- `MAX_PII_REDACT_MS` (10)

Set any of these before running `pytest` to tune the performance thresholds
without editing the tests.
