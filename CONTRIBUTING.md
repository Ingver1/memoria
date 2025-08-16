# Contributing

## Continuous Integration modes

The CI pipeline runs in two configurations:

- **minimal** – installs the package in editable mode without optional
  dependencies. Lightweight stubs located in `tests/_stubs` stand in for
  heavy libraries so only basic checks and tests are executed.
- **full** – installs optional dependencies such as `numpy`, `fastapi`,
  `httpx`, `pydantic` and `faiss` to exercise the complete test suite with
  skips disabled.

When running tests locally, the repository provides a `sitecustomize.py` at the
project root that sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to disable auto-loading
of third-party pytest plugins. Python automatically discovers this module, so no
`PYTHONPATH` tweaks are required:

```bash
pytest
```

For more details on local testing setup, see the [development guide](docs/development.md#testing-prerequisites).

## Optional dependencies

Optional third-party libraries should not be imported at module import time.
Instead, use helper functions such as `require_faiss()`, `require_httpx()`, or
`require_sentence_transformers()` from `memory_system.utils.dependencies`. They
attempt to import the dependency and raise a clear `ImportError` with guidance
on how to install the necessary extras.

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

## Test database seeding

The test suite uses a temporary SQLite database for isolation. A new database
file is created for each run by setting the `UMS_DB_PATH` environment variable
to a unique temporary directory. Only the schema is created; no sample data is
inserted automatically. If you need to inspect the database contents after a
test run, set `UMS_DB_PATH` to a persistent location before running `pytest`

## Deterministic property-based tests

Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/).
To ensure reproducible failures and keep execution time reasonable, the test
suite registers a global Hypothesis profile with a fixed random seed and a
reduced `max_examples` count.  CI sets `HYPOTHESIS_SEED` to a specific value;
setting the same variable locally will run the same sequences of examples.

```bash
export HYPOTHESIS_SEED=1234
pytest
```

Adjust the seed if you need different examples when debugging.
