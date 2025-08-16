#!/usr/bin/env bash
set -euo pipefail

# Run the API server using Gunicorn.
# WORKERS, TIMEOUT, and GRACEFUL_TIMEOUT can be set to override defaults.

WORKERS=${WORKERS:-4}
TIMEOUT=${TIMEOUT:-120}
GRACEFUL_TIMEOUT=${GRACEFUL_TIMEOUT:-30}

exec gunicorn \
  -k uvicorn.workers.UvicornWorker \
  -w "$WORKERS" \
  --timeout "$TIMEOUT" \
  --graceful-timeout "$GRACEFUL_TIMEOUT" \
  memory_system.api.app:app
