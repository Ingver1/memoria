# Deployment Guide

This guide covers how to run the memoria service and connect a client. It assumes a
Linux host with Python 3.11 or Docker installed.

## 1. Server setup

1. **Install dependencies**
   ```bash
   git clone https://github.com/your-org/memoria.git
   cd memoria
   pip install -e ".[api,faiss]"
   ```
2. **Configure environment**
   - `AI_DATA_DIR` – directory for the SQLite database and FAISS index.
   - `AI_SECURITY__ENCRYPTION_KEY` – optional base64 key for at‑rest encryption.
3. **Start the API server**
   ```bash
   uvicorn memory_system.api.app:create_app --factory --host 0.0.0.0 --port 8000
   ```

   For production deployments, launch the server with Gunicorn:

   ```bash
   WORKERS=4 TIMEOUT=120 GRACEFUL_TIMEOUT=30 ./scripts/run_gunicorn.sh
   ```

   `WORKERS` controls the number of worker processes, `TIMEOUT` sets the request
   timeout in seconds and `GRACEFUL_TIMEOUT` configures the shutdown grace
   period.

`create_app` uses FastAPI's lifespan to initialise the `EnhancedMemoryStore` and
attach it to `app.state.memory_store`. The server exposes OpenAPI docs at
`http://localhost:8000/docs`.

## 2. Client connection

Clients interact with the FastAPI layer over HTTP.

### Add a memory
```bash
curl -X POST "http://localhost:8000/api/v1/memory/add" \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world", "metadata": {"user_id": 1}}'
```

### Search
```bash
curl "http://localhost:8000/api/v1/memory/search?query=hello&limit=5"
```

### Python SDK example
```python
from memoria.memory_system.api.client import MemoryClient

client = MemoryClient("http://localhost:8000")
client.add(text="remember me", metadata={"user_id": 1})
results = client.search(query="remember", limit=3)
```

For more advanced configuration such as running behind a reverse proxy or using
Docker Compose with observability tools, see `docs/architecture.md`.
