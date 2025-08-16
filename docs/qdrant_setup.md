# Qdrant setup

This guide explains how to use [Qdrant](https://qdrant.tech/) as the remote
vector index for the Unified Memory System.

## Install dependencies

Install the optional extras that pull in the `qdrant-client` package:

```bash
pip install ai-memory[qdrant]
```

## Run Qdrant locally

Start a standalone Qdrant service (Docker image shown):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

The server listens on `http://localhost:6333` by default.

## Configure UMS

Point the memory system at the running Qdrant instance:

```bash
export AI_DATABASE__BACKEND=qdrant
export AI_DATABASE__QDRANT_URL=http://localhost:6333
export AI_DATABASE__QDRANT_COLLECTION=memory
```

Run the API as usual:

```bash
uvicorn memory_system.api.app:create_app --factory --reload
```

## Notes

- Qdrant stores vectors only; SQLite continues to hold metadata and IDs.
- SQLCipher encryption does **not** extend to Qdrant. Use TLS and/or disk
  encryption for the Qdrant server when handling sensitive data.
- `LocalPersonalStore` offloads vectors to Qdrant automatically when a
  Qdrant URL is configured and the local FAISS index grows beyond its
  threshold.
