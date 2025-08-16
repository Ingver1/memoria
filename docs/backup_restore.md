# Backup and Restore

This document outlines how to back up and restore memoria data and describes the
import/export format.

## Backup scenarios
- **Automated snapshots** – the maintenance job periodically calls
  `replicate()` to write a compressed archive to `AI_DATA_DIR`.
- **Manual backup** – trigger the `/api/v1/admin/backup` endpoint or run
  `ai-mem backup` from the CLI.

## Restore
1. Stop the running service.
2. Unpack the desired snapshot into `AI_DATA_DIR`.
3. Restart the API server; the FAISS index and SQLite metadata are loaded
   automatically.

## Data import/export format
Snapshots are gzipped tar archives containing:
- `faiss.index` – binary FAISS vector store.
- `metadata.db` – SQLite database with JSON1 columns.
- `manifest.json` – schema version and creation time.

These files allow migrations across machines or environments. For selective
exports, use `/api/v1/memory/export` to receive JSON lines and
`/api/v1/memory/import` to restore them.
