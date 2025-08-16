# Unified Memory System

   Enterprise-grade memory system with vector search, FastAPI and monitoring.

   ## Features

   - Asynchronous storage and retrieval APIs
   - Semantic search via FAISS HNSW + cosine similarity
   - PII filtering, encryption-at-rest, automated backups
   - FastAPI layer with auth, CORS, Prometheus metrics, rate limiting
   - Logs automatically scrub PII when enabled
   - Pluggable summarization strategies for memory consolidation
   - Optional MiniLM cross-encoder reranker for higher precision
   - Optional libSQL backend using `libsql-client`
   - Idempotent write operations via `Idempotency-Key` headers

When the storage backend is configured with a `libsql://` DSN (for example
`libsql://<url>?auth_token=...`), WAL mode and checkpointing are handled by the
remote server. Local `wal` settings and checkpoint intervals are ignored.

### Summarization strategies

Clusters can be condensed using different approaches:

- `head2tail` *(default)* – join the two highest-importance texts with an ellipsis.
- `concat` – concatenate all texts in the cluster.

Callers may also provide custom callables implementing their own summarization
logic when invoking the maintenance APIs.

### Attention strategies

Memory ranking is customizable through scoring strategies. ``order_by_attention``
accepts a callable that returns raw similarity scores. Built‑in options include
token overlap and MiniLM embedding similarity, but custom functions may be
supplied. See [attention_strategies.md](attention_strategies.md) for details.

### Ranking best memories

The `/api/v1/memory/best` endpoint scores memories using weighted attributes:

- `importance`
- `emotional_intensity`
- `valence_pos` *(positive valence weight)*
- `valence_neg` *(negative valence penalty)*

Defaults are configured via `RankingConfig` and may be overridden with
environment variables such as `AI_RANKING__IMPORTANCE=2.0`.
Per-request weights can be supplied as query parameters.  Results may also
be limited to a specific ``level`` or filtered by metadata fields such as
``user_id``:

```bash
curl "http://localhost:8000/api/v1/memory/best?limit=2&level=1&user_id=42&importance=2.0&valence_neg=1.0"
```

## Database Encryption

Encryption features depend on the
[`cryptography`](https://pypi.org/project/cryptography/) package. To enable
SQLCipher for the SQLite backend, set `encrypt_at_rest` to `true` and supply a
base64 Fernet key via the `encryption_key` setting or the
`AI_SECURITY__ENCRYPTION_KEY` environment variable. The key is passed to the
database as the `cipher_secret` parameter in the DSN.

### Key management

Keys can be loaded from different backends via the `kms_backend` option in
`SecurityConfig`. Supported values are `local`, `vault` and `aws`. Additional
connection details may be supplied through `kms_params`. When using AWS, supply
the key ID via `kms_key_id` or the `AI_SECURITY__KMS_KEY_ID` environment
variable.

```bash
AI_SECURITY__KMS_BACKEND=aws
AI_SECURITY__KMS_KEY_ID=arn:aws:kms:...:key/123
AI_SECURITY__KMS_PARAMS='{"region_name":"us-east-1"}'
```

Standard AWS environment variables (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
``AWS_REGION`` etc.) are honoured by the backend.

### Secure preset

The ``scripts.secure_entry`` module bootstraps SQLCipher with a key from
``CryptoContext`` (AWS KMS or local fallback). Run with ``AI_ENV=production``
to enable a hardened configuration.

When `kms_backend` is set to `local`, the `encryption_key` field is used
directly. External backends are expected to return a Fernet compatible key.

## Vector index encryption and rotation

FAISS index shards and their ID maps are encrypted at rest using the same
`CryptoContext` infrastructure that powers database encryption.  Key material
is stored through pluggable backends—by default a local JSON keyring—but can be
swapped for Vault or AWS KMS implementations.  Keys rotate automatically on a
30‑day schedule with deprecated keys retired after a 90‑day grace period.  No
additional configuration is required; the keyring is consulted on save/load and
will transparently decrypt historical indices.

## Audit logging

Read and write operations performed by the SQLite memory store are now recorded
to the `memory_system.audit` logger.  Each operation logs the type (`add`,
`get`, `search`, `update`, `delete`) and relevant identifiers to aid in
compliance audits.

## Password hashing

Password and API token hashes now rely on Argon2 via the ``argon2-cffi``
package, offering strong protection against brute‑force attacks. When Argon2
is unavailable, the standard library's ``hashlib.scrypt`` is used as a secure
fallback without requiring additional configuration.

## aiosqlite fork

This repository bundles a thin `aiosqlite`-compatible wrapper around
`sqlite3`. Differences from the upstream project are detailed in
[aiosqlite_fork.md](aiosqlite_fork.md).

## Operations guides

   - [Deployment guide](deployment.md) — server setup and client connection
   - [LLM-agent integration](agent_integration.md) — step-by-step tutorial
   - [Backup and restore](backup_restore.md) — snapshots and data import/export
   - [Cross-encoder reranking](cross_encoder.md) — optional semantic reranker
   - [Vector store plugins](vector_store_plugins.md) — extend with custom backends
   - [Qdrant setup](qdrant_setup.md) — configure remote vector index
