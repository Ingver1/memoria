# Unified Memory System

   Enterprise-grade memory system with vector search, FastAPI and monitoring.

   ## Features

   - Asynchronous storage and retrieval APIs
   - Semantic search via FAISS HNSW + cosine similarity  
   - PII filtering, encryption-at-rest, automated backups
   - FastAPI layer with auth, CORS, Prometheus metrics, rate limiting
   - Logs automatically scrub PII when enabled
   - Pluggable summarization strategies for memory consolidation
   - Optional libSQL backend using `libsql-client`

### Summarization strategies

Clusters can be condensed using different approaches:

- `head2tail` *(default)* – join the two highest-importance texts with an ellipsis.
- `concat` – concatenate all texts in the cluster.

Callers may also provide custom callables implementing their own summarization
logic when invoking the maintenance APIs.

### Ranking best memories

The `/api/v1/memory/best` endpoint scores memories using weighted attributes:

- `importance`
- `emotional_intensity`
- `valence_pos` *(positive valence weight)*
- `valence_neg` *(negative valence penalty)*

Defaults are configured via `RankingConfig` and may be overridden with
environment variables such as `AI_RANKING__IMPORTANCE=2.0`.
Per-request weights can be supplied as query parameters:

```bash
curl "http://localhost:8000/api/v1/memory/best?limit=2&importance=2.0&valence_neg=1.0"
```

## Database Encryption

To enable SQLCipher for the SQLite backend, set `encrypt_at_rest` to `true`
and supply a base64 Fernet key via the `encryption_key` setting or the
`AI_SECURITY__ENCRYPTION_KEY` environment variable. The key is passed to the
database as the `cipher_secret` parameter in the DSN.

### Key management

Keys can be loaded from different backends via the `kms_backend` option in
`SecurityConfig`. Supported values are `local`, `vault` and `aws` (the latter
two are placeholders). Additional connection details may be supplied through
`kms_params`.

```bash
AI_SECURITY__KMS_BACKEND=local
AI_SECURITY__KMS_PARAMS='{}'
```

When `kms_backend` is set to `local`, the `encryption_key` field is used
directly. External backends are expected to return a Fernet compatible key.
