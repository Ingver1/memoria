# Unified Memory System
   
   Enterprise-grade memory system with vector search, FastAPI and monitoring.
   
   ## Features
   
   - Asynchronous storage and retrieval APIs
   - Semantic search via FAISS HNSW + cosine similarity  
   - PII filtering, encryption-at-rest, automated backups
   - FastAPI layer with auth, CORS, Prometheus metrics

## Database Encryption

To enable SQLCipher for the SQLite backend, set `encrypt_at_rest` to `true`
and supply a base64 Fernet key via the `encryption_key` setting or the
`AI_SECURITY__ENCRYPTION_KEY` environment variable. The key is passed to the
database as the `cipher_secret` parameter in the DSN.
