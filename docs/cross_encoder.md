# Cross-Encoder Reranking

UMS can optionally rerank search results with a cross-encoder.  Two backends
are available:

* A local MiniLM model from
  [`sentence-transformers`](https://www.sbert.net/)
* [Cohere's Rerank](https://docs.cohere.com/docs/rerank) API

Both score each memory against the query for improved relevance.

## Requirements

Install the extra dependency:

```bash
pip install sentence-transformers
```

A GPU is used automatically when available; otherwise the model runs on CPU.
If the package or CUDA is unavailable the reranker falls back to a no-op.

## Activation

Enable the reranker via settings and request it during search:

```python
from memory_system.unified_memory import search
from memory_system.settings import UnifiedSettings

settings = UnifiedSettings.for_testing()
settings.ranking.use_cross_encoder = True

results = await search("example", reranker="cross", k=5)
```

The flag can also be set with `AI_RANKING__USE_CROSS_ENCODER=1` in the
environment.

To use the Cohere backend set `AI_RANKING__CROSS_ENCODER_PROVIDER=cohere` and
provide a `COHERE_API_KEY`.  The model defaults to the multilingual
`rerank-multilingual-v3.5` variant and can be overridden with
`COHERE_RERANK_MODEL`.
