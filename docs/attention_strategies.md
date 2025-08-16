# Attention strategies

Memoria ranks memories relative to a query using *attention* scores. The
``attention_weights`` and ``order_by_attention`` helpers accept a *scoring*
callable that returns raw similarity scores for each memory. These scores are
combined with optional metadata weights and ``importance`` attributes before
softmax normalization.

## Built‑in strategies

- ``score_token_overlap`` – counts shared non‑stopword tokens between the query
  and memory text. This is the default for backward compatibility.
- ``score_embeddings`` – cosine similarity between MiniLM embeddings of the
  query and memories.
- ``score_tfidf`` – basic TF‑IDF implementation useful for experimentation.

## Custom strategies

Any callable taking ``(memories, query)`` and returning a list of scores can be
used:

```python
from memory_system.core.keyword_attention import order_by_attention

def my_scorer(memories, query):
    return [0.0 for _ in memories]  # custom logic here

ordered = order_by_attention(mems, "hello", scoring=my_scorer)
```

Custom scorers may leverage caching via ``memory_system.unified_memory._get_cache``
or incorporate domain‑specific metadata. Registering additional strategies is as
simple as exposing a function and passing it to ``order_by_attention``.
