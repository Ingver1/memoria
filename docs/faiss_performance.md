# FAISS performance options

The `FaissHNSWIndex` supports several features to improve throughput
and memory usage.

## Dataset driven defaults

When the expected dataset size is known the vector store can derive
sensible parameters automatically.  The `faiss.dataset_size` setting
enables this behaviour.  Given a dataset size \(N\):

* `ivf_nlist` ≈ `nlist_scale * sqrt(N)` (with `nlist_scale` between 4–16)
* `ef_search` ≈ `ef_search_scale * ivf_nlist`
* `ef_construction` ≈ `ef_construction_scale * ef_search`
* `pq_m` ≈ `vector_dim // pq_m_div`

These multipliers are exposed in the configuration allowing rapid
experimentation without code changes.

```toml
[faiss]
dataset_size = 100000           # number of vectors expected
nlist_scale = 8.0               # optional override of sqrt multiplier
ef_search_scale = 0.5           # scales ef_search from nlist
ef_construction_scale = 2.0     # scales ef_construction from ef_search
pq_m_div = 24                   # derives pq_m from vector dimensionality
```

Explicit numeric values for `nlist`, `ef_search`, `ef_construction`,
`pq_m` or `pq_bits` take precedence over the derived heuristics.

## GPU acceleration

Set `use_gpu=True` when constructing the index or export
`AI_FAISS__USE_GPU=1` to move FAISS structures to available GPUs:

```python
from memory_system.core.index import FaissHNSWIndex
index = FaissHNSWIndex(dim=384, use_gpu=True)
```

On failure the index falls back to CPU automatically.

## Quantization

Product quantization can reduce memory footprint and speed up search.
Choose an alternative `index_type` and associated parameters:

```python
FaissHNSWIndex(
    dim=384,
    index_type="IVFPQ",  # or "HNSWPQ", "OPQ"
    pq_m=16,
    pq_bits=8,
)
```

Adjust `ivf_nlist` and `ivf_nprobe` for IVF based indices.  These options
allow experimenting with different accuracy/latency trade‑offs.

## Benchmarking workflow

Use the lightweight benchmark in `bench/faiss_hnsw.py` to compare
different parameter choices.  The script inserts a deterministic set of
vectors and measures build and search times:

```bash
python -m bench.faiss_hnsw --n-vectors 100000 --dim 384
```

Iterate over the configuration multipliers or explicit values while
recording the benchmark output to determine the best trade‑off for your
dataset.
