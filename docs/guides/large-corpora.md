---
description: >-
  Approximate vector search, the sparse eigensolver, and measured speed and memory on 10,000 to
  100,000 chunks.
---

# Large corpora

With the core package, SpectRAG uses exact numpy computations. That is fast up to a few thousand
chunks. For larger corpora, install the `scale` extra:

```bash
pip install "spectrag[scale]"       # scipy, faiss-cpu, hnswlib
```

## What changes with `scale`

| Part | Without `scale` | With `scale` |
|---|---|---|
| Spectral coordinates | dense eigensolver (`numpy.linalg.eigh`): builds an *n* × *n* matrix | above 2,000 chunks: sparse solver (`scipy.sparse.linalg.eigsh`), same eigenvectors |
| Nearest neighbours (semantic edges, dense search) | exact, in blocks | `ann_backend = "auto"`: exact up to 20,000 chunks, then FAISS HNSW (or hnswlib) |

The choice is automatic. To force a backend:

```python
SpectRAGIngestion("my_index", {"ann_backend": "faiss"}).ingest("docs/")   # or "hnsw", "exact"
```

With an approximate backend, dense scores are computed for the `dense_top_n` (1,000) nearest
chunks only. HNSW quality and speed are set by `hnsw_m`, `hnsw_ef_construction` and
`hnsw_ef_search` (see [Configuration](configuration.md#vector-search)).

## Measured performance

All numbers come from runs in the repository on an Apple Silicon Mac. Commands and details are
in the
[production-readiness report](https://github.com/aion7891/Spect-RAG/blob/main/docs/PRODUCTION_READINESS.md#5-performance-from-phases-13).

| Corpus | Ingestion | Search (median) | Index on disk | Memory |
|---|---|---|---|---|
| MultiHopRAG: 609 articles → 10,335 chunks, hash embedder | 7.7 s after embedding | 4.1 ms | 49 MB | about 1 GB peak during ingestion |
| MultiHopRAG: 7,807 chunks, `nomic-embed-text` | dominated by embedding through Ollama | 3.5–4.4 ms, plus 19 ms to embed the question | 39.5 MB | 256 MB after loading |
| 100,000 synthetic 768-d vectors, kNN graph only | exact 70.9 s, FAISS 10.4 s (the same edges) | | | |

Search time is the SpectRAG pipeline (seeds, expansion, re-scoring, selection) with a
12,000-token budget. Embedding the question is extra and depends on your model.

## Tips

* **Embedding dominates ingestion time** with a real model. Batch your custom embedder and run
  it on a GPU if you can.
* **Utility questions** call an LLM once per chunk; leave `question_fn` out for large corpora,
  or generate questions for a sample first to see whether they help.
* **Semantic chunking** embeds every sentence. `semantic_embedder = "hash"` makes it much
  faster; documents with headings are split by heading and need no sentence embeddings.
* **Graph settings** can be tuned later without re-embedding: see
  [Hyperparameter tuning](tuning.md).
* **Opening an index** loads its arrays into memory: the 7,807-chunk MultiHopRAG index with
  768-dimensional vectors used 256 MB after loading.

To re-run the vector-search comparison: `python benchmarks/ann_benchmark.py` (from a clone).
