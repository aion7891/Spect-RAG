# Changelog

## 0.1.0 (unreleased): first public release

The base release, with every core part of SpectRAG (arXiv:2610.36991). Default settings follow
the paper where it gives values; tuned defaults are planned for a later version.

- **Ingestion:** structural and semantic chunking, six edge signals (semantic, adjacency, section,
  entity, utility questions, citation), weighted chunk graph (Eq. 10), normalised Laplacian and
  spectral coordinates (Eq. 11–13), spectral clusters.
- **Retrieval:** hybrid seeds (dense + BM25 + utility questions, RRF), two-hop graph expansion,
  proximity re-scoring (Personalized PageRank, spectral closeness, heat kernel), token-budgeted
  selection (Eq. 19–22) and the paper's alternative selectors (`mmr`, `protected_mmr`,
  `anchor_expand_ce`), answers with any LLM.
- **Scale:** exact, FAISS or HNSW vector search; sparse eigensolver above 2,000 chunks; binary
  index files.
- **Configuration:** every setting validated with clear messages; per-search overrides;
  `rebuild_graph()` changes graph settings without re-embedding; grid/random-search tuning.
- **Production:** exceptions that subclass the matching built-ins, a `spectrag` logger,
  versioned index format, safe (atomic) index saving, retries for Ollama calls, the `spectrag`
  command-line tool, `quickstart.py`.
- **Visualization:** an interactive, self-contained HTML graph view of the chunk graph and of a
  question's retrieval path (`spectrag graph`, `save_graph_view()`).
- **Evaluation:** the paper's metrics and a reproducible MultiHopRAG benchmark (`benchmarks/`).
