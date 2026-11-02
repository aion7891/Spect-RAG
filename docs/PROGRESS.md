# SpectRAG upgrade: progress

This file records each phase of the production upgrade: what was done, the measured results,
the tests that were run, and what is still open. Every number here comes from a command that
was actually run; the commands are listed so they can be repeated.

| Phase | Scope | Status |
|---|---|---|
| 1 | Code and mathematical audit, performance | ✅ Done |
| 2 | Algorithm improvements: MMR and other selectors, configuration, tuning, FAISS/HNSW | ⏳ Pending |
| 3 | MultiHopRAG benchmark, baselines, tuning on validation data, tables and charts | ⏳ Pending |
| 4 | Production package: configuration, errors, logging, PyPI metadata, `quickstart.py` | ⏳ Pending |
| 5 | Interactive graph view, README redesign, animated architecture flow | ⏳ Pending |
| 6 | Final testing and validation, production-readiness report | ⏳ Pending |

---

## Phase 1: code and mathematical audit

### How the audit was done

1. Every module was re-read against the paper (Eq. 3, 10–13, 17–27; Tables 1–2; Algorithm 1).
   Phases 2–6 had already been reviewed once; this pass also covered seeding, expansion,
   diffusion, selection, prompt building, metrics and evaluation.
2. The real pipeline was profiled with `cProfile` on the **full MultiHopRAG corpus** (609 news
   articles, 1.06 M words → 10,335 chunks of 256 tokens), the dataset used in Phase 3.
3. Each fix was checked by tests. Each vectorised rewrite was checked against the original
   loop version for **identical** output (`tests/test_equivalence.py`).

### Findings and fixes

| # | Finding | Kind | Fix |
|---|---|---|---|
| 1 | `save_index` took **53%** of ingestion: 2.9 M raw signal edges written as indented JSON | Performance | Edges and signals saved as binary `.npy` / `.npz`; chunks as compact JSON. Old JSON indexes still load |
| 2 | **2.79 M entity edges** on 10k chunks: the regex extractor produced fragments (`u`, `s` from "U.S."), months, weekdays and "new"; common names below the 5% cap still linked hundreds of chunks | Correctness and performance | One-letter fragments and months/weekdays/"new" removed; cap lowered to 1% of chunks (≥ 20); entity and section signals keep each chunk's `graph_k` strongest edges → **69,570** entity edges |
| 3 | Graph walk used Python dicts (4.9 M lookups per 300 queries) and fully sorted every layer; hop 2 was computed even when hop 1 already filled the 200-candidate cap | Performance | Graph stored once as CSR arrays; the walk uses numpy; it stops when the cap is reached |
| 4 | RRF, rankings and the "has questions" mask looped over all chunks in Python for every query | Performance | Vectorised (`np.argsort`, `np.lexsort`, array RRF) with the same tie order |
| 5 | BM25 scoring looped over every posting in Python | Performance | Postings stored as numpy arrays; one vectorised update per query word |
| 6 | Greedy selection evaluated Eq. 21 one candidate at a time, and `k not in picked` scanned a list | Performance | Eq. 21 computed for all candidates at once (numpy); a boolean mask for picked candidates |
| 7 | `hash_embed` recomputed md5 1.6 M times (its cache lasted for one call only) | Performance | Module-level `functools.lru_cache` |
| 8 | Retrieval loaded the raw signals, which it never uses | Memory | `load_index(..., skip=("signals",))` |
| 9 | `ingest()` only accepted files | Usability | New `ingest_documents(docs)` for documents already in memory |

### Mathematics verified

| Check | Result |
|---|---|
| Eigenpairs on the real 10,335-node graph (sparse `eigsh` path) | max ‖L u − λ u‖ = **5.6e-15**; U orthonormal; orthogonal to the trivial `√degree` vector (max \|uᵀu₁\| = 5.2e-15); λ₂ = 0.0108 > 0 (the graph is connected) |
| Personalized PageRank (Eq. 18) | equals the closed form β(I − (1−β)Ã)⁻¹r (existing test) |
| Sparse vs dense eigensolver | same eigenvalues to 1e-6 (existing test) |
| Spectral closeness and clustering under eigenvector sign flips | invariant: flipping the sign of coordinate k flips both the candidate and the seed centre, so the cosine is unchanged; k-means on row-normalised points mirrors |
| Vectorised expansion, RRF/top-seeds, BM25, selection vs the original loops | **identical** results on 71 random cases with deliberate ties |
| BM25 vs a from-scratch implementation of the formula | equal (`test_bm25_matches_reference`) |

No mathematical errors were found in Eq. 3, 10–13, 18, 21–22 or the Section 4.5 metrics. The
problems were efficiency and entity-extraction noise.

### Measured on MultiHopRAG (10,335 chunks, offline hash embedder, 300 real queries)

Both runs used the same script (`ingest(path)` on the 609 articles as files, then `search()` with
a 2,560-token budget). "Before" is the Phase 10 commit (`6e244e6`), "after" is this phase.

| | Before | After | Change |
|---|---|---|---|
| Ingestion time | 13.2 s | **7.7 s** | −42% |
| Peak memory (ingest + load + 300 searches) | 1,556 MB | **1,046 MB** | −33% |
| Index size on disk | 159 MB | **49 MB** | −69% |
| `search()` latency, median | 36.4 ms | **4.1 ms** | **8.9× faster** |
| `search()` latency, p95 | 46.4 ms | **4.3 ms** | |
| Graph edges (fused, after `graph_k`) | 91,699 | 94,559 | about 9 per chunk |

The graph edges changed slightly because the entity signal is now cleaner (finding 2).

### Tests

```text
uv run --all-extras pytest -q                     248 passed, 1 skipped (opt-in LLM test)
minimal install (numpy + pytest only)             244 passed, 5 skipped (pypdf / scipy / LLM)
all 9 example scripts                             run without errors
```

New tests: `tests/test_equivalence.py` (71 cases, against `tests/reference.py`); the binary store
and the `skip` option; loading old JSON indexes; the entity filter.

### Open after Phase 1

- `semantic_edges()` is still `O(n²)` at ingestion (in blocks, so memory is fine) → Phase 2 (ANN).
- The values marked ⚠️ in the settings (α, precision floor, τ, …) are still guesses → Phase 2–3
  (tuning on MultiHopRAG validation data).
- The README still describes the JSON file layout; it is rewritten in Phase 5.
