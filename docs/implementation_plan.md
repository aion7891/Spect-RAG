# Implementation phases

The goal is an understandable package with exactly two public classes:
`SpectRAGIngestion` for offline indexing and `SpectRAGRetrieval` for online
retrieval. Both use the same index directory. All helper functions belong under
`src/spect_rag/utility/`; use plain dictionaries and arrays for internal records
instead of adding a framework of classes.

Phases 1-5 are complete. **Stop after every phase for user review.**
Phase 6 below describes planned work, not implemented capabilities.

| Phase | Scope | Reviewable result and validation |
| --- | --- | --- |
| 1: analysis and package scaffold | Read the supplied PDF, map paper steps to code, document uncertainties, preserve the prototype, define two public interfaces and package metadata. | Import the installed package from outside the repository; inspect the analysis and this plan. No algorithm claims yet. |
| 2: loading and chunking | Add `utility/loading.py` and `utility/chunking.py`; load text, Markdown, and machine-readable PDFs; carry source/page/section metadata; use `cl100k_base` token counts and sentence-aware windows. Add structural chunking first and clearly expose the semantic path when embedding support arrives. | Inspect real chunks from a small document; check maximum size, overlap policy, source metadata, and short/oversized sentences and table rows. Scanned PDFs must report that OCR is required. No silent text loss. |
| 3: embeddings and extracted metadata | Add embedding and extraction functions, caller-supplied model callbacks, sentence-distance semantic boundaries, entity/concept extraction, utility-question text and embeddings, citation targets, and extraction caching. | Verify embedding shape, finiteness, normalization, task prefixes, callback behavior, and reusable cached outputs. Use deterministic test vectors; distinguish them from real semantic embeddings. |
| 4: graph, spectrum, and persistence | Implement six edge signals, configurable fusion and sparsification, sparse graph construction, normalized Laplacian, nontrivial eigenvectors, FAISS exact indexes, BM25 rebuild data, and local persistence. Wire the complete ingestion class. | Ingest and reload a small corpus; inspect per-signal edges, stable chunk order, eigenpair residuals, empty/singleton/disconnected graphs, and matching artifacts. Preserve all isolated chunks. |
| 5: retrieval and context selection | Implement dense/BM25/question RRF seeds, deterministic bounded two-hop expansion, spectral/PPR proximity, relevance blending, protected core, precision-oriented tail, coverage tie-break, and backfill. Wire the retrieval class. | Hand-check a tiny graph with bridge evidence; enforce token budget and candidate cap; inspect scores, stable rankings, source records, and missing optional signals. Retrieval must work without a generation LLM. |
| 6: document example and end-to-end evaluation | Add an executable example and small corpus, optional grounded answer generation through the retrieval class, a readable retrieval trace, citations, and basic evidence precision/recall evaluation against labeled questions. | Run ingestion -> reload -> retrieval on documents; separately test optional generation. Document observed results, actual model configuration, limitations, and costs. |

## Target class responsibilities

`SpectRAGIngestion.ingest(document_paths)` builds the persisted index. Its
body should read in pipeline order, calling small utility functions whose
docstrings cite the corresponding paper section. It will not also answer queries.

`SpectRAGRetrieval.retrieve(query, token_budget=12000)` loads/uses that index
and return evidence with text, source metadata, and scores. Optional generation
can later be another method on this same class. Provider-specific embedding and
LLM code will be callbacks; the package will not hard-code API keys or a provider.

Both `ingest` and `retrieve` are implemented. Answer generation remains for Phase 6.
Phase 2 adds `SpectRAGIngestion.prepare(document_paths)`, which loads documents
and returns inspectable chunk records without writing an index. Phase 3 adds
optional semantic chunking and `enrich(document_paths)`, which embeds chunks,
extracts concepts/questions through callbacks, and resolves local citations.
Explicitly configured caches can persist model outputs. Phase 4 adds six graph
signals, fusion/sparsification, spectral coordinates, FAISS indexes, BM25 rebuild
data, and checked index snapshots. `SpectRAGRetrieval.load()` verifies/reloads
those artifacts. Constructors do not create files or load models.

Phase 5 implements exact dense/question search, BM25/RRF seeds, bounded two-hop
expansion, spectral/PPR re-scoring, and token-budgeted selection. Retrieval returns
ranked evidence records and exposes a JSON-serializable `last_trace`. A retriever
pins its loaded snapshot; call `load()` explicitly to refresh after a rebuild.

## Persisted artifacts

Each snapshot has a versioned manifest with chunk ordering, model identity/dimension, tokenizer,
graph settings, artifact checksums, and input hashes; UTF-8 JSON for records; NumPy arrays for chunk,
question, and spectral vectors; sparse graph data; and FAISS index files. Rebuild
BM25 from saved tokenized text. No pickle is used. Preserve
question text as well as vectors so the extraction can be inspected.

Index loading rejects incompatible or incomplete artifacts instead of mixing
models, chunk orders, or graph versions. A CURRENT pointer publishes a complete
snapshot; rebuilding preserves old snapshots. Updating the corpus means a
full rebuild; incremental indexing is outside the first implementation.

## Review decisions

The proposed baseline follows Algorithm 1. Local NetworkX graph storage replaces
Neo4j in the first implementation. The first signal configuration in Section 5.4
is the starting weight preset; unspecified numerical settings will be explicitly
named implementation defaults when their phase is reviewed. Separate cluster/MMR
selection and exact heat diffusion are extensions after the baseline works.
