# Phase 4: graph, spectrum, and persisted index

Offline ingestion is now implemented:

`prepare -> enrich -> six signals -> weighted graph -> spectrum -> save -> verify/reload`

`SpectRAGIngestion.ingest()` returns a verified index bundle.
`SpectRAGRetrieval.load()` independently reloads it. Query-time retrieval remains
for Phase 5; these operations do not generate answers.

## Read the implementation in this order

1. `src/spect_rag/ingestion.py`: `ingest()` orchestrates the steps in order.
2. `utility/validation.py`: checks enriched record identities, positions, model
   settings, token counts, vector normalization, and resolved citation targets.
3. `utility/graph_signals.py`: computes six separate evidence signals.
4. `utility/graph.py`: merges signals, fuses weights, thresholds edges, and keeps
   all chunk nodes, including isolates.
5. `utility/spectral.py`: constructs a sparse normalized Laplacian and selects
   low-frequency nontrivial eigenvectors.
6. `utility/indexing.py`: writes a complete snapshot, publishes it, verifies
   checksums/alignment, and rebuilds BM25 when loading.

All helper functions remain under `src/spect_rag/utility/`. There are still only
two classes in the installed package. Local NetworkX/JSON storage implements the
previously documented replacement for the paper's Neo4j deployment.

## Six signals: Section 3.2.1 and Equation 10

| Signal | Implemented score | Candidate restriction |
| --- | --- | --- |
| Semantic | `max(0, cosine(chunk_i, chunk_j))` | Highest positive `graph_neighbors` per chunk |
| Adjacency | 1 for consecutive positions in the same document | All consecutive pairs |
| Section | Common heading-prefix depth / maximum path depth | Same document and root heading; highest `relation_neighbors` |
| Entity/concept | Jaccard overlap of case-folded concept sets | Shared concepts; highest `relation_neighbors` |
| Utility questions | Maximum positive cosine across all question pairs | Highest `graph_neighbors` distinct chunks per chunk |
| Citation | 1 for a resolved explicit reference | First `relation_neighbors` target chunks in corpus order per citing chunk |

Section affinity, entity Jaccard, maximum-question matching, and the neighbor
caps are explicit implementation choices where the manuscript does not provide
complete formulas/policies. Empty concept/question sets provide no evidence.
Unresolved bibliography references and external URLs do not become guessed edges.

The graph is undirected. Directed neighbor selections and citation relations
are symmetrized by union; repeating a pair uses the maximum score for a signal,
not a double-counted sum. All six signal scores are retained on every kept edge.
No self-loops are added. Equal neighbor scores use corpus row order as a tie-break.

Weights follow the paper's first Section 5.4 configuration:

| Semantic | Adjacency | Section | Entity | Utility | Citation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.300 | 0.084 | 0.250 | 0.200 | 0.083 | 0.083 |

`edge.weight = sum(configured_weight[signal] * edge[signal])`.
Custom named weights override this preset. Weights must be finite, nonnegative,
and not all zero. They need not sum to one; missing signals are not silently
renormalized. Zero-weight signals are disabled. Only positive fused weights at
or above the threshold are kept.

The defaults below are package choices, not established paper constants:

| Argument to `ingest()` | Default | Meaning |
| --- | ---: | --- |
| `graph_neighbors` | 10 | Per-source cap for semantic and question similarity |
| `relation_neighbors` | 20 | Per-source cap for section, concept, and citation links |
| `edge_threshold` | 0.05 | Minimum retained fused weight |
| `edge_weights` | Configuration 1 | Optional named overrides |
| `spectral_dimension` | 32 | Maximum number of positive Laplacian modes |

Both neighbor limits can be zero for ablation; adjacency remains independently
controlled by its weight. A high threshold may create isolates, which are retained.
The stored graph reports candidate/kept edge counts, isolates, components, and
counts of retained edges carrying each signal.

These caps bound edge storage by `O(n * (graph_neighbors + relation_neighbors + 1))`.
They do not cap every node's final degree: many sources can select one target.
They can also drop useful section/concept/citation links. The exact similarity
searches and large shared-concept/section groups still have quadratic worst-case
computation. No dense n-by-n similarity matrix is materialized, and no large-corpus
scalability benchmark has been performed. Weight auto-calibration is not implemented.

## Spectral coordinates: Equations 11-13

For positive-degree nodes, the sparse operator is
`L_sym = I - D^(-1/2) W D^(-1/2)`. Isolates use zero rows and zero diagonal,
avoiding division by zero. This is an explicit zero-degree convention.

Connected components are decomposed separately. Components up to 256 nodes use
dense symmetric `eigh`; larger components use sparse `eigsh` with a fixed start
vector, unless all their modes are requested. All modes at or below `1e-9` are
excluded. From the component results, the globally smallest positive eigenvalues
are selected, up to `spectral_dimension`. Signs use a largest-magnitude pivot
convention. Coordinates are eigenvector rows, not normalized semantic embeddings.

Consequences:

- Empty, singleton, and edgeless graphs have zero usable spectral dimensions.
  Empty graphs are supported by graph/spectral utilities; an empty saved retrieval
  index is rejected because it has no evidence.
- A connected n-node graph has at most n-1 nontrivial modes. Tiny corpora return
  fewer than 32 dimensions rather than padding with artificial modes.
- Every component's zero mode is excluded, not just the first global eigenvector.
  Isolated chunks retain zero coordinates.
- When truncation selects no positive mode from a component, all its rows are
  zero. Phase 5 must treat isolates/components absent from the retained basis as
  unavailable spectral evidence, rather than inventing proximity from zeros.
- A numerical tolerance can also exclude extremely small positive eigenvalues.
  Repeated eigenvalues permit equivalent bases; bitwise coordinates across BLAS
  versions/platforms are not promised.

Eigenpair residuals must be at most `1e-6`. Loading verifies shape, finiteness,
orthonormal columns, positive sorted eigenvalues, component count, and residuals
against the saved graph. Eigensolver failure does not publish a new index.

## Snapshot layout

```text
indexes/my_docs/
    CURRENT                        # published generation ID
    snapshots/
        <generation-id>/
            manifest.json          # version, models, settings, hashes, counts, order
            chunks.json            # text, question text, concepts, citations, provenance
            embeddings.npy         # chunk vectors in chunk_ids order
            question_embeddings.npy
            question_owners.npy    # owning chunk row for every question vector
            spectral_coords.npy    # same chunk order
            eigenvalues.npy
            graph.json             # sparse edge list and all signal attributes
            lexical_tokens.json
            dense.faiss            # exact IndexFlatIP
            questions.faiss        # exact IndexFlatIP, possibly zero rows
```

The manifest records source content hashes and model identities/settings, the
`cl100k_base` tokenizer, chunking/graph/spectral configuration, ordered chunk IDs,
and SHA-256 checksums for all ten data artifacts. Array loading disables pickle.
BM25 is rebuilt from saved Unicode word tokens; it is not pickled. For a corpus
with no lexical words, `bm25` is explicitly `None`. Original source files and
model services are not required to reload a snapshot.

Saving writes a temporary snapshot, finishes its manifest, renames it to its
generation directory, and atomically replaces CURRENT. Failure before publication
leaves the previous generation readable. A full rebuild publishes a new snapshot
and retains old ones; it does not incrementally modify the old graph. Old snapshots
can be removed manually when they are no longer needed, keeping CURRENT's target.
Model caches remain separate from these index snapshots.

Loading validates checksums before reading the data, then checks chunk/model/order
consistency, question-vector owners, fused graph weights, spectral residuals,
FAISS dimensions/counts/vector content, and BM25 tokens. An incomplete or mixed
snapshot is rejected. The optional utility call
`load_index(path, embedding_model=..., embedding_dimension=..., document_prefix=...)`
also checks compatibility with the caller's expected embedding configuration.

The returned bundle contains `chunks`, `graph`, `embeddings`,
`question_embeddings`, `question_owners`, `spectral_coords`, `eigenvalues`,
`dense_index`, `question_index`, `lexical_tokens`, `bm25`, `manifest`, and
`snapshot_dir`. Loaded chunks also carry their own spectral-coordinate rows.

## Run and review

```bash
uv run python examples/ingest_documents.py examples/documents/getting_started.md examples/documents/storage_notes.txt --demo --index-dir indexes/demo
uv run python -m unittest discover -s tests -v
```

The demo uses the same explicitly fake callbacks as Phase 3. The six-chunk
example builds 15 edges, including evidence from every signal, and five spectral
dimensions. Running it again reuses model caches while publishing a fresh complete
snapshot. Real use requires your own embedding/extraction callbacks.

The supplied 25-page paper was indexed/reloaded with those test callbacks: 63
chunks, 591 edges, 32 spectral dimensions, and maximum eigenpair residual around
`1.4e-15`. Its demo extractor produced no concepts/questions, so this is an
integration/numerical check, not a semantic model or retrieval-quality result.

Tests cover hand-computed six-signal fusion, question-pair aggregation, neighbor
caps, sparse storage, isolates, every component's zero modes, known path-graph
eigenvalues through the sparse solver, save/reload/FAISS/BM25 alignment, corrupted
or mismatched artifacts, and failed/successful rebuilds. Query ranking and context
selection remain for Phase 5. Stop here for review before proceeding.
