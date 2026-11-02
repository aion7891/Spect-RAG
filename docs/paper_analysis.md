# SpectRAG paper analysis

Source: [supplied arXiv:2610.36991v2 PDF](2610.36991v2.pdf), 25 pages.
This analysis uses the supplied document, not an independently verified author
implementation. Page numbers below refer to the PDF's printed pages.

## What we need to implement

SpectRAG retrieves an evidence set under a token budget. Dense similarity finds
entry points; a graph helps recover related evidence; graph-spectral coordinates
and Personalized PageRank help rank candidates; a final selector chooses the
context. The graph's nodes are document chunks, not extracted entities.

There are two pipelines. Tables 1-3 and Algorithm 1 provide the clearest executable
baseline. Equations 9 and 21 describe a broader set-level objective; Algorithm 1
is a practical approximation, not an exact optimizer of those equations.

## Offline ingestion: Table 1, Sections 3.2-3.4

1. **Load and chunk.** Retain document identity, order, section paths, references,
   and page provenance where available. Section 3.3 specifies structural splitting
   for headed documents. For unstructured documents, embed sentences and split
   where adjacent-sentence embedding distance exceeds the document's 90th
   percentile. Pack chunks to at most 1,200 `cl100k_base` tokens, with a target
   100-token sentence-aligned overlap. Merge chunks shorter than 32 tokens where
   size limits allow. Preserve table rows. The paper does not resolve conflicts
   between oversized sentences/rows, strict size caps, and never splitting them;
   our policy needs to be explicit when chunking is implemented.
2. **Embed.** The paper uses `nomic-embed-text`, 768 dimensions, with
   `search_document:` on indexed text. L2-normalize vectors and build a FAISS
   `IndexFlatIP`. Normalized inner product is cosine similarity. Section 6.2
   confirms this is exact search.
3. **Extract six edge signals.** Each signal is in [0, 1]. Keep its individual
   value on the edge so the graph can be explained:

   | Signal | Evidence connecting two chunks |
   | --- | --- |
   | Semantic | Positive cosine similarity, restricted to graph nearest neighbors |
   | Adjacency | Consecutive chunks in the same document |
   | Section hierarchy | Shared section heading or parent path |
   | Entity/concept | Shared extracted entities, terms, organizations, or dates |
   | Utility questions | Similarity between generated questions the chunks answer |
   | Citation/dependency | Resolved references, links, footnotes, or dependencies |

   The paper uses `gpt-4o-mini` for concepts and utility questions and caches
   extraction. Store generated question text and its embeddings. Citation strings
   must be resolved to actual corpus targets before creating edges.
4. **Fuse and sparsify.** Equation 10: `w(i,j) = sum(weight[r] * signal[r](i,j))`.
   Drop edges below a configurable threshold. No self-loops are needed for the
   baseline. Restricting only semantic neighbors does not guarantee a sparse
   graph: shared sections/entities can still create many pairs. Candidate-pair
   limits and graph diagnostics need an explicit policy for larger corpora.
5. **Compute spectral coordinates.** Equations 11-13 use
   `L_sym = I - D^(-1/2) W D^(-1/2)` and low-frequency, nontrivial eigenvectors.
   The target dimension is 32. This is graph position, not another semantic
   embedding. Small graphs cannot supply 32 nontrivial modes. Disconnected graphs
   have multiple zero eigenvalues, and isolates require a degree-zero convention.
   Implement and test these cases explicitly rather than dividing by zero or
   blindly dropping one eigenvector.
6. **Persist.** The paper uses Neo4j, `chunks.pkl`, and FAISS files. Proposed local
   adaptation: NetworkX/sparse graph data, JSON records and manifest, NumPy arrays,
   and FAISS files. This preserves the algorithms while simplifying local testing;
   it does not reproduce the paper's Neo4j deployment.

Section 5.4 reports two actual signal-weight presets:

| Signal | Configuration 1 (proposed baseline) | Configuration 2 |
| --- | ---: | ---: |
| Semantic | 0.300 | 0.150 |
| Adjacency | 0.084 | 0.100 |
| Section | 0.250 | 0.200 |
| Entity | 0.200 | 0.250 |
| Utility question | 0.083 | 0.280 |
| Citation | 0.083 | 0.020 |

## Online retrieval: Table 2 and Algorithm 1, page 12

1. Embed the query with the same model and appropriate query task prefix.
2. Rank chunks by dense cosine, BM25 lexical relevance, and query-to-stored-question
   similarity. The question-to-chunk aggregation rule is not specified precisely;
   maximum question similarity is a reasonable implementation choice.
3. Fuse rankings using reciprocal rank fusion:
   `RRF(c) = sum(1 / (k_rrf + rank[channel,c]))` over participating channels.
   Keep 20 seeds. A missing question channel must not be replaced by another dense
   vote. Carry a dense/BM25 hybrid relevance vector for downstream candidates.
4. Expand two graph hops, preserving seeds and limiting the candidate set to 200.
   Use deterministic ordering and bounded traversal so a high-degree graph does
   not require materializing an unlimited neighborhood before applying the cap.
5. Calculate spectral-coordinate closeness to seeds and Personalized PageRank on
   the weighted candidate subgraph. Table 2 uses their mean as hybrid proximity.
6. Blend `score = (1 - alpha) * relevance + alpha * proximity`. Score scales,
   kernel, seed weights, and `alpha` need documented implementation defaults.
7. Select context with a protected high-confidence core, a precision-oriented
   tail, document-coverage tie-breaks, and backfill, under a default budget of
   approximately 12,000 tokens. The number of chunks is adaptive. Backfill's
   relationship to the tail floor must be stated; the paper does not define a
   universal minimum score for the returned context.
8. Return ranked evidence and source metadata. Optionally format context grouped
   by document and generate a grounded answer. Keep retrieval usable on its own.

The evidence token budget and the generator's full prompt/context-window budget
are different. The implementation must state whether separators/headers count and
reserve space for instructions, query, and answer when generation is added.

## Paper ambiguities and reproducibility limits

| Issue | Decision for the proposed baseline |
| --- | --- |
| Equation 2 lists five signals, omitting section hierarchy. | Follow the complete six-signal Equation 10 and Section 3.2.1. |
| Table 1 describes a flat FAISS index as approximate. | Follow Section 6.2: exact `IndexFlatIP`. |
| Section 3.7 gives heat diffusion `exp(-tL)s`; Algorithm 1 uses spectral proximity and PPR. | Implement Algorithm 1 first. Do not label an RBF over coordinates as exact heat diffusion. |
| Equation 21 includes cluster gain and redundancy penalties; Algorithm 1 does not give their implementation. | Start with the documented protected-core selector; expose cluster/MMR variants later with explicit choices. |
| Several algorithm constants are absent. | Mark graph-neighbor count, sparsification threshold, RRF constant, dense/BM25 blend, proximity kernel, PageRank damping, protected-core size, and tail threshold as implementation choices. |
| Eigenvector conventions for small/disconnected graphs are absent. | Choose and test a consistent convention in Phase 4; do not promise 32 useful modes for every corpus. |
| Section 4 says results should be populated after experiments, but later sections contain numeric results. | Treat reported numbers as claims in the supplied manuscript, not validated results for this package. |
| Some result prose disagrees with tables. | Avoid performance promises. For example, Section 4.9 reports EM 72.0 while Table 7 reports 71.2; the discussion calls SpectRAG best on null questions while Table 8 favors HyDE/NaiveRAG. |

The prototype's `alpha=0.4`, core size 5, tail floor 0.35, coverage bonus 0.08,
RRF constant 60, and PPR damping 0.85 are not established paper defaults.
Figure 3 also illustrates a 0.5/0.5 relevance/proximity blend, underscoring that
the prototype's blend is a choice. Exact benchmark reproduction requires more
configuration detail and running the evaluation; this phase does neither.

## Review of the original retrieval prototype

The original `src/spect_rag.py` has been preserved byte-for-byte as
`reference/initial_retrieval_prototype.py`. It is a useful Algorithm 1 sketch,
but not a complete implementation:

- It assumes chunks, embeddings, graph, and spectral coordinates already exist.
  There is no loading, chunking, graph construction, indexing, or persistence.
- Dense retrieval scans with Python/NumPy rather than using a persisted FAISS index.
- Missing questions fall back to dense scores, giving the same signal two votes.
- Min-max normalization turns any constant vector, including all zeros, into ones.
  This can manufacture a strong relevance signal where none exists.
- Expansion explores the full two-hop neighborhood before applying its cap; ties
  can depend on set iteration order.
- Isolated/missing graph nodes and mismatched chunk/vector identifiers can fail
  during proximity scoring. Dimensions, finiteness, and normalization are assumed.
- The selector's document bonus can change relevance ordering, rather than only
  breaking ties. It does not implement cluster coverage or redundancy penalties.
- Backfill admits below-floor candidates. This may be a deliberate policy, but
  cannot be described as an absolute precision guarantee.
- Metadata can overwrite canonical IDs/scores in returned records. Input validation
  and a precise formatted-context budget contract are missing.

The new package will implement and verify these steps incrementally. Phase 1
contains only public interfaces, not a claim that these issues are already fixed
in a working retrieval system.
