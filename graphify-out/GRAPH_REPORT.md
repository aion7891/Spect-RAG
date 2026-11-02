# Graph Report - docs  (2026-11-03)

## Corpus Check
- Corpus: one 25-page PDF; the detector did not measure PDF word count.

## Summary
- 29 nodes · 39 edges · 5 communities detected
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.88)
- Token usage: unavailable; numeric zeros in machine-readable metadata are placeholders.

## Community Hubs (Navigation)
- [[_COMMUNITY_Chunking and Edge Signals|Chunking and Edge Signals]]
- [[_COMMUNITY_Offline Spectral Indexing|Offline Spectral Indexing]]
- [[_COMMUNITY_Hybrid Seed Retrieval|Hybrid Seed Retrieval]]
- [[_COMMUNITY_Context Selection and Generation|Context Selection and Generation]]
- [[_COMMUNITY_Expansion and Relevance Propagation|Expansion and Relevance Propagation]]

## God Nodes (most connected - your core abstractions)
1. `Offline indexing` - 7 edges
2. `Six-signal weighted fusion and sparsification` - 7 edges
3. `Hybrid seed retrieval` - 6 edges
4. `Existing SpectRAG retrieval prototype` - 5 edges
5. `Weighted chunk graph` - 4 edges
6. `Symmetric normalized Laplacian` - 4 edges
7. `Two-hop BFS with candidate cap` - 4 edges
8. `Relevance and proximity blending` - 4 edges
9. `Token-budget context selection` - 4 edges
10. `Nontrivial low-frequency spectral coordinates` - 3 edges

## Surprising Connections (you probably didn't know these)
- `Existing SpectRAG retrieval prototype` --implements_approximation_of--> `Hybrid seed retrieval`  [INFERRED]
  docs/reference/initial_retrieval_prototype.py → docs/2610.36991v2.pdf
- `Existing SpectRAG retrieval prototype` --implements_approximation_of--> `Two-hop BFS with candidate cap`  [INFERRED]
  docs/reference/initial_retrieval_prototype.py → docs/2610.36991v2.pdf
- `Existing SpectRAG retrieval prototype` --implements_approximation_of--> `Document-grouped grounded generation`  [INFERRED]
  docs/reference/initial_retrieval_prototype.py → docs/2610.36991v2.pdf
- `Existing SpectRAG retrieval prototype` --implements_approximation_of--> `Relevance and proximity blending`  [INFERRED]
  docs/reference/initial_retrieval_prototype.py → docs/2610.36991v2.pdf
- `Existing SpectRAG retrieval prototype` --implements_approximation_of--> `Token-budget context selection`  [INFERRED]
  docs/reference/initial_retrieval_prototype.py → docs/2610.36991v2.pdf

## Communities

### Community 0 - "Chunking and Edge Signals"
Cohesion: 0.25
Nodes (8): Document adjacency edges, Structure-aware and semantic chunking, Citation and dependency edges, Nomic dense embeddings, Six-signal weighted fusion and sparsification, Entity and concept overlap edges, Section hierarchy edges, Semantic similarity edges

### Community 1 - "Offline Spectral Indexing"
Cohesion: 0.47
Nodes (6): Symmetric normalized Laplacian, Offline indexing, Graph and index persistence, SpectRAG, Nontrivial low-frequency spectral coordinates, Weighted chunk graph

### Community 2 - "Hybrid Seed Retrieval"
Cohesion: 0.33
Nodes (6): BM25 lexical retrieval, Exact FAISS IndexFlatIP search, Hybrid seed retrieval, Reciprocal Rank Fusion, Query-to-utility-question matching, Utility-question similarity edges

### Community 3 - "Context Selection and Generation"
Cohesion: 0.47
Nodes (6): Context-set relevance diversity coverage and connectivity, Document-grouped grounded generation, Relevance and proximity blending, Spectral eigenspace closeness, Token-budget context selection, Existing SpectRAG retrieval prototype

### Community 4 - "Expansion and Relevance Propagation"
Cohesion: 0.67
Nodes (3): Two-hop BFS with candidate cap, Heat-kernel diffusion formulation, Personalized PageRank

## Knowledge Gaps
- **9 isolated node(s):** `SpectRAG`, `Document adjacency edges`, `Entity and concept overlap edges`, `Citation and dependency edges`, `Graph and index persistence` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Weighted chunk graph` connect `Offline Spectral Indexing` to `Chunking and Edge Signals`, `Expansion and Relevance Propagation`?**
  _High betweenness centrality (0.396) - this node is a cross-community bridge._
- **Why does `Six-signal weighted fusion and sparsification` connect `Chunking and Edge Signals` to `Offline Spectral Indexing`, `Hybrid Seed Retrieval`?**
  _High betweenness centrality (0.330) - this node is a cross-community bridge._
- **Why does `Two-hop BFS with candidate cap` connect `Expansion and Relevance Propagation` to `Offline Spectral Indexing`, `Hybrid Seed Retrieval`, `Context Selection and Generation`?**
  _High betweenness centrality (0.317) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Existing SpectRAG retrieval prototype` (e.g. with `Hybrid seed retrieval` and `Two-hop BFS with candidate cap`) actually correct?**
  _`Existing SpectRAG retrieval prototype` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `SpectRAG`, `Document adjacency edges`, `Entity and concept overlap edges` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
## Measurement and Scope Notes

Token usage was unavailable for this extraction. The zero token counts in generated metadata are placeholders, not measured zero cost. PDF word count was not measured by the detector; its reported zero does not describe the actual paper length. This graph maps the supplied research paper and includes one explicitly identified comparison node for the existing retrieval prototype; it is not an exhaustive source-code graph.
