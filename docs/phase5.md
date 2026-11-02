# Phase 5: retrieval and context selection

This phase implements the retrieval portion of Algorithm 1, steps 1-12 and the
ranked evidence output. It does not call a generation LLM. Source grouping,
grounded answer prompting, and labeled evaluation remain for Phase 6.

There are still exactly two classes: ingestion and retrieval. Every helper stays
under `src/spect_rag/utility/`.

## Read the implementation in this order

1. `src/spect_rag/retrieval.py`: validates inputs, loads the index, embeds the
   query, and calls the following stages in order.
2. `utility/ranking.py`: exact FAISS scores, BM25, positive-only RRF seed fusion,
   and dense/BM25 relevance for every chunk.
3. `utility/expansion.py`: deterministic two-hop expansion with a bounded heap.
4. `utility/proximity.py`: spectral support, seed personalization, weighted PPR,
   spectral closeness, and the relevance/proximity blend.
5. `utility/selection.py`: protected core, tail floor, document-coverage tie-break,
   and optional backfill under the evidence token budget.
6. `utility/retrieval_trace.py`: ranked source records and JSON-serializable trace.

## Configure retrieval

Use your real batch embedding adapter and the same explicit model identity as
the ingestion index:

```python
from spect_rag import SpectRAGRetrieval

retrieval = SpectRAGRetrieval(
    "indexes/my_docs",
    embed_fn=my_embed,
    embedding_model="nomic-embed-text@your-revision",
    query_prefix="search_query: ",
    cache_dir="tmp/my-model-cache",
)
evidence = retrieval.retrieve("How are these topics connected?", token_budget=12_000)
trace = retrieval.last_trace
```

`my_embed` is your client adapter, not a built-in service. Its input is already
prefixed, and it returns one vector per input in order. Query vectors receive
the same dimensionality/finiteness/nonzero/normalization checks as ingestion.
The dimension comes from the saved index unless `embedding_dimension` is supplied.
An explicitly supplied dimension or model mismatch fails before a callback call.
`document_prefix` optionally asserts the index's document-prefix setting;
`query_prefix` independently controls the query task label and may be empty for
models that do not use task prefixes.

Loading needs no original source documents or extraction/generation model.
`SpectRAGRetrieval(path).load()` still works for model-free index inspection.
The first positive-budget query lazily loads and verifies the index. Repeated
queries reuse that snapshot; `load()` explicitly refreshes CURRENT after an
ingestion rebuild and clears the previous trace. Constructors perform no model
calls or file writes. Query caching is opt-in via `cache_dir`.

## 1. Hybrid seeds and relevance

Dense search uses the saved exact FAISS `IndexFlatIP`. All chunk scores are
retrieved so an expanded chunk still has direct query relevance. Utility-question
search uses its own exact index and takes the maximum positive question cosine
for each owning chunk. A chunk without questions has no question vote.

BM25 uses the saved/rebuilt index and the same Unicode word tokenizer used during
ingestion. Scores can be zero or negative for tiny/common-term corpora. Negative
BM25 and cosine scores are clipped to zero for ranking/relevance; raw dense and
BM25 values remain in the trace. `utility_raw` is already the positive per-chunk
question maximum, or zero when absent.

Each positive entry receives its 1-based channel rank, with corpus row order
breaking ties. Seed fusion is:

`RRF(c) = sum(1 / (rrf_k + rank(channel, c)))`

Only participating positive entries vote. A missing/zero channel does not receive
a tied full-corpus ranking or a substitute dense vote. Keep up to `seed_k` positive
RRF entries. If every channel has no positive score, return no seeds/evidence.

This positive-entry gate is an explicit package policy. The paper does not fully
specify nonpositive scores or unavailable channels; it should not be read as a
calibrated answerability detector. Real embeddings can match unrelated text.

For the full-corpus relevance vector, max-scale positive dense and BM25 scores
separately. Their mean is used when both have positive signal; one available
signal receives full weight if the other is absent. Questions influence seeds,
not this direct relevance vector, matching the paper's dense/BM25 hybrid design.
Max scaling preserves constant positive signals; all-zero vectors remain zero.
No candidate-only re-normalization changes these values later.

## 2. Bounded graph expansion

Every seed is retained. Two BFS levels add neighbors using this priority:

1. Higher initial hybrid relevance.
2. Stronger edge to the current frontier.
3. Earlier corpus row.

A bounded heap retains only the best remaining-capacity nodes at each level.
Repeated neighbor appearances update the retained strongest frontier edge.
Stale heap entries are compacted so the neighbor buffer stays bounded. The full
two-hop neighborhood is not materialized before truncation. Scanning high-degree
adjacency still costs time; the cap bounds candidate/buffer size, not all work.

This is capped BFS, not a globally optimized top set over the full two-hop union.
A large first hop can fill the cap before the second hop is visited. The returned
weighted subgraph includes all retained seeds and any isolates. Trace hops are
0 for seeds, 1 or 2 for discovered chunks.

## 3. Spectral/PPR re-scoring

Seed personalization weights are proportional to initial hybrid relevance. If
all seed relevance is zero, as can happen with question-only seeds, use uniform
seed weights. Non-seeds have zero teleport mass.

Personalized PageRank runs on the real weighted candidate subgraph with damping
0.85, the same teleport/dangling distribution, tolerance `1e-10`, and at most
1,000 iterations. Starting from the personalization distribution keeps disconnected
zero-mass components exactly zero. Non-convergence raises an error. An edgeless
subgraph directly uses its seed distribution.

Spectral closeness uses a unit-scale RBF over saved eigenvector rows:

`spectral_raw(c) = sum(seed_weight(s) * exp(-||z_c - z_s||²))`

Only seeds in the same supported full-graph component contribute. Isolates and
components absent from the retained positive basis have no spectral evidence.
A zero row *within* a supported component can be a valid coordinate and is not
discarded just for being zero.

Max-scale PPR and spectral values over candidates. Proximity is their mean where
spectral evidence is available; otherwise it is PPR alone. Then:

`score(c) = (1 - alpha) * relevance(c) + alpha * proximity(c)`

This follows Algorithm 1's hybrid proximity. It is not exact heat-kernel diffusion
from Equation 17. Kernel, normalization, seed weighting, and convergence settings
are documented implementation choices where the manuscript lacks full detail.

## 4. Token-budgeted context

The budget is exactly `sum(selected_chunk.token_count) <= token_budget`, matching
Equation 19. Each count covers stored chunk text, including overlaps already in
that chunk. Instructions, query, formatting separators, source headers, and answer
space are not part of this budget; Phase 6 must reserve those separately.

Selection uses positive final scores only:

- **Core:** keep up to `core_size` highest-scoring chunks that fit. Oversized
  chunks are skipped intact. Core chunks may be below the tail floor.
- **Tail:** repeatedly choose fitting chunks with score at least `precision_floor`.
  Document coverage only breaks score ties within `1e-9`; it never adds a bonus
  that can displace meaningfully higher relevance.
- **Backfill:** when enabled, add remaining fitting positive-score chunks in score
  order, including below-floor chunks. It never adds zero-score filler.

The floor is a composite-score threshold, not measured statistical precision.
For an absolute floor across every selected chunk, use `core_size=0` and
`backfill=False`. A small budget may return no chunks; text is never truncated.
Unused budget is permitted when no eligible whole chunk fits.

Returned records are ordered by final score, with corpus row tie-breaks. They
also carry `selection_order` so the distinct context-building order is inspectable.
This baseline does not implement cluster coverage, semantic MMR, or the full
Equation 21 objective. Those remain optional extensions beyond the chosen baseline.

## Defaults and output

| `retrieve()` argument | Default |
| --- | ---: |
| `token_budget` | 12000 |
| `seed_k` | 20 |
| `max_candidates` | 200 |
| `rrf_k` | 60 |
| `alpha` | 0.4 |
| `core_size` | 5 |
| `precision_floor` | 0.35 |
| `backfill` | true |

The paper specifies the nominal 20 seeds, 200 candidates, two hops, approximately
12k tokens, and mean spectral/PPR proximity. RRF constant, alpha, core size, floor,
normalization, and missing-signal policies are package choices. `max_candidates`
must be at least `seed_k`. Integers/weights/booleans are checked explicitly.
Zero budget returns an empty trace/result without loading the index or calling a
model. Empty/non-string queries are rejected.

Evidence retains source text, chunk/document identity, page/section provenance,
concepts/questions/citations, and token counts. New fields include `rank`, `score`,
`relevance_score`, `ppr_raw`, `ppr_score`, `spectral_raw`, `spectral_score`,
`spectral_available`, `proximity_score`, `selection_order`, and
`selection_reason` (`core`, `tail`, or `backfill`). Raw embedding arrays are omitted
so evidence is JSON-serializable.

`last_trace` contains settings, pinned snapshot generation, active channels and
their top seed-k rankings, seed/candidate IDs, per-candidate raw channel/RRF/final
scores, graph hops, seed mass, selection/exclusion reasons, selection order, and
used/unused text tokens. Exclusion reasons reflect the budget remaining after
selection. Candidate details are bounded by the cap; this is not a copy of every
corpus vector. A failed query clears the previous trace.

## Run and review

Build the demo index if needed, then retrieve:

```bash
uv run python examples/ingest_documents.py examples/documents/getting_started.md examples/documents/storage_notes.txt --demo --index-dir indexes/demo
uv run python examples/retrieve_documents.py "What does a battery store?" --demo --index-dir indexes/demo --seed-k 2 --max-candidates 5 --token-budget 80 --trace-output tmp/phase5/trace.json
uv run python -m unittest discover -s tests -v
```

The callbacks are explicitly fake word-hash vectors/keyword rules, not real
semantic models. The two-document run selected three complete chunks using
exactly 80 tokens, with dense, BM25, and utility channels active. A run against the
saved paper index used 1,160 of 1,200 tokens, three seeds, and 20 candidates.
These are integration/budget checks, not retrieval-quality measurements.

The 90-test suite includes hand-computed RRF, PageRank and spectral proximity,
two-hop bridge recovery, deterministic caps/ties, missing/negative/zero signals,
spectral fallbacks, core/floor/backfill behavior, coverage tie-breaks, token limits,
query cache/model compatibility, snapshot refreshes, and serializable traces.

Stop here for review. Phase 6 adds optional grounded generation and labeled
retrieval evaluation, with actual observed metrics reported separately.
