# SpectRAG

A readable Python implementation of the supplied SpectRAG paper, developed in
reviewable phases. There are two public classes; helper functions live in
`src/spect_rag/utility/`.

**Current status: Phase 5 complete.** `SpectRAGIngestion.prepare()` loads and
chunks documents. `enrich()` adds normalized embeddings, optional semantic
boundaries, concepts, utility questions, and citation targets using caller-supplied
callbacks. `ingest()` builds and saves the graph, spectrum, FAISS indexes, and
BM25 data. `SpectRAGRetrieval.retrieve()` returns ranked, token-budgeted evidence
through hybrid seeds, graph expansion, and spectral/PageRank scoring. Inspect
`last_trace` to see the stages. Grounded answer generation remains for Phase 6.
Real-model retrieval quality has not been measured.

- [Paper analysis and implementation decisions](docs/paper_analysis.md)
- [Phases and review checkpoints](docs/implementation_plan.md)
- [Phase 2: loading, chunking, and validation](docs/phase2.md)
- [Phase 3: callbacks, enrichment, and caching](docs/phase3.md)
- [Phase 4: graph, spectrum, and index snapshots](docs/phase4.md)
- [Phase 5: retrieval, scoring, and context selection](docs/phase5.md)
- [Supplied research paper](docs/2610.36991v2.pdf)

## Install and inspect

Use Python 3.14 or newer, matching the existing project configuration:

```bash
uv sync
uv run python -c "from spect_rag import SpectRAGIngestion, SpectRAGRetrieval; print(SpectRAGIngestion, SpectRAGRetrieval)"
```

Alternatively, install with `python -m pip install -e .`.

```text
src/spect_rag/
    __init__.py          # exports exactly the two public classes
    ingestion.py         # offline pipeline orchestration
    retrieval.py         # online pipeline orchestration
    utility/
        loading.py       # file loading, PDF extraction, heading/table blocks
        chunking.py      # token counting, sentence windows, chunk provenance
        semantic_chunking.py # adjacent-sentence embedding breakpoints
        embeddings.py    # task prefixes, vector validation and normalization
        extraction.py    # concept/question prompt and response validation
        citations.py     # textual references and local corpus targets
        cache.py         # versioned, atomic JSON caches
        graph_signals.py # six bounded signal extractors
        graph.py         # configurable weighted fusion and sparsification
        spectral.py      # component-aware Laplacian eigenvectors
        validation.py    # shared enriched-record checks
        indexing.py      # versioned snapshots, FAISS, BM25, and reload checks
        ranking.py       # query scores, RRF seeds, and hybrid relevance
        expansion.py     # deterministic bounded two-hop BFS
        proximity.py     # spectral proximity and Personalized PageRank
        selection.py     # protected core, precision tail, coverage, backfill
        retrieval_trace.py # serializable evidence and per-stage trace
```

The existing retrieval prototype is preserved in
`docs/reference/initial_retrieval_prototype.py`. It is reference material, not
part of the installed package.

## Prepare documents now

```python
from spect_rag import SpectRAGIngestion

chunks = SpectRAGIngestion().prepare(["examples/documents/getting_started.md"])
for chunk in chunks:
    print(chunk["chunk_id"], chunk["token_count"], chunk["section_path"])
    print(chunk["text"])
```

Run the example and optionally save all chunk records:

```bash
uv run python examples/prepare_documents.py examples/documents/getting_started.md --output tmp/phase2/sample_chunks.json
uv run python examples/prepare_documents.py docs/2610.36991v2.pdf --output tmp/phase2/paper_chunks.json
uv run python -m unittest discover -s tests -v
```

The first tokenizer use downloads the `cl100k_base` vocabulary; subsequent runs
use tiktoken's cache. Preparation requires no embedding model, LLM, or API key.

Chunks have a hard 1,200-token maximum and a target 100-token overlap using whole
sentences/rows. An oversized sentence/row raises an error rather than being cut.
Markdown headings and numbered PDF headings provide section paths. Sentence
splitting is an English punctuation heuristic; PDF layout/table extraction has
limits described in the [Phase 2 guide](docs/phase2.md). By default `prepare()`
uses sentence windows. Set `semantic_chunking=True` with an embedding callback
to use embedding-based boundaries for unstructured documents.

## Enrich documents now

Configure `embed_fn(list_of_prefixed_texts)` to return a vector per input and
`extract_fn(prompt)` to return a dictionary with `concepts` and
`utility_questions`. The [Phase 3 guide](docs/phase3.md) defines the full contract,
configuration, cache behavior, and returned fields.

The runnable wiring example uses explicitly labelled **fake callbacks**. It
requires no model service or API key and makes no semantic-quality claims:

```bash
uv run python examples/enrich_documents.py examples/documents/getting_started.md examples/documents/storage_notes.txt --demo --output tmp/phase3/enriched_chunks.json
```

Run it twice to see the cache eliminate callback calls. Replace the demo callbacks
with real model clients in your application; fake callbacks are outside the
installed package.

## Build and reload an index now

Using an ingestion instance configured with the callbacks described in the
[Phase 3 guide](docs/phase3.md):

```python
from spect_rag import SpectRAGRetrieval

index = ingestion.ingest(["docs/my_document.pdf", "docs/notes.md"])
print(index["graph"].graph["stats"])
print(index["spectral_coords"].shape)

reloaded = SpectRAGRetrieval(ingestion.index_dir).load()
print(reloaded["manifest"]["counts"])
```

Runnable demo with test callbacks:

```bash
uv run python examples/ingest_documents.py examples/documents/getting_started.md examples/documents/storage_notes.txt --demo --index-dir indexes/demo
```

Each rebuild publishes a new snapshot under the index directory and retains
older snapshots. The manifest records models, settings, row order, and checksums;
loading rejects missing or inconsistent artifacts. See the [Phase 4 guide](docs/phase4.md)
for signal formulas, neighbor limits, spectral conventions, and the file layout.

## Retrieve evidence now

Configure retrieval with the same embedding model/callback used for ingestion.
The callback receives the query task prefix; no generation LLM is required.
The dimension is read from the saved index unless explicitly supplied:

```python
from spect_rag import SpectRAGRetrieval

# my_embed is your batch embedding adapter, described in the Phase 3 guide.
retrieval = SpectRAGRetrieval(
    "indexes/my_docs", embed_fn=my_embed,
    embedding_model="nomic-embed-text@your-revision",
)
evidence = retrieval.retrieve("How are these topics connected?", token_budget=12_000)
for item in evidence:
    print(item["rank"], item["chunk_id"], item["score"], item["text"])
print(retrieval.last_trace)
```

After building the demo index above, run:

```bash
uv run python examples/retrieve_documents.py "What does a battery store?" --demo --index-dir indexes/demo --seed-k 2 --max-candidates 5 --token-budget 80 --trace-output tmp/phase5/trace.json
```

The budget covers the sum of selected chunk-text token counts, including any
overlap already present in those chunks. It does not include future prompt
headers, query, instructions, or answer space. The default backfill can admit
positive-score chunks below the tail floor; use `backfill=False` to disable it.
See the [Phase 5 guide](docs/phase5.md) for exact policies and defaults.

Retrievers pin a verified snapshot for repeated queries. Use `load()` to refresh
after rebuilding the index. Answer generation remains for Phase 6.

Embedding and extraction callbacks are now configurable on ingestion.
The proposed first implementation uses local files and NetworkX, avoiding a
Neo4j service requirement for testing. That is a documented storage adaptation
of the paper and is part of the Phase 1 design review.

Each completed phase stops for your review before the next phase begins.
