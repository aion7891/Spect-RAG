# SpectRAG

A readable Python implementation of the supplied SpectRAG paper, developed in
reviewable phases. There are two public classes; helper functions live in
`src/spect_rag/utility/`.

**Current status: Phase 2 complete.** You can load and inspect token-aware chunks
from text, Markdown, and machine-readable PDFs through
`SpectRAGIngestion.prepare()`. Full indexing (`ingest`) and retrieval (`retrieve`)
remain unimplemented. No retrieval quality has been measured.

- [Paper analysis and implementation decisions](docs/paper_analysis.md)
- [Phases and review checkpoints](docs/implementation_plan.md)
- [Phase 2: loading, chunking, and validation](docs/phase2.md)
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
limits described in the [Phase 2 guide](docs/phase2.md). Unstructured text uses
sentence windows; embedding-based semantic chunking comes in Phase 3.

## Intended usage after later phases

This describes the target interface; full indexing/retrieval cannot run yet:

```python
from spect_rag import SpectRAGIngestion, SpectRAGRetrieval

ingestion = SpectRAGIngestion(index_dir="indexes/my_docs")
ingestion.ingest(["docs/my_document.pdf", "docs/notes.md"])

retrieval = SpectRAGRetrieval(index_dir="indexes/my_docs")
evidence = retrieval.retrieve("How are the two topics connected?", token_budget=12_000)
for chunk in evidence:
    print(chunk["chunk_id"], chunk["score"], chunk["text"])
```

Embedding and LLM callbacks will be added when those steps are implemented.
The proposed first implementation uses local files and NetworkX, avoiding a
Neo4j service requirement for testing. That is a documented storage adaptation
of the paper and is part of the Phase 1 design review.

Each completed phase stops for your review before the next phase begins.
