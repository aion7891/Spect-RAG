# SpectRAG

A readable Python implementation of the supplied SpectRAG paper, developed in
reviewable phases. There are two public classes; helper functions live in
`src/spect_rag/utility/`.

**Current status: Phase 1 only.** The paper analysis, implementation plan, and
installable package scaffold are complete. Ingestion and retrieval methods are
explicit placeholders that raise `NotImplementedError`. No documents have been
indexed and no retrieval quality has been measured.

- [Paper analysis and implementation decisions](docs/paper_analysis.md)
- [Phases and review checkpoints](docs/implementation_plan.md)
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
    utility/             # helper functions, added phase by phase
```

The existing retrieval prototype is preserved in
`docs/reference/initial_retrieval_prototype.py`. It is reference material, not
part of the installed package.

## Intended usage after later phases

This example describes the target interface; it cannot run in Phase 1:

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
