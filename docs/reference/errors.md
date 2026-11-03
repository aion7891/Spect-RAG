---
description: The exceptions SpectRAG raises, and how to see what it is doing.
---

# Errors and logging

## Exceptions

Every error SpectRAG raises on purpose is a `spectrag.SpectRAGError`. Each one also inherits
the matching built-in exception, so code that already catches `ValueError` or `KeyError` keeps
working.

| Exception | Also a | Raised when |
|---|---|---|
| `SpectRAGError` | `Exception` | (the base class) |
| `SettingsError` | `ValueError` | a setting has the wrong type, is out of range, or is not an allowed choice |
| `UnknownSettingError` | `SettingsError`, `KeyError` | a setting name is unknown, usually a typo; the message suggests the closest name |
| `IndexError_` | `FileNotFoundError` | the index folder is missing or incomplete, was made by a newer version, or the target folder is not a SpectRAG index |
| `EmbedderError` | `ConnectionError` | the embedder cannot be reached (for example Ollama is not running), the query vector does not match the index, or the index is opened with a different embedder from the one that built it (see [Index format](index-format.md#safety)) |
| `DocumentError` | `ValueError` | documents passed to `ingest_documents()` are not `{"doc_id", "text"}` dicts, or two share a `doc_id`; also an evaluation snippet found in no chunk |
| `NoDocumentsError` | `DocumentError` | there is no document with text to ingest |
| `DocumentNotFoundError` | `DocumentError`, `FileNotFoundError` | the file or folder passed to `ingest()` does not exist |
| `MissingPackageError` | `ImportError` | an optional package is needed: `faiss-cpu` or `hnswlib` for `ann_backend`, `pypdf` for PDF files |
| `LLMError` | `ConnectionError` | the Ollama LLM cannot be reached or does not reply, after every retry |

`IndexError_` has a trailing underscore so that it does not hide Python's built-in `IndexError`.

```python
import spectrag
from spectrag import SpectRAGRetrieval

try:
    search = SpectRAGRetrieval("my_index", {"token_budjet": 4000})
except spectrag.UnknownSettingError as error:
    print(error)     # Unknown setting: 'token_budjet' (did you mean 'token_budget'?)
except spectrag.SpectRAGError as error:
    print(error)
```

Examples of messages:

```
selector must be one of spectrag, mmr, protected_mmr, anchor_expand_ce, topk, got 'greedy'
overlap_tokens (1200) must be smaller than chunk_tokens (1000)
No index at my_index. Build one first: SpectRAGIngestion('my_index').ingest(path)
Cannot reach Ollama at http://localhost:11434. Is 'ollama serve' running and 'ollama pull nomic-embed-text' done?
Use rebuild_graph() to change graph settings: graph_k
This index was built with a different embedder (the check sentence has cosine 0.000, below 0.99): open the index with the embedder it was built with (ollama).
This index was built with your own embed function (my_embed). Pass it again: SpectRAGRetrieval(index_dir, embed_fn=...)
```

Two more cases raise `SettingsError`: an index built with your own embed function opened
without `embed_fn` (the message says to pass it again), and a graph setting passed to
`search(settings=...)` (*Use rebuild_graph() to change graph settings*).

The `spectrag` command prints any of these as one line, `spectrag: error: ...`, and exits
with code 1 instead of showing a traceback.

## Logging

SpectRAG is silent by default, like any library. Its messages go to the standard `logging`
logger named `spectrag`.

```python
import spectrag

spectrag.enable_logging()          # INFO: one line per pipeline step, with its time
spectrag.enable_logging("DEBUG")   # also one line per question
```

```
2026-11-03 15:24:59,310 spectrag INFO: chunking: 0.00 s (documents=3, chunks=8)
2026-11-03 15:24:59,311 spectrag INFO: embedding: 0.00 s (backend=exact)
2026-11-03 15:24:59,311 spectrag INFO: edge signals: 0.00 s (semantic=28, adjacency=5, section=3, entity=14, utility=0, citation=3)
2026-11-03 15:24:59,311 spectrag INFO: graph and spectral: 0.00 s (edges=21, dims=7)
2026-11-03 15:24:59,314 spectrag INFO: saved index: 0.00 s (folder=my_index)
```

On the command line, `-v` does the same: `spectrag -v ingest docs/ --index my_index`.

To route the messages through your own logging setup, configure the `spectrag` logger instead
of calling `enable_logging()`:

```python
import logging
logging.getLogger("spectrag").setLevel(logging.INFO)
```
