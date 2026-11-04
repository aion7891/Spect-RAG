---
description: Every public class and function, with its arguments and what it returns.
---

# Python API

```python
from spectrag import (SpectRAGIngestion, SpectRAGRetrieval, make_settings, DEFAULTS,
                      enable_logging, __version__)
```

## SpectRAGIngestion

The offline pipeline (paper Table 1): load → chunk → embed → edge signals → weighted graph →
spectral coordinates → save. Run it once per document set.

```python
SpectRAGIngestion(index_dir, settings=None, entity_fn=None, question_fn=None)
```

| Argument | Meaning |
|---|---|
| `index_dir` | the folder to write the index to. An existing SpectRAG index there is replaced; any other non-empty folder raises `IndexError_` |
| `settings` | a dict of [settings](../guides/configuration.md) to change |
| `entity_fn` | optional `text -> list of entities`; the default is rule-based ([details](../guides/custom-models.md#entities)) |
| `question_fn` | optional `text -> list of questions`, for utility edges ([details](../guides/custom-models.md#utility-questions)) |

Raises `SettingsError` or `UnknownSettingError` for invalid settings.

### `ingest(path)`

Ingest a file or a folder (searched recursively for `.txt`, `.md`, `.markdown` and `.pdf`),
save the index, and return a summary. Each file is one document; its `doc_id` is its path
relative to the folder.

```python
info = SpectRAGIngestion("my_index").ingest("docs/")
```

Raises `NoDocumentsError` if no file with text is found.

### `ingest_documents(docs)`

Ingest documents that are already in memory: a list of `{"doc_id": ..., "text": ...}` dicts.
Every `doc_id` must be unique. Raises `DocumentError` for a malformed list and `NoDocumentsError`
if every text is empty.

```python
info = SpectRAGIngestion("my_index").ingest_documents([{"doc_id": "a", "text": "..."}])
```

### The summary

Both methods return the dict that is also saved as `info.json`:

| Key | Meaning |
|---|---|
| `documents`, `chunks` | how many of each |
| `signal_edges` | edges per signal, before fusion: `{"semantic": 28, "entity": 14, ...}` |
| `graph_edges` | edges in the fused graph |
| `weights_used` | the signal weights after empty signals shared theirs |
| `spectral_dims`, `clusters` | coordinates per chunk and spectral clusters (fewer than asked for on very small corpora) |
| `ann_backend` | the vector-search backend used: `exact`, `faiss` or `hnsw` |
| `format_version`, `spectrag_version` | the index format (2) and the package version that wrote it |
| `seconds` | how long ingestion took |

### Lower-level steps

`make_chunks(docs)` returns the chunks with their entities, references and utility questions;
`make_graph(signals)` fuses edge signals into `(edges, weights)`; `make_spectral(edges,
node_count)` returns `(coords, eigenvalues, clusters)`. `ingest_documents()` calls them in turn.

## SpectRAGRetrieval

The online pipeline (paper Table 2, Algorithm 1): seeds → graph expansion → proximity
re-scoring → token-budget selection → context. Create it once, then search many times.

```python
SpectRAGRetrieval(index_dir, settings=None, embed_fn=None, rerank_fn=None)
```

| Argument | Meaning |
|---|---|
| `index_dir` | an index made by `SpectRAGIngestion` |
| `settings` | settings to change. The object starts from the settings the index was built with. Graph settings rebuild the graph in memory from the saved signals |
| `embed_fn` | required if the index was built with your own embed function: pass the same function again |
| `rerank_fn` | optional `(question, texts) -> scores`, used by `selector="anchor_expand_ce"` |

Raises `IndexError_` if the folder is missing, incomplete, or written by a newer index format.

### `search(question, token_budget=None, settings=None, query_vector=None)`

Find the context for a question.

| Argument | Meaning |
|---|---|
| `question` | the question text |
| `token_budget` | tokens of context; `None` uses the `token_budget` setting (12,000) |
| `settings` | overrides for this call only, e.g. `{"selector": "mmr", "alpha": 0.5}`; graph settings are not allowed here |
| `query_vector` | an embedding of the question, if you already have one (skips embedding) |

Returns a dict:

| Key | Contents |
|---|---|
| `question` | the question |
| `context` | the selected chunks as one text, grouped by document (`### Source: <doc_id>` blocks; documents ordered by their best chunk, chunks in reading order) |
| `prompt` | a grounded prompt containing the context and the question |
| `chunks` | the selected chunks, in the order they were picked (see below) |
| `tokens` | the tokens used, never more than the budget |
| `seeds` | chunk ids of the seeds |
| `candidates` | chunk ids of every chunk the graph walk reached |

Each item of `chunks`:

| Key | Contents |
|---|---|
| `id`, `doc_id`, `position`, `section`, `tokens`, `text` | the chunk: its id, document, place in the document, section path (a list of headings), size and text |
| `rank` | 1 for the first chunk picked |
| `reason` | `core`, `tail` or `backfill` (the `spectrag` selector); `mmr`; `anchor` or `expand` (`anchor_expand_ce`) |
| `hop` | 0 for a seed, 1 or 2 for chunks reached through the graph |
| `score` | the re-scored value: (1 − α)·relevance + α·proximity |
| `relevance`, `proximity` | the two parts of the score, each 0…1 |

### `answer(question, llm_fn, token_budget=None, settings=None)`

`search()`, then `llm_fn(result["prompt"])`. Returns the search result with an extra `answer` key.
`llm_fn` is any function `prompt -> text`, e.g. `spectrag.utils.llm.ollama_llm("gemma3:4b")`.

### `rebuild_graph(overrides=None)`

Change graph settings (`edge_weights`, `edge_threshold`, `graph_k`, `spectral_dims`,
`n_clusters`) without re-embedding: the saved signals are fused again and the spectral
coordinates recomputed. In memory only; the index on disk is unchanged.

```python
search.rebuild_graph({"graph_k": 5, "edge_threshold": 0.1})
```

### `retrieve(question, settings=None, query_vector=None)`

Stages 1–4 without formatting, for inspection and evaluation. Returns `(seeds, scored, picks)`:

* `seeds`: `seeds` (ids), `relevance`, `dense`, `bm25`, `utility` and `rrf` scores for every chunk,
  and the `rankings` that were fused.
* `scored`: `candidates` (ids), and aligned with them `hop`, `relevance`, `ppr`, `spectral`,
  `proximity` and `scores`.
* `picks`: `[(candidate number, reason), ...]` in pick order; `scored["candidates"][k]` is the
  chunk id.

### Attributes

| Attribute | Contents |
|---|---|
| `settings` | the settings in use |
| `index` | the loaded index: `chunks`, `embeddings`, `edges`, `spectral`, `eigenvalues`, `info`, … |
| `embed` | the embedder, as `embed(texts, kind="query")` |

## Settings helpers

| Name | Meaning |
|---|---|
| `DEFAULTS` | the default settings (a dict; do not modify it) |
| `make_settings(overrides=None)` | a fresh copy of the defaults with your changes, checked; raises `UnknownSettingError` or `SettingsError` |
| `spectrag.config.EDGE_WEIGHTS_CONFIG_1`, `EDGE_WEIGHTS_CONFIG_2` | the paper's two edge-weight sets (1 is the default) |

## Logging and version

| Name | Meaning |
|---|---|
| `enable_logging(level="INFO")` | print SpectRAG's messages to stderr: one timed line per pipeline step; `"DEBUG"` adds one line per question |
| `__version__` | the package version, e.g. `"0.1.4"` |
| `INDEX_FORMAT` | the index format this version writes and reads (2) |

## Utilities

| Function | Module | Meaning |
|---|---|---|
| `ollama_llm(model="gemma3:4b", url=..., num_ctx=None, timeout=600, max_tokens=None, retries=2)` | `spectrag.utils.llm` | an `llm_fn` for a local Ollama model, temperature 0 |
| `make_question_fn(llm_fn, count=3)` | `spectrag.utils.questions` | a `question_fn` that asks an LLM for utility questions |
| `save_graph_view(index_dir, path="graph.html", question=None, settings=None, max_nodes=1500)` | `spectrag.utils.graph_view` | write the [graph view](../guides/graph-view.md); returns the path |
| `graph_data(index_dir, question=None, settings=None, max_nodes=1500)` | `spectrag.utils.graph_view` | the graph view's data as Python dicts |
| `tune(search, questions, grid, objective="recall", budget=None, trials=None, seed=0)` | `spectrag.utils.tuning` | [hyperparameter search](../guides/tuning.md); returns trials, best first |
| `load_questions(path)` | `spectrag.utils.evaluation` | read a questions JSON file |
| `evaluate_retrieval(search, questions, budget)` | `spectrag.utils.evaluation` | SpectRAG against dense and hybrid top-k with the same budget; returns `rows` and `summary` |
| `evaluate_answers(search, questions, budget, llm_fn)` | `spectrag.utils.evaluation` | the same for answers (needs `answers` in each question): exact match, token F1, contains |
| `build_prompt(question, context, short=False)` | `spectrag.utils.prompt` | the grounded prompt; `short=True` asks for a bare answer |
| `count_tokens(text)` | `spectrag.utils.tokens` | the token count SpectRAG uses (`tiktoken` with the `tokens` extra, otherwise an estimate) |

Every module in `spectrag.utils` is short and does one step; read the source for the
lower-level functions. The [source on GitHub](https://github.com/aion7891/Spect-RAG/tree/main/src/spectrag)
links each one to its section or equation in the paper.
