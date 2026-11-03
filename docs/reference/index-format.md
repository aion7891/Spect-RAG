---
description: What an index folder contains, and how it is saved and versioned.
---

# Index format

An index is a folder. Each part of the index is one file: numpy arrays are `.npy`, the six raw
edge signals are one `.npz`, and everything else is JSON.

| File | Contents |
|---|---|
| `chunks.json` | every chunk: `id`, `doc_id`, `position`, `section`, `mode` (`structural` or `semantic`), `text`, `tokens`, `entities`, `references`, `utility_questions`, `cluster` |
| `embeddings.npy` | the chunk embeddings, one L2-normalised row per chunk |
| `edges.npy` | the fused graph: one row `[i, j, weight]` per edge |
| `signals.npz` | the six raw edge signals before fusion; used by `rebuild_graph()` and tuning |
| `spectral.npy`, `eigenvalues.npy` | the spectral coordinates (one row per chunk) and their eigenvalues |
| `question_vectors.npy`, `question_owners.npy` | embeddings of the utility questions, and the chunk each belongs to |
| `settings.json` | the settings the index was built with |
| `info.json` | the [ingestion summary](python-api.md#the-summary), including `format_version` and `spectrag_version` |
| `vectors.faiss` or `vectors.hnsw` | the approximate vector index, only with the `faiss` or `hnsw` backend |

## Safety

* **Atomic saves.** Ingestion writes into `<folder>.partial` and moves it into place only when
  every file is written, so a crash never leaves half an index. An existing SpectRAG index in
  the target folder is replaced; a folder holding anything else is never overwritten.
* **No code on load.** Arrays are loaded with `allow_pickle=False`, so opening an index never
  runs stored code.
* **Versioned.** `format_version` (currently 2) is checked when an index is opened. A newer
  format than the installed package reads raises `IndexError_` and asks you to upgrade.

## Things to know

* An index built with a custom embed function stores only the function's name in
  `settings.json`; pass the function again with `SpectRAGRetrieval(dir, embed_fn=...)`.
* The files are plain numpy and JSON, so you can read them with any tool:

```python
import json
import numpy as np

chunks = json.load(open("my_index/chunks.json"))
coords = np.load("my_index/spectral.npy")      # shape (chunks, spectral_dims)
```
