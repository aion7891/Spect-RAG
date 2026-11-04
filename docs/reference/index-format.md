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
| `settings.json` | the settings the index was built with; your own embed function is saved by name, as `"custom:<name>"` |
| `embedder_probe.npy` | the embedder's fingerprint: the vector of one fixed sentence, checked when the index is opened (since 0.1.2) |
| `info.json` | the [ingestion summary](python-api.md#the-summary), including `format_version` and `spectrafuse_version` |
| `vectors.faiss` or `vectors.hnsw` | the approximate vector index, only with the `faiss` or `hnsw` backend |

## Safety

* **Atomic saves.** Ingestion writes into `<folder>.partial` and moves it into place only when
  every file is written, so a crash never leaves half an index. A folder holding anything other
  than a SpectraFuse index is never overwritten.
* **Safe replace.** An existing index is replaced in three steps: it is renamed to
  `<folder>.old`, `<folder>.partial` is renamed to `<folder>`, and `<folder>.old` is deleted.
  The old index is never deleted before the new one is in place, so a complete index always
  exists (since 0.1.3). If the new index can't be moved into place, the old one is put back.
  If the program stops between the two renames, opening the index reads `<folder>.old` (with a
  warning), and the next ingest renames it back. Opening an index never moves or deletes a
  folder.
* **No code on load.** Arrays are loaded with `allow_pickle=False`, so opening an index never
  runs stored code.
* **Versioned.** `format_version` (currently 2) is checked when an index is opened. A newer
  format than the installed package reads raises `IndexError_` and asks you to upgrade.
* **The same embedder.** Opening an index embeds the fingerprint sentence again and compares
  it with `embedder_probe.npy`. A different size, or a cosine below 0.99, raises
  `EmbedderError`, because vectors from two different models can't be compared, even when
  both have 768 dimensions (for example `hash` and `nomic-embed-text`). The threshold allows
  the small run-to-run differences of a GPU. Indexes made before 0.1.2 have no fingerprint;
  they open as before, with a log message that the embedder was not checked.

## Things to know

* An index built with a custom embed function (a function, a `functools.partial` or a
  callable object) stores only its name in `settings.json`, as `"custom:<name>"`; pass the
  function again with `SpectRAGRetrieval(dir, embed_fn=...)`.
* Don't run two ingests into the same folder at the same time: both write `<folder>.partial`.
  Searches can run while an index is being replaced.
* The files are plain numpy and JSON, so you can read them with any tool:

```python
import json
import numpy as np

chunks = json.load(open("my_index/chunks.json"))
coords = np.load("my_index/spectral.npy")      # shape (chunks, spectral_dims)
```
