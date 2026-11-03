---
description: >-
  Spectral graph-augmented retrieval with token-budgeted context selection for
  multi-hop RAG. One pip install, numpy as the only required dependency.
---

# SpectRAG

![SpectRAG](assets/banner.svg)

SpectRAG retrieves context for retrieval-augmented generation (RAG) when the answer is spread
over several documents. It links every chunk of your documents into a weighted graph, walks that
graph from the chunks that match the question, and packs the best chunks into a fixed token
budget.

```bash
pip install spectrag
```

**Links:** [GitHub](https://github.com/aion7891/Spect-RAG) ·
[PyPI](https://pypi.org/project/spectrag/) ·
[Paper (arXiv:2610.36991)](https://arxiv.org/abs/2610.36991) ·
[Interactive explainers](interactive/index.html)

!!! info

    SpectRAG is an open-source implementation of the paper *SpectRAG: Weighted Chunk Graphs for
    Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation* by Zafer
    Sahinoglu, Abdulazeez Mohammed, Buse Yaren Kazangirler and Cahit Berkay Kazangirler. If you use
    it, please [cite the paper](help/faq.md#how-do-i-cite-spectrag).

<figure markdown="span">
  ![Chunks of 14 documents appear as nodes of a graph, coloured by spectral cluster; links between documents light up; then the command pip install spectrag.](assets/animations/graph-build.gif)
  <figcaption markdown="span">The chunk graph of the 14-document example set, built by `spectrag ingest`. Red links join chunks from different documents.</figcaption>
</figure>

## Why SpectRAG

Most RAG pipelines rank chunks one by one by how similar they are to the question, then keep
the top *k*. That fails on multi-hop questions: the passage that completes the answer often
shares no words with the question, and near-duplicates of the best match fill the context.

| Aspect | Plain top-k RAG | SpectRAG |
|---|---|---|
| **Unit of retrieval** | chunks ranked one by one | a *set* of chunks chosen together |
| **Near-duplicates** | often fill the context | penalised (redundancy term) |
| **Bridge evidence** (not similar to the question, but needed) | usually missed | reached by walking a weighted chunk graph |
| **Structure** (sections, entities, citations, order) | ignored | six edge signals and spectral coordinates |
| **Context size** | a fixed *k* | a token budget, filled by relevance, coverage and diversity |

## What it does, in one picture

![SpectRAG pipeline: offline ingestion builds a chunk graph and its spectral coordinates; online retrieval seeds, expands, re-scores and selects within a token budget.](assets/pipeline.svg)

* **Offline, once per document set** (`SpectRAGIngestion`): split documents into chunks, embed
  them, link them with six signals (meaning, reading order, section, shared entities, the
  questions a chunk answers, citations), and compute each chunk's spectral coordinates.
* **Online, per question** (`SpectRAGRetrieval`): find seed chunks with dense + BM25 search,
  walk the graph two hops, re-score with Personalized PageRank and spectral closeness, and
  select chunks under the token budget.

Read more in [Architecture](concepts/architecture.md), or step through it in the
[interactive walkthrough](interactive/step-by-step.html).

## Features

* **The full method from the paper:** six edge signals, the normalised Laplacian and spectral
  coordinates, hybrid seeds with reciprocal rank fusion, two-hop graph expansion, PageRank and
  spectral proximity, the token-budget selector, and the paper's alternative selectors.
* **Light and offline by default:** `numpy` is the only required dependency. An offline
  embedder works out of the box; plug in `nomic-embed-text` through Ollama, any embedding
  function, and any LLM.
* **Built to scale:** exact, FAISS or HNSW vector search, a sparse eigensolver above 2,000
  chunks, binary index files. Search takes about 4 ms per question on 10,000 chunks.
* **Configurable and tunable:** 39 validated settings, per-question overrides, graph settings
  changed without re-embedding, and a grid / random-search tuner.
* **See what it retrieves:** an interactive HTML view of the chunk graph and of each question's
  retrieval path.
* **Measured, not claimed:** a reproducible benchmark on MultiHopRAG (1,955 test questions)
  with confidence intervals, ablations, latency and memory.

## Results at a glance

On 1,955 MultiHopRAG test questions, with a 12,000-token budget for every method:

| Method | Evidence recall |
|---|---|
| **SpectRAG, tuned on validation questions** | **0.846** |
| Hybrid RAG (dense + BM25) | 0.832 |
| SpectRAG, default settings | 0.803 |
| BM25 | 0.788 |
| Standard RAG (dense top-k) | 0.758 |

!!! warning

    The default settings follow the paper and score **below** hybrid retrieval on this dataset.
    Tune on your own validation questions before relying on them: see
    [Hyperparameter tuning](guides/tuning.md) and [Benchmarks](results/benchmarks.md).

## Where to go next

| If you want to... | Read |
|---|---|
| install the package | [Installation](getting-started/installation.md) |
| get a context for your first question in five minutes | [Quick start](getting-started/quick-start.md) |
| understand the pipeline | [Architecture](concepts/architecture.md) and [The spectral part](concepts/spectral.md) |
| change how it behaves | [Configuration](guides/configuration.md) |
| make it better on your data | [Hyperparameter tuning](guides/tuning.md) |
| use your own embedder or LLM | [Your own models](guides/custom-models.md) |
| look up a class or a command | [Python API](reference/python-api.md) and [Command line](reference/cli.md) |

Python 3.10–3.14 on Linux and macOS are tested. MIT licence.
