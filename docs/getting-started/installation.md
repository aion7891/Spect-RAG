---
description: Install SpectRAG from PyPI, choose the extras you need, and set up Ollama.
---

# Installation

## Install the package

=== "pip"

    ```bash
    pip install spectrag                # core: numpy only
    pip install "spectrag[all]"         # every optional extra
    ```

=== "uv"

    ```bash
    uv add spectrag                     # core: numpy only
    uv add "spectrag[all]"              # every optional extra
    ```

=== "From source"

    ```bash
    git clone https://github.com/aion7891/Spect-RAG.git && cd Spect-RAG
    pip install -e ".[all,dev]"         # or: uv sync --all-extras
    ```

Check that it works:

```bash
spectrag --version                  # spectrag 0.1.2
```

## Optional extras

The core package needs only `numpy`. Extras add features you may not need:

| Extra | Installs | Adds |
|---|---|---|
| `tokens` | `tiktoken` | exact `cl100k_base` token counts (otherwise a word-based estimate) |
| `pdf` | `pypdf` | reading `.pdf` files |
| `scale` | `scipy`, `faiss-cpu`, `hnswlib` | a sparse eigensolver and approximate nearest-neighbour search, for large corpora |
| `all` | all of the above | |
| `dev` | `pytest`, `build` | running the tests and building the package (from a clone) |

Install several at once with `pip install "spectrag[tokens,pdf]"`.

!!! info

    **Which extras do I need?** Install `tokens` if the token budget must match OpenAI-style token
    counts exactly, `pdf` to ingest PDF files, and `scale` once your corpus has more than a few
    thousand chunks (see [Large corpora](../guides/large-corpora.md)).

## Embeddings and LLMs with Ollama (optional)

SpectRAG runs offline by default, with a hashing embedder that needs no model. For real
semantic search, and for LLM answers, it can use a local [Ollama](https://ollama.com) server:

```bash
ollama pull nomic-embed-text        # the paper's embedding model (768 dimensions)
ollama pull gemma3:4b               # any chat model, for answers and utility questions
```

Then choose the Ollama embedder when you build an index:

```bash
spectrag ingest docs/ --index my_index --set embedder=ollama
```

```python
SpectRAGIngestion("my_index", {"embedder": "ollama"}).ingest("docs/")
```

!!! warning

    The offline `hash` embedder matches words, not meaning. It is good for trying the package and
    for tests, but use `ollama` or [your own embedder](../guides/custom-models.md) for real
    retrieval quality.

You can use any other embedding model or LLM instead: see
[Your own models](../guides/custom-models.md).

## Supported platforms

| Platform | Supported |
|---|---|
| Python | 3.10, 3.11, 3.12, 3.13, 3.14 (CI tests 3.10, 3.12 and 3.14). There is no upper version limit; Python 3.15 waits for a numpy release that supports it. |
| Operating systems | Linux and macOS are tested in CI |
| Document formats | `.txt`, `.md`, `.markdown`, and `.pdf` (with the `pdf` extra) |

## Next step

Build your first index: [Quick start](quick-start.md).
