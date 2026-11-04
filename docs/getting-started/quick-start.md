---
description: Build an index and get a token-budgeted context for a question in five minutes.
---

# Quick start

SpectraFuse has two steps. **Ingestion** builds an index from your documents, once.
**Retrieval** finds the context for each question, in a few milliseconds.

## From the command line

Put some `.txt`, `.md` or `.pdf` files in a folder, then:

```bash
pip install spectrafuse
spectrafuse ingest docs/ --index my_index
spectrafuse search "Where is Helios Energy based?" --index my_index --budget 300
```

`search` prints the ranked chunks, then the context for your LLM, grouped by source document:

```
 1. helios_energy.txt              score 0.997  97 tokens  (core)
 2. helios_energy.txt              score 0.519  59 tokens  (core)
 3. aurora_labs.md                 score 0.317  113 tokens  (core)

### Source: helios_energy.txt
Helios Energy is a solar power company based in Porto.
...
```

Add `--json` for machine-readable output, or ask an LLM directly (needs
[Ollama](installation.md#embeddings-and-llms-with-ollama-optional)):

```bash
spectrafuse answer "Where is Helios Energy based?" --index my_index --model gemma3:4b
```

All commands and options: [Command line](../reference/cli.md).

## From Python

This example runs as it is, offline, right after `pip install spectrafuse`:

```python
from spectrafuse import SpectRAGIngestion, SpectRAGRetrieval

docs = [
    {"doc_id": "aurora", "text": "# Aurora Labs\nAurora Labs was founded in 2014 in Lisbon by Maya Chen and Tomas Ruiz. "
     "It builds low-cost soil sensors for greenhouse farms.\nIn 2019 Maya Chen left Aurora Labs. "
     "She later joined Helios Energy as its chief technology officer."},
    {"doc_id": "helios", "text": "# Helios Energy\nHelios Energy is a solar power company based in Porto. "
     "It builds solar farms for small towns in southern Europe."},
    {"doc_id": "farming", "text": "# River Valley Farming\nThe River Valley cooperative uses Aurora soil "
     "sensors to save water on 120 farms."},
]

SpectRAGIngestion("my_index").ingest_documents(docs)        # offline, once

search = SpectRAGRetrieval("my_index")                        # online, per question
result = search.search("In which city is the company that hired Aurora's co-founder based?",
                       token_budget=200)
for chunk in result["chunks"]:
    print(chunk["rank"], chunk["doc_id"], round(chunk["score"], 3), chunk["reason"])
print(result["tokens"], "tokens")
print(result["context"])
```

Output:

```
1 helios 0.993 core
2 aurora 0.432 core
3 farming 0.058 core
105 tokens
### Source: helios
# Helios Energy
Helios Energy is a solar power company based in Porto.
It builds solar farms for small towns in southern Europe.

### Source: aurora
# Aurora Labs
...
```

The question needs two facts from two documents: Maya Chen joined Helios Energy (in *aurora*),
and Helios Energy is based in Porto (in *helios*). Both are in the context.

To ingest files instead, pass a file or a folder: `SpectRAGIngestion("my_index").ingest("docs/")`.

### What `search()` returns

| Key | Contents |
|---|---|
| `context` | the selected chunks as one text, one `### Source:` block per document: give this to your LLM |
| `prompt` | a grounded prompt with the context and the question, ready for any LLM |
| `chunks` | the selected chunks, in the order they were picked, each with `rank`, `doc_id`, `text`, `tokens`, `score`, `relevance`, `proximity`, `hop` and `reason` |
| `tokens` | the total tokens of the selected chunks (never more than the budget) |
| `seeds`, `candidates` | chunk ids of the seeds and of every chunk reached by the graph walk |

`reason` says why a chunk was picked: `core` (one of the best-scored chunks, always kept),
`tail` (added by the greedy selector), or `backfill` (fills budget that is left over).

### Get an answer

`answer()` runs `search()` and then any function that takes a prompt and returns text:

```python
from spectrafuse.utils.llm import ollama_llm

result = search.answer("In which city is the company that hired Aurora's co-founder based?",
                       ollama_llm("gemma3:4b"))
print(result["answer"])         # the LLM's answer; result["chunks"] are its sources
```

To use OpenAI, Anthropic or any other API, wrap it in a function: see
[Your own models](../guides/custom-models.md).

## Use real embeddings

The default embedder works offline but matches words, not meaning. With Ollama running:

```python
SpectRAGIngestion("my_index", {"embedder": "ollama"}).ingest("docs/")
search = SpectRAGRetrieval("my_index")      # remembers the embedder the index was built with
```

## The quick-start script

A clone of the repository includes `quickstart.py`. It needs no arguments: it ingests the sample
documents in `examples/docs/` and prints the context for two questions. Every setting is at the
top of the file; set `USE_OLLAMA = True` for nomic embeddings and LLM answers.

```bash
git clone https://github.com/aion7891/SpectraFuse.git && cd SpectraFuse
pip install -e . && python quickstart.py
```

## Next steps

* [Architecture](../concepts/architecture.md): what happens inside each step.
* [Configuration](../guides/configuration.md): the token budget, the selector and 37 more settings.
* [Hyperparameter tuning](../guides/tuning.md): make it better on your own questions.
