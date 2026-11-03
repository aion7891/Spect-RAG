---
description: The spectrag command and every option.
---

# Command line

Installing the package adds the `spectrag` command.

```
spectrag [--version] [-v] <command> ...
```

| Global option | Meaning |
|---|---|
| `--version` | print the version and exit |
| `-v`, `--verbose` | print one timed line per pipeline step (to stderr) |

Every command takes `--index FOLDER`. All commands except `info` take `--set KEY=VALUE`
(repeatable) to change any [setting](../guides/configuration.md). Values are read as JSON, and
anything that is not valid JSON is taken as text:

```bash
--set chunk_tokens=512 --set embedder=ollama --set n_clusters=null
--set 'edge_weights={"semantic": 0.5, "entity": 0.5}'
```

On an error raised by SpectRAG (a bad setting, a missing index), the command prints
`spectrag: error: <message>` and exits with status 1.

## `spectrag ingest`

Build an index from a file or a folder (`.txt`, `.md`, `.pdf`; folders are searched
recursively). Prints the [ingestion summary](python-api.md#the-summary) as JSON.

```bash
spectrag ingest PATH --index FOLDER [--set KEY=VALUE ...]
spectrag ingest docs/ --index my_index --set embedder=ollama --set chunk_tokens=512
```

## `spectrag search`

Print the selected chunks (rank, document, score, tokens, reason), then the context.

```bash
spectrag search QUESTION --index FOLDER [--budget N] [--json] [--set KEY=VALUE ...]
spectrag search "Where is Helios Energy based?" --index my_index --budget 2000
```

| Option | Meaning |
|---|---|
| `--budget N` | token budget (default: the `token_budget` setting, 12,000) |
| `--json` | print `question`, `tokens`, `chunks` and `context` as JSON |

## `spectrag answer`

Search, then answer with a local Ollama model. Prints the answer and the source documents.

```bash
spectrag answer QUESTION --index FOLDER [--budget N] [--model NAME] [--set KEY=VALUE ...]
spectrag answer "Where is Helios Energy based?" --index my_index --model gemma3:4b
```

| Option | Meaning |
|---|---|
| `--budget N` | token budget |
| `--model NAME` | the Ollama model (default `gemma3:4b`) |

For other LLMs, use [`answer()` in Python](../guides/custom-models.md#llms-for-answers).

## `spectrag graph`

Write the interactive [graph view](../guides/graph-view.md).

```bash
spectrag graph --index FOLDER [--out FILE] [--question TEXT] [--max-nodes N] [--set KEY=VALUE ...]
spectrag graph --index my_index --out graph.html --question "Where is Helios Energy based?"
```

| Option | Meaning |
|---|---|
| `--out FILE` | the HTML file to write (default `graph.html`) |
| `--question TEXT` | highlight this question's retrieval path |
| `--max-nodes N` | the most chunks to draw (default 1,500) |

## `spectrag info`

Print what an index contains (its `info.json`): documents, chunks, edges per signal, the weights
used, the backend and the version that built it.

```bash
spectrag info --index my_index
```

## From Python

The command is also available as a function, which returns the exit status:

```python
from spectrag.cli import main
main(["search", "Where is Helios Energy based?", "--index", "my_index", "--json"])
```
