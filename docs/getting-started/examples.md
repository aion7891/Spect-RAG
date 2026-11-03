---
description: Runnable scripts for every stage of the pipeline, and a small evaluation set.
---

# Examples

The repository's [`examples/`](https://github.com/aion7891/Spect-RAG/tree/main/examples) folder
has one short script per stage of the pipeline. Each runs offline with no arguments; add
`ollama` (and a model name) to use real embeddings and an LLM. Run them from a clone:

```bash
git clone https://github.com/aion7891/Spect-RAG.git && cd Spect-RAG
pip install -e ".[all]"
python examples/search_example.py
```

## One script per stage

| Script | Shows | Run with Ollama |
|---|---|---|
| [`load_example.py`](https://github.com/aion7891/Spect-RAG/blob/main/examples/load_example.py) | loading documents, counting tokens and sentences | |
| [`chunk_example.py`](https://github.com/aion7891/Spect-RAG/blob/main/examples/chunk_example.py) | chunking; `60 20` shows packing and overlap with small chunks | |
| [`embed_example.py`](https://github.com/aion7891/Spect-RAG/blob/main/examples/embed_example.py) | embedding, saving, loading and a plain dense search | `ollama` |
| [`edges_example.py`](https://github.com/aion7891/Spect-RAG/blob/main/examples/edges_example.py) | the six edge signals | `ollama gemma3:4b` (adds LLM utility questions) |
| [`ingest_example.py`](https://github.com/aion7891/Spect-RAG/blob/main/examples/ingest_example.py) | the whole offline pipeline with `SpectRAGIngestion` | `ollama gemma3:4b` |
| [`seed_example.py`](https://github.com/aion7891/Spect-RAG/blob/main/examples/seed_example.py) | hybrid seeds: dense + BM25 + utility questions, fused with RRF | `ollama` |
| [`expand_example.py`](https://github.com/aion7891/Spect-RAG/blob/main/examples/expand_example.py) | graph expansion and proximity re-scoring | `ollama` |
| [`search_example.py`](https://github.com/aion7891/Spect-RAG/blob/main/examples/search_example.py) | the full pipeline with the two main classes | `ollama 400` (400-token budget) |
| [`answer_example.py`](https://github.com/aion7891/Spect-RAG/blob/main/examples/answer_example.py) | an answer with sources | needs Ollama: `ollama qwen3:4b` picks the model |
| [`evaluate_example.py`](https://github.com/aion7891/Spect-RAG/blob/main/examples/evaluate_example.py) | SpectRAG against dense and hybrid top-k on 12 questions | `ollama gemma3:4b` (adds answer scores) |

## Sample documents

* [`examples/docs/`](https://github.com/aion7891/Spect-RAG/tree/main/examples/docs): three
  short documents (Aurora Labs, Helios Energy, River Valley Farming), used by `quickstart.py`
  and most examples.
* [`examples/eval/`](https://github.com/aion7891/Spect-RAG/tree/main/examples/eval): 14 documents
  and 12 questions with their evidence passages, 9 of them multi-hop. The format of
  `questions.json` is the one [`tune()`](../guides/tuning.md) and the evaluation helpers read:

```json
[
  {"question": "In which city is the company that hired the co-founder of Aurora Labs as chief technology officer based?",
   "answers": ["Porto"], "type": "bridge",
   "evidence": ["She later joined Helios Energy as its chief technology officer",
                "Helios Energy is a solar power company based in Porto"]}
]
```

Each `evidence` string is copied exactly from a document. A chunk counts as gold evidence when
it contains the string.

## Compare against plain top-k

`spectrag.utils.evaluation.evaluate_retrieval()` scores SpectRAG, dense top-k and hybrid top-k
with the same token budget:

```python
from spectrag import SpectRAGIngestion, SpectRAGRetrieval
from spectrag.utils.evaluation import evaluate_retrieval, load_questions

SpectRAGIngestion("eval_index").ingest("examples/eval/docs")
search = SpectRAGRetrieval("eval_index")
questions = load_questions("examples/eval/questions.json")
report = evaluate_retrieval(search, questions, budget=512)
print(report["summary"])     # precision, recall, f1, mrr, complete for dense, hybrid, spectrag
```

`report["rows"]` holds the same metrics for each question. With the offline hash embedder and
the default settings, SpectRAG does not beat hybrid top-k on this small set (evidence recall
0.889 against 0.944 at 512 tokens). Real embeddings and [tuning](../guides/tuning.md) matter.

## Interactive examples

* [SpectRAG, step by step](../interactive/step-by-step.html): every stage on
  a 12-chunk corpus.
* [The spectral part, explained visually](../interactive/spectral.html).
* [Graph view of the evaluation set](../interactive/graph-view.html), with
  one question's retrieval path.
