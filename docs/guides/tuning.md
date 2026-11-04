---
description: >-
  Find better settings for your documents with grid or random search on your own validation
  questions.
---

# Hyperparameter tuning

The default settings follow the paper. They are a reasonable start, but they are not tuned for
your documents: on the MultiHopRAG benchmark they scored **below** plain hybrid retrieval
(evidence recall 0.803 against 0.832), and settings tuned on 300 validation questions scored
**above** it (0.846). Tuning is the most effective thing you can do for retrieval quality.

## What you need

1. **An index** of the documents your users ask about.
2. **Validation questions** with their evidence: 30 to 300 questions is a good range. Keep a
   separate set of test questions, never used for tuning, to check the result honestly.

Questions are a JSON list. Each `evidence` string must be copied exactly from a document; a
chunk counts as evidence when it contains the string (whitespace is normalised).

```json
[
  {"question": "Who leads the company that makes the panels used at the largest site of Helios Energy?",
   "evidence": ["The Évora Solar Park is the largest site operated by Helios Energy",
                "The park uses 180,000 Sunforge X7 panels"]}
]
```

`answers` and `type` fields are optional; `type` lets you break results down by question type.

## Run a search

```python
from spectrafuse import SpectRAGIngestion, SpectRAGRetrieval
from spectrafuse.utils.evaluation import load_questions
from spectrafuse.utils.tuning import tune

SpectRAGIngestion("eval_index", {"embedder": "ollama"}).ingest("examples/eval/docs")
search = SpectRAGRetrieval("eval_index")
questions = load_questions("examples/eval/questions.json")

grid = {
    "alpha": [0.1, 0.3, 0.5],
    "selector": ["spectrag", "protected_mmr"],
    "length_power": [0.0, 0.5],
    "edge_threshold": [0.05, 0.1],
}
results = tune(search, questions, grid, objective="recall", budget=4000, trials=20)

best = results[0]
print(best["settings"], best["recall"], best["complete"])
```

`tune()` tries combinations from the grid and returns one row per trial, best first. Each row
holds the `settings` of that trial and its average scores.

| Argument | Meaning |
|---|---|
| `search` | a `SpectRAGRetrieval` object; it is left as it was afterwards |
| `questions` | the validation questions |
| `grid` | `{setting: [values to try]}`; any retrieval or graph setting |
| `objective` | the metric to sort by: `"recall"` (share of evidence found, the default), `"complete"` (share of questions with *all* their evidence), `"precision"`, `"f1"` or `"mrr"` |
| `budget` | the token budget for every trial; use the budget you will use in production |
| `trials` | how many random combinations to try; `None` tries the full grid |
| `seed` | the random seed for choosing combinations (default 0, so runs repeat) |

!!! info

    **Graph settings are cheap to tune.** `edge_threshold`, `graph_k`, `edge_weights`,
    `spectral_dims` and `n_clusters` are applied by re-fusing the saved edge signals: nothing is
    embedded again. Trials are ordered so that the graph is rebuilt only when it changes. The query
    embeddings are also computed once, for all trials.

!!! warning

    Chunking and embedding settings (`chunk_tokens`, `embedder`, `semantic_k`, …) cannot be tuned
    with `tune()`, because they need a new index. To compare them, build one index per value and run
    `tune()` on each.

## Which settings to try

These are the settings the MultiHopRAG benchmark searched over:

| Setting | Try | Why |
|---|---|---|
| `selector` | `spectrag`, `protected_mmr`, `mmr` | the largest single-setting effect on MultiHopRAG |
| `length_power` | 0.0, 0.5 | 0.5 favours short chunks, which can crowd out long, useful ones |
| `precision_floor` | 0.2, 0.4, 0.6 | how weak a candidate may be and still enter the greedy tail |
| `alpha` | 0.0, 0.15, 0.3, 0.5 | how much graph proximity counts against relevance |
| `bm25_weight` | 0.3, 0.5, 0.7 | keyword matching against dense similarity in the relevance score |
| `seed_count` | 10, 20, 40 | more seeds give the graph walk more starting points |
| `proximity_mode` | `hybrid`, `ppr`, `heat` | the graph signal |
| `graph_k`, `edge_threshold` | 5 / 10 / 20, 0.05 / 0.1 | how dense the graph is |

The benchmark used exactly this grid with 30 random trials, on 300 validation questions. It took
39 seconds.

## What tuning found on MultiHopRAG

The best trial raised validation recall from 0.794 (defaults) to 0.841:

```python
{"alpha": 0.3, "precision_floor": 0.2, "seed_count": 20, "selector": "protected_mmr",
 "proximity_mode": "hybrid", "bm25_weight": 0.7, "length_power": 0.5,
 "edge_threshold": 0.05, "graph_k": 5}
```

On the 1,955 separate test questions these settings gave 0.846, against 0.832 for hybrid
retrieval (+0.014, 95% CI +0.007 to +0.021).

* The top three trials tied at 0.841 with different `alpha`, `seed_count` and
  `proximity_mode`. Four of the top five used `protected_mmr`. The selector mattered more than
  the exact proximity weights.
* The worst trial (heat proximity, `alpha` 0.7, 10 seeds) scored 0.536. Bad combinations exist, so
  tune rather than guess.

!!! info

    These settings were tuned on one dataset of news articles. Use them as a starting point for
    your own search, not as new defaults.

## Use the best settings

Retrieval settings and graph settings can both be passed when you open the index:

```python
search = SpectRAGRetrieval("my_index", best["settings"])
```

Graph settings that differ from the index rebuild the graph in memory each time the index is
opened. To make them permanent, ingest again with them:

```python
SpectRAGIngestion("my_index", {"embedder": "ollama", **best["settings"]}).ingest("docs/")
```

## Check the result

Always measure on questions you did not tune on:

```python
from spectrafuse.utils.evaluation import evaluate_retrieval

test = load_questions("test_questions.json")
report = evaluate_retrieval(SpectRAGRetrieval("my_index", best["settings"]), test, budget=4000)
print(report["summary"])    # SpectraFuse against dense and hybrid top-k, same budget
```

To score generated answers as well (exact match, token F1, contains),
`evaluate_answers(search, questions, budget, llm_fn)` does the same with any LLM function.
