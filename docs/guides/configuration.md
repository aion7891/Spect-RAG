---
description: Every setting, its default, its allowed values, and when it takes effect.
---

# Configuration

Settings are a plain dict. You only pass the ones you want to change; every other setting keeps
its default from `spectrag.DEFAULTS`.

```python
from spectrag import SpectRAGIngestion, SpectRAGRetrieval

SpectRAGIngestion("my_index", {"embedder": "ollama", "chunk_tokens": 512}).ingest("docs/")

search = SpectRAGRetrieval("my_index", {"token_budget": 4000, "selector": "protected_mmr"})
result = search.search("my question", settings={"alpha": 0.5})      # this call only
```

```bash
spectrag ingest docs/ --index my_index --set embedder=ollama --set chunk_tokens=512
spectrag search "my question" --index my_index --set selector=protected_mmr --budget 4000
```

On the command line, `--set` values are read as JSON (`--set n_clusters=null`,
`--set 'edge_weights={"semantic": 0.5, "entity": 0.5}'`), and anything that is not valid JSON
is taken as text (`--set selector=mmr`).

## When a setting takes effect

| Kind | Settings | How to change them |
|---|---|---|
| **Ingestion** | chunking, embedding, `semantic_k`, the vector-index build settings | at ingestion; changing them needs a new index |
| **Graph** | `edge_weights`, `edge_threshold`, `graph_k`, `spectral_dims`, `n_clusters` | at ingestion, or later with `search.rebuild_graph({...})` or `SpectRAGRetrieval(dir, settings)`: the saved signals are fused again, nothing is re-embedded |
| **Retrieval** | seeds, expansion, re-scoring, selection | at any time: `SpectRAGRetrieval(dir, settings)`, or per call with `search(q, settings=...)` |

`SpectRAGRetrieval` starts from the settings the index was built with, then applies yours. A
graph setting passed to `search(settings=...)` raises an error that tells you to use
`rebuild_graph()` instead.

!!! info

    **Every value is checked.** A misspelled name raises `UnknownSettingError` with a suggestion
    (`Unknown setting: 'token_budjet' (did you mean 'token_budget'?)`); a wrong value raises
    `SettingsError` with the allowed range. See [Errors and logging](../reference/errors.md).

**Source** says whether a default comes from the paper or was chosen because the paper gives no
value. The paper's values are a starting point, not the best values for your data: see
[Hyperparameter tuning](tuning.md).

## Chunking

| Setting | Default | Allowed | Meaning | Source |
|---|---|---|---|---|
| `chunk_tokens` | 1200 | whole number ≥ 16 | largest chunk, in tokens | paper |
| `overlap_tokens` | 100 | whole number ≥ 0, less than `chunk_tokens` | tokens repeated from the end of the previous chunk | paper |
| `min_chunk_tokens` | 32 | whole number ≥ 0 | smaller chunks are merged into the previous one | paper |
| `semantic_percentile` | 90 | 0–100 | text without headings is split where the distance between neighbouring sentences is above this percentile | paper |
| `semantic_embedder` | `None` | `None`, `"hash"`, `"ollama"`, or a function | the embedder for those sentence distances; `None` uses `embedder`, `"hash"` is much faster | chosen |

## Embedding

| Setting | Default | Allowed | Meaning | Source |
|---|---|---|---|---|
| `embedder` | `"hash"` | `"hash"`, `"ollama"`, or a function `texts -> vectors` | the embedding model: offline hashing, `nomic-embed-text` through Ollama, or [your own](custom-models.md) | chosen |
| `embed_dims` | 768 | whole number ≥ 8 | size of the hash embedder's vectors | paper |
| `ollama_model` | `"nomic-embed-text"` | text | the Ollama embedding model | paper |
| `ollama_url` | `"http://localhost:11434"` | text | the Ollama server | chosen |

## Vector search

| Setting | Default | Allowed | Meaning | Source |
|---|---|---|---|---|
| `ann_backend` | `"auto"` | `"auto"`, `"exact"`, `"faiss"`, `"hnsw"` | nearest-neighbour search; `"auto"` is exact up to 20,000 chunks, then FAISS or hnswlib if installed | chosen |
| `hnsw_m` | 32 | whole number ≥ 2 | links per node in the HNSW graph | chosen |
| `hnsw_ef_construction` | 200 | whole number ≥ 1 | HNSW build effort | chosen |
| `hnsw_ef_search` | 128 | whole number ≥ 1 | HNSW search effort | chosen |
| `dense_top_n` | 1000 | whole number ≥ 1 | with an approximate backend, dense scores for the top *n* chunks only | chosen |

See [Large corpora](large-corpora.md) for when to switch.

## Graph

| Setting | Default | Allowed | Meaning | Source |
|---|---|---|---|---|
| `edge_weights` | semantic 0.300, section 0.250, entity 0.200, adjacency 0.084, utility 0.083, citation 0.083 | dict of signal → number ≥ 0, positive sum | α of Eq. 10 (the paper's configuration 1); signals with no edges share their weight with the others | paper |
| `semantic_k` | 10 | whole number ≥ 1 | nearest neighbours per chunk for semantic and utility edges (ingestion) | chosen |
| `edge_threshold` | 0.05 | number ≥ 0 | τ: fused edges lighter than this are dropped | chosen |
| `graph_k` | 10 | whole number ≥ 1 | each chunk keeps at most its `graph_k` strongest edges | chosen |
| `spectral_dims` | 32 | whole number ≥ 1 | spectral coordinates per chunk (m) | paper |
| `n_clusters` | `None` | `None` or whole number ≥ 1 | spectral clusters for ClusterGain; `None` means about √(n/2) | chosen |

The paper's second weight set is available too:

```python
from spectrag.config import EDGE_WEIGHTS_CONFIG_2   # semantic .15, section .20, entity .25,
                                                     # adjacency .10, utility .28, citation .02
search.rebuild_graph({"edge_weights": EDGE_WEIGHTS_CONFIG_2})
```

## Retrieval

| Setting | Default | Allowed | Meaning | Source |
|---|---|---|---|---|
| `seed_count` | 20 | whole number ≥ 1 | seeds from hybrid search | paper |
| `hops` | 2 | whole number ≥ 0 | graph expansion depth; 0 turns expansion off | paper |
| `max_candidates` | 200 | whole number ≥ 1 | the most chunks expansion may collect | paper |
| `token_budget` | 12000 | whole number ≥ 1 | tokens of context per question (B) | paper |
| `rrf_k` | 60 | whole number ≥ 1 | the RRF constant | chosen |
| `bm25_k1` | 1.5 | number ≥ 0 | BM25 term-frequency saturation | chosen |
| `bm25_b` | 0.75 | 0–1 | BM25 length normalisation | chosen |
| `bm25_weight` | 0.5 | 0–1 | relevance = (1 − w)·dense + w·BM25 | chosen |
| `alpha` | 0.3 | 0–1 | weight of graph proximity in the score; 0 switches it off | chosen |
| `proximity_mode` | `"hybrid"` | `"hybrid"`, `"ppr"`, `"spectral"`, `"heat"`, `"none"` | the graph signal: PageRank and spectral closeness (mean), one of them, the heat kernel (Eq. 17), or none | paper |
| `ppr_beta` | 0.15 | 0–1 | PageRank restart share β (Eq. 18) | chosen |
| `ppr_steps` | 30 | whole number ≥ 1 | PageRank iterations | chosen |
| `diffusion_t` | 1.0 | number ≥ 0 | heat-kernel time t, used by spectral closeness and `"heat"` | chosen |

## Selection

| Setting | Default | Allowed | Meaning | Source |
|---|---|---|---|---|
| `selector` | `"spectrag"` | see below | how chunks are chosen under the budget | paper |
| `core_size` | 3 | whole number ≥ 0 | best-scored chunks always kept (the protected core, or the anchors) | chosen |
| `precision_floor` | 0.4 | 0–1 | greedy candidates must score at least this × the best score | chosen |
| `select_weights` | a 1.0, b 0.3, c 0.2, d 0.1, lambda 0.5 | dict with keys `a b c d lambda`, numbers ≥ 0 | weights of Eq. 21: score, proximity, cluster gain, document gain, redundancy | chosen |
| `length_power` | 0.5 | 0–1 | ρ of Eq. 22: how strongly short chunks are preferred; 0 means not at all | chosen |
| `mmr_lambda` | 0.7 | 0–1 | for `mmr` and `protected_mmr`: relevance against novelty | chosen |

### Selectors

| `selector` | What it does |
|---|---|
| `spectrag` | the paper's selector: protected core, greedy utility per token (Eq. 21–22), then backfill |
| `mmr` | Maximal Marginal Relevance under the budget: relevance against similarity to chunks already chosen |
| `protected_mmr` | the protected core first, then MMR |
| `anchor_expand_ce` | `core_size` anchors, then the anchors' graph neighbours ranked by a cross-encoder (`rerank_fn`, see [Your own models](custom-models.md)) or by score, then backfill |
| `topk` | score order until the budget is full (an ablation) |

Every selector stops at the token budget and never picks a chunk twice.

!!! warning

    With the defaults (`selector = "spectrag"`, `length_power = 0.5`), SpectRAG prefers many very
    short chunks. On MultiHopRAG this lowered Recall@10 to 0.433; `length_power = 0` raised it to
    0.580, and `topk`, `mmr` and `protected_mmr` reached 0.845 evidence recall without tuning. See
    [Benchmarks](../results/benchmarks.md#what-the-defaults-get-wrong).

## Look at the current settings

```python
import spectrag

spectrag.DEFAULTS["token_budget"]          # 12000
settings = spectrag.make_settings({"alpha": 0.5})   # defaults + your changes, checked
search.settings                            # the settings a SpectRAGRetrieval object uses
```

```bash
spectrag info --index my_index             # what the index contains, and how it was built
```
