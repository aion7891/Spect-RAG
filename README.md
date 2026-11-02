# SpectRAG — a small, readable Python implementation

SpectRAG is a retrieval method for RAG (Retrieval-Augmented Generation). It comes from the paper
**"SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop
Retrieval-Augmented Generation"** (Sahinoglu et al., arXiv:2610.36991v2), included at
[`docs/2610.36991v2.pdf`](docs/2610.36991v2.pdf).

This repository rebuilds the method as a **lightweight, beginner-friendly Python package**:
short files, small functions, `numpy` as the only required dependency, and tests for every phase.

> **Project status:** Phases 1–5 of 10 are done: paper analysis, architecture, package skeleton,
> settings, document loader, token counter, chunking, embeddings, index saving, and the six edge
> signals. Graph fusion, spectral coordinates and `SpectRAGIngestion` (Phase 6) are next. See
> [Section 11](#11-implementation-phases-and-progress) for progress and
> [Section 15](#15-what-is-implemented-so-far) for the code that exists today.

---

## Table of contents

1. [SpectRAG in plain words](#1-spectrag-in-plain-words)
2. [Paper analysis](#2-paper-analysis)
3. [Architecture](#3-architecture)
4. [Ingestion flow (offline)](#4-ingestion-flow-offline)
5. [Retrieval flow (online)](#5-retrieval-flow-online)
6. [Execution flow](#6-execution-flow)
7. [Package folder structure](#7-package-folder-structure)
8. [Planned scripts and functions](#8-planned-scripts-and-functions)
9. [Settings: paper values and our defaults](#9-settings-paper-values-and-our-defaults)
10. [Assumptions and deviations from the paper](#10-assumptions-and-deviations-from-the-paper)
11. [Implementation phases and progress](#11-implementation-phases-and-progress)
12. [Installation and usage (planned)](#12-installation-and-usage-planned)
13. [Testing strategy](#13-testing-strategy)
14. [Known issues](#14-known-issues)
15. [What is implemented so far](#15-what-is-implemented-so-far)

---

## 1. SpectRAG in plain words

A normal RAG system splits documents into **chunks**, turns every chunk into a vector, and returns
the *top-k* chunks closest to the question. The paper names three problems with that:

| Problem | What goes wrong |
|---|---|
| **Redundancy** | Near-duplicate chunks sit close together in vector space, so top-k returns several copies of the same fact. |
| **Missing bridge evidence** | Multi-hop questions need "connector" chunks that are not similar to the question itself, so top-k skips them. |
| **Structure blindness** | Chunks are treated as unrelated, even though they share sections, entities, citations, and document order. |

SpectRAG treats retrieval as **picking a good *set* of chunks**, not a list of individually good chunks:

1. **Build a graph** where every chunk is a node. Edges join chunks that are related in any of six
   ways: similar meaning, next to each other, same section, shared names, one cites the other, or
   they answer similar questions.
2. **Compute "spectral coordinates"** from the graph's Laplacian matrix. Each chunk gets a small
   vector (32 numbers) that says *where it sits in the graph's communities*, not what it says.
3. At question time, **find seed chunks** using dense vectors + BM25 keywords + question matching.
4. **Walk the graph** two hops out from the seeds to collect connector chunks.
5. **Spread relevance** across the graph (PageRank-style), so connectors close to strong seeds
   score higher.
6. **Fill a token budget** (about 12k tokens): keep the best chunks, then add chunks that cover new
   communities and new documents and repeat little, until the budget is full.

---

## 2. Paper analysis

### 2.1 The core idea (Eq. 8–9)

The paper sets retrieval up as choosing a context set `C` that maximises

```
F(C, q) = Σ Rel(c, q)  −  λ Σ Red(ci, cj)  +  γ Cov(C, q)  +  η Conn(C; G)
          relevance       redundancy          coverage        graph connectivity
```

With `λ = γ = η = 0` this reduces to plain top-k. Picking the best set exactly is too slow, so
the paper approximates it with the pipeline below and a greedy selection rule (Eq. 21–22).

### 2.2 Offline pipeline (paper Table 1), run once per document collection

| Step | Paper | What it produces |
|---|---|---|
| Chunking | Structural (headings) or semantic (sentence-embedding breakpoints at the 90th percentile). `cl100k_base` tokens: 1200 per chunk, 100 overlap. Never cuts mid-sentence; chunks under 32 tokens merge into the previous one; tables are kept whole as Markdown. | Chunks with text, position, section path, references |
| Embedding | `nomic-embed-text` (768-d), `search_document:` prefix, L2-normalised, FAISS `IndexFlatIP` (exact search) | One vector per chunk |
| Edge signals | Six extractors (Eq. 10); entities and utility questions come from `gpt-4o-mini` | Six edge lists `(i, j, weight)` |
| Graph build | `w_ij = Σ α_r · w_r,ij`; drop edges with `w_ij < τ` | Weighted graph `G = (V, E, W)` |
| Spectral decomposition | `L_sym = I − D^-1/2 W D^-1/2` (Eq. 11); the 32 smallest non-trivial eigenvectors (Eq. 12–13) | 32-d spectral coordinates `z_i` per chunk |
| Neo4j write | Stores the graph in Neo4j | Persistent graph |
| Persist | `chunks.pkl` + `faiss.index` | Files on disk |

**The six edge signals (Section 3.2.1), each scaled to [0, 1]:**

| Signal | Meaning | Paper weight (config 1) | Paper weight (config 2) |
|---|---|---|---|
| Semantic | `max(0, cos(e_i, e_j))`, only for each node's `k_g` nearest neighbours | 0.300 | 0.150 |
| Hierarchy (section) | Same heading or parent section | 0.250 | 0.200 |
| Entity | Overlap in names, terms, organisations, dates | 0.200 | 0.250 |
| Adjacency | Consecutive chunks in the same document | 0.084 | 0.100 |
| Utility question | Similarity of LLM-generated "questions this chunk answers" | 0.083 | 0.280 |
| Citation | References, hyperlinks, footnotes, "see Section X" | 0.083 | 0.020 |

### 2.3 Online pipeline (paper Table 2 and Algorithm 1), run per question

| Stage | Paper | Output |
|---|---|---|
| 1. Hybrid seed retrieval | Reciprocal Rank Fusion (RRF) of dense cosine, BM25, and query-to-utility-question matching. The relevance passed downstream is hybrid (dense + BM25). | Seed set `S0` (20 chunks) and a relevance score for every chunk |
| 2. Graph expansion | Two-hop breadth-first search (BFS) from `S0`, capped at 200 nodes | Candidate set `S1` and its subgraph |
| 3. Proximity re-scoring | `s2 = (1 − α)·relevance + α·proximity`; proximity = mean of *spectral closeness* and *Personalized PageRank* from the seeds | Re-scored candidates `S2` |
| 4. Token-budget selection | About 12k tokens: protected core, tail precision floor, document-coverage tie-break, backfill (Eq. 19–22) | Final context `C`, usually 10–15 chunks |
| 5. Generation | Context grouped by source document; `gpt-4o-mini`, temperature 0, grounded prompt | Answer and ranked sources |

**Diffusion (Section 3.7).** The exact form is `s' = exp(−t·L_sym)·s` (Eq. 17). The paper's
practical version is the repeated smoothing in Eq. 18:
`s(r+1) = β·s + (1 − β)·Ã·s(r)` with `Ã = D^-1/2 A D^-1/2`. This is Personalized PageRank.

**Selection score (Eq. 21–22).** Each greedy step picks the candidate that maximises

```
U(c | C) = a·s_c + b·Prox(c, S0) + c·ClusterGain(c, C) + d·DocGain(c, C) − λ·Red(c, C)
pick  argmax  U(c | C) / tokens(c)^ρ          (ρ in [0, 1] sets how much length matters)
```

### 2.4 Main reported results (for context only; we do not reproduce the benchmarks)

- On MultiHopRAG: highest overall recall (91.8) and Recall@10 (84.0). Strong hybrid retrieval is
  still better on early precision (R@2, R@5, MRR).
- Answer accuracy is highest of the five systems, by a small margin (EM 71.2 vs 70.6). Gains are
  largest on *inference* questions.
- The graph has 2.8–6.6× fewer nodes than LightRAG, and the index is about 6× smaller (46 MB vs 274 MB).
- In LLM-judge comparisons, **diversity** is the most consistent win.

### 2.5 Inconsistencies found in the paper, and how we resolve them

| Where | Inconsistency | Our choice |
|---|---|---|
| Eq. 2 vs Eq. 10 | Eq. 2 lists 5 signals, Eq. 10 lists 6 | Use the 6 signals of Eq. 10 |
| Eq. 15 vs Table 2 / Alg. 1 | Eq. 15 seeds by dense similarity only; Table 2 uses RRF of three rankings | Follow Table 2 / Algorithm 1 (hybrid RRF) |
| Section 3.2.3 vs Table 1 | "Smallest *non-trivial*" vs "32 smallest" eigenvectors | Skip the trivial first eigenvector (Eq. 13: `u2 … u_{m+1}`) |
| Eq. 17 vs Table 2 | Heat-kernel diffusion vs Personalized PageRank | Use PPR (Eq. 18) as the default |
| Eq. 9 vs Eq. 21 | `Cov` and `Conn` in the objective vs `ClusterGain`, `DocGain`, `Prox` in the greedy rule | Implement Eq. 21–22, the rule the paper actually uses |
| Section 6 vs Table 8 | The text says SpectRAG is best on null questions; the table shows it is not | Does not affect the implementation |

---

## 3. Architecture

Two main classes share one index folder on disk:

```mermaid
flowchart LR
    subgraph OFFLINE["SpectRAGIngestion (run once)"]
        S["Settings<br/>config.py"]:::done
        A["Documents<br/>.txt / .md / .pdf<br/>loader.py + tokens.py"]:::done --> B["Chunker<br/>chunker.py"]:::done
        B --> C["Embedder<br/>embedder.py"]:::done
        C --> D["Edge signals x6<br/>edges.py"]:::done
        D --> E["Weighted graph"]:::doing
        E --> F["Spectral coordinates"]:::todo
    end
    F --> G[("Index folder, store.py<br/>chunks.json, embeddings.npy: done<br/>edges.json, spectral.npy: pending")]:::doing
    subgraph ONLINE["SpectRAGRetrieval (per question)"]
        H["Question"] --> I["Hybrid seeds<br/>dense + BM25 + utility"]:::todo
        I --> J["2-hop graph expansion"]:::todo
        J --> K["Proximity re-scoring<br/>spectral + PageRank"]:::todo
        K --> L["Token-budget selection"]:::todo
        L --> M["Context + sources"]:::todo
        M -.optional.-> N["LLM answer"]:::todo
    end
    G --> I
    classDef done fill:#2e7d32,color:#fff,stroke:#1b5e20
    classDef doing fill:#f9a825,color:#000,stroke:#f57f17
    classDef todo fill:#9e9e9e,color:#fff,stroke:#616161
```

Colours: green = done, yellow = in progress, grey = pending.

**Design rules**

- **Pluggable models.** Embedding, entity extraction, utility questions, and answer generation are
  plain Python functions you can swap. Defaults need no internet and no API key, so every test
  runs offline.
- **No database server.** The paper uses Neo4j and FAISS. We keep the graph as a Python dict and
  search with `numpy` dot products (the same exact math as `IndexFlatIP`).
- **Files you can read.** The index is JSON plus `.npy` files, so you can open and inspect it.

---

## 4. Ingestion flow (offline)

```mermaid
flowchart TD
    A["load_docs(folder)"]:::done --> B{"has_structure()?"}:::done
    B -- yes --> C["split_by_sections()<br/>structural mode"]:::done
    B -- no --> D["split_by_meaning()<br/>semantic mode, 90th pct breaks"]:::done
    C --> U["split_units()<br/>sentences + whole tables"]:::done
    D --> U
    U --> E["pack_units()<br/>1200 tokens, 100 overlap"]:::done
    E --> F["merge_small_chunks()<br/>&lt; 32 tokens"]:::done
    F --> G["make_embedder()<br/>'search_document:' prefix"]:::done
    G -. "embed_fn for semantic breaks" .-> D
    F --> X1["add_entities()<br/>add_references()<br/>add_utility_questions()"]:::done
    G --> H1["semantic_edges()"]:::done
    G --> H6["utility_edges()"]:::done
    F --> H2["adjacency_edges()"]:::done
    F --> H3["section_edges()"]:::done
    X1 --> H4["entity_edges()"]:::done
    X1 --> H5["citation_edges()"]:::done
    X1 --> H6
    H1 & H2 & H3 & H4 & H5 & H6 --> I["fuse_edges()<br/>w = Σ α·w_r, drop w &lt; τ"]:::doing
    I --> J["normalized_laplacian()"]:::todo
    J --> K["spectral_coords()<br/>32 smallest non-trivial"]:::todo
    K --> L["cluster_coords()<br/>k-means for ClusterGain"]:::todo
    L --> M["save_index()"]:::todo
    classDef done fill:#2e7d32,color:#fff,stroke:#1b5e20
    classDef doing fill:#f9a825,color:#000,stroke:#f57f17
    classDef todo fill:#9e9e9e,color:#fff,stroke:#616161
```

Note: in semantic mode, `split_units()` runs before `split_by_meaning()`, so tables stay whole there too.

---

## 5. Retrieval flow (online)

```mermaid
flowchart TD
    Q["question"] --> E1["embed query<br/>'search_query:' prefix"]
    E1 --> D1["dense_scores()"]
    Q --> B1["bm25_scores()"]
    E1 --> U1["utility_scores()"]
    D1 & B1 & U1 --> R["rrf_fuse()<br/>top 20 = seeds S0"]
    D1 & B1 --> REL["hybrid_relevance()<br/>for every chunk"]
    R --> X["expand_graph()<br/>BFS 2 hops, max 200 = S1"]
    X --> P1["spectral_closeness()"]
    X --> P2["personalized_pagerank()"]
    P1 & P2 --> P["proximity = mean of both"]
    REL & P --> S2["rescore()<br/>(1-α)·rel + α·prox"]
    S2 --> C1["protected core"]
    C1 --> C2["greedy tail: Eq. 21-22<br/>precision floor"]
    C2 --> C3["backfill until budget full"]
    C3 --> OUT["format_context()<br/>grouped by document"]
```

---

## 6. Execution flow

What happens when a user runs the package end to end:

```mermaid
sequenceDiagram
    participant U as User
    participant I as SpectRAGIngestion
    participant D as Index folder
    participant R as SpectRAGRetrieval
    participant L as LLM (optional)
    U->>I: ingest("examples/docs")
    I->>I: load, chunk, embed, build graph, spectral
    I->>D: save_index()
    U->>R: SpectRAGRetrieval("my_index")
    R->>D: load_index()
    U->>R: search("question")
    R->>R: seeds, expand, rescore, select
    R-->>U: context, chunks, scores
    U->>R: answer("question", llm_fn)
    R->>L: grounded prompt
    L-->>U: answer + sources
```

---

## 7. Package folder structure

Planned layout (files are created phase by phase):

```
Spect-RAG/
├── README.md
├── pyproject.toml
├── docs/
│   └── 2610.36991v2.pdf            # the paper
├── examples/
│   ├── docs/                       # small sample documents
│   ├── ingest_example.py
│   └── search_example.py
├── src/
│   └── spectrag/
│       ├── __init__.py             # exports SpectRAGIngestion, SpectRAGRetrieval
│       ├── config.py               # default settings (paper values)
│       ├── ingestion.py            # class SpectRAGIngestion
│       ├── retrieval.py            # class SpectRAGRetrieval
│       └── utils/
│           ├── __init__.py
│           ├── loader.py           # read .txt/.md/.pdf files
│           ├── tokens.py           # count tokens, split sentences
│           ├── sections.py         # find headings, split by section
│           ├── units.py            # sentences + whole tables
│           ├── semantic_split.py   # semantic chunking
│           ├── chunker.py          # pack units into chunks
│           ├── embedder.py         # hashing embedder + Ollama nomic-embed-text
│           ├── entities.py         # simple entity extraction
│           ├── references.py       # "Section 2.1" and file-name references
│           ├── questions.py        # utility questions with any LLM (optional)
│           ├── edge_list.py        # add_edge / as_list helpers
│           ├── edges_structure.py  # adjacency, section, citation signals
│           ├── edges_content.py    # semantic, entity, utility signals
│           ├── edges.py            # build all six signals
│           ├── graph.py            # fuse weights, adjacency, neighbours
│           ├── spectral.py         # Laplacian, eigenvectors, clusters
│           ├── store.py            # save/load the index
│           ├── bm25.py             # BM25 keyword scoring
│           ├── seeding.py          # dense/BM25/utility scores + RRF
│           ├── expansion.py        # BFS graph expansion
│           ├── diffusion.py        # PageRank + spectral proximity + re-score
│           ├── selection.py        # token-budget context selection
│           └── prompt.py           # format context, optional answer
└── tests/
    ├── test_tokens.py
    ├── test_chunker.py
    └── ...                         # one test file per utils module
```

**Why `src/spectrag/utils/` rather than `src/utils/`:** a top-level package called `utils` would
clash with any other `utils` module once installed. Nesting it under `spectrag` keeps everything
inside `src/` and makes it importable as `spectrag.utils`.

---

## 8. Planned scripts and functions

Every function is meant to be about 5–20 lines and do one job.

| File | Main functions | Purpose |
|---|---|---|
| `config.py` | `DEFAULTS`, `make_settings(overrides)` | One dict of settings; users override only what they need |
| `utils/loader.py` | `load_docs(path)`, `read_file(path)` | Read a file or a folder into `{"doc_id", "text"}` dicts |
| `utils/tokens.py` | `count_tokens(text)`, `split_sentences(text)` | `tiktoken` when installed, otherwise a word-based estimate |
| `utils/sections.py` | `is_heading()`, `has_structure()`, `split_by_sections()` | Find headings and section paths |
| `utils/units.py` | `split_units()`, `split_table()` | Sentences and whole tables |
| `utils/semantic_split.py` | `split_by_meaning()`, `find_breakpoints()` | Break where sentence distance reaches the 90th percentile |
| `utils/chunker.py` | `pack_units()`, `take_overlap()`, `merge_small_chunks()`, `chunk_documents()` | Chunks with size limit and overlap |
| `utils/embedder.py` | `make_embedder()`, `hash_embed()`, `ollama_embed()`, `normalize()` | Offline default embedder; optional real `nomic-embed-text`; or your own function |
| `utils/entities.py` | `extract_entities(text)`, `add_entities(chunks)` | Capitalised phrases and their words, years (regex) |
| `utils/references.py` | `extract_references(text)`, `add_references(chunks)` | Section numbers and file names a chunk refers to |
| `utils/questions.py` | `make_question_fn(llm_fn)`, `add_utility_questions()`, `ollama_llm()` | Optional LLM-generated utility questions |
| `utils/edges_structure.py` | `adjacency_edges()`, `section_edges()`, `citation_edges()` | Structure signals |
| `utils/edges_content.py` | `semantic_edges()`, `entity_edges()`, `utility_edges()`, `knn_edges()` | Content signals |
| `utils/edges.py` | `build_edge_signals()` | Six lists of `[i, j, weight]` |
| `utils/graph.py` | `fuse_edges()`, `build_matrix()`, `neighbors()` | Weighted sum, threshold `τ`, adjacency matrix |
| `utils/spectral.py` | `normalized_laplacian()`, `spectral_coords()`, `cluster_coords()` | Eq. 11–13 plus k-means labels |
| `utils/store.py` | `save_index()`, `load_index()` | Any dict → `.npy` (arrays) and `.json` (everything else) |
| `utils/bm25.py` | `build_bm25()`, `bm25_scores()` | Classic BM25 in pure Python |
| `utils/seeding.py` | `dense_scores()`, `utility_scores()`, `rrf_fuse()`, `hybrid_relevance()` | Online stage 1 |
| `utils/expansion.py` | `expand_graph()` | Online stage 2 |
| `utils/diffusion.py` | `personalized_pagerank()`, `spectral_closeness()`, `rescore()` | Online stage 3 (Eq. 18) |
| `utils/selection.py` | `select_context()`, `pick_core()`, `utility_gain()`, `backfill()` | Online stage 4 (Eq. 19–22) |
| `utils/prompt.py` | `format_context()`, `build_prompt()` | Group by document; build the grounded prompt |
| `ingestion.py` | `SpectRAGIngestion.ingest(path)` | Runs the whole offline pipeline |
| `retrieval.py` | `SpectRAGRetrieval.search(q)`, `.answer(q, llm_fn)` | Runs the whole online pipeline |

---

## 9. Settings: paper values and our defaults

✅ = stated in the paper. ⚠️ = not stated in the paper, so we chose a value (all can be changed).

| Setting | Default | Source |
|---|---|---|
| `chunk_tokens` / `overlap_tokens` | 1200 / 100 (alt. 256 / 64) | ✅ |
| `min_chunk_tokens` | 32 | ✅ |
| `semantic_percentile` | 90 | ✅ |
| `embedder` | `"hash"` (offline). `"ollama"` = the paper's model, or pass your own function | D3 |
| `embed_dims` | 768 (size of the hash vectors; matches `nomic-embed-text`) | ✅ |
| `ollama_model` / `ollama_url` | `nomic-embed-text` / `http://localhost:11434` | ✅ / ⚠️ |
| `edge_weights` | config 1: sem .300, sec .250, ent .200, adj .084, utl .083, cit .083 | ✅ |
| `spectral_dims` (m) | 32 (capped at n − 2 for small corpora) | ✅ |
| `seed_count` | 20 | ✅ |
| `hops` / `max_candidates` | 2 / 200 | ✅ |
| `token_budget` | 12000 | ✅ |
| `semantic_k` (k_g) | 10 | ⚠️ |
| `edge_threshold` (τ) | 0.05 | ⚠️ |
| `rrf_k` | 60 (the usual RRF constant) | ⚠️ |
| `alpha` (re-score blend) | 0.3 | ⚠️ |
| `ppr_beta` / `ppr_steps` | 0.15 / 30 | ⚠️ |
| `core_size` (protected core) | 3 | ⚠️ |
| `precision_floor` | 0.4 × best score | ⚠️ |
| Eq. 21 weights `a, b, c, d, λ` | 1.0, 0.3, 0.2, 0.1, 0.5 | ⚠️ |
| `length_power` (ρ) | 0.5 | ⚠️ |
| `n_clusters` | `round(sqrt(n / 2))`, at least 2 | ⚠️ |

---

## 10. Assumptions and deviations from the paper

| # | Paper | This implementation | Why |
|---|---|---|---|
| D1 | Neo4j graph database | In-memory Python dict, saved as JSON | No server needed; BFS over a dict is easy to follow |
| D2 | FAISS `IndexFlatIP` | `numpy` dot product on normalised vectors | Same exact result with no extra dependency |
| D3 | `nomic-embed-text` | Default: offline hashing embedder (words + word pairs, md5-hashed into 768 slots with ±1 signs). Optional: real `nomic-embed-text` through Ollama (`urllib`, no new package), with the `search_document:` / `search_query:` prefixes. Or any function `texts -> vectors` | Tests run offline and fast; the real model stays available |
| D4 | `gpt-4o-mini` entity extraction | Default: regex extractor (capitalised phrases and their words, years; single words that start a sentence are skipped). Optional: pass your own `entity_fn` | No API key needed |
| D5 | `gpt-4o-mini` utility questions | Default: **off** (its weight is shared among the other signals). Optional: `make_question_fn(llm_fn)` with any LLM, e.g. `ollama_llm("gemma3:4b")` | Avoids any LLM requirement |
| D6 | Generation with `gpt-4o-mini` | `search()` returns the context; `answer()` takes any `llm_fn(prompt)` | Works with any LLM provider |
| D7 | `cl100k_base` tokenizer | `tiktoken` if installed, else about 0.75 words per token | `tiktoken` stays optional |
| D8 | PDF table extraction | Markdown tables are kept whole; PDF text read with optional `pypdf` | Reliable PDF table parsing is out of scope |
| D9 | Auto-calibrated edge weights | Fixed priors only (config 1 or 2) | The paper does not describe the calibration method |
| D10 | Heat-kernel diffusion `exp(−tL)` | PPR iteration (Eq. 18) by default | The paper names PPR in Table 2 and Algorithm 1 |
| D11 | "Spectral eigenspace closeness" (not defined) | Cosine between a candidate's `z` and the relevance-weighted centre of the seeds' `z`, mapped to [0, 1] | Simplest reading of the term |
| D12 | ClusterGain needs spectral clusters (method not given) | Small `numpy` k-means on spectral coordinates | Standard spectral clustering (von Luxburg [5]) |
| D13 | Isolated nodes (degree 0) make `D^-1/2` undefined | Use 0 for those entries | Keeps the Laplacian well defined |
| D14 | Alternative selectors (`mmr`, `protected_mmr`, `anchor_expand_ce`) | Not implemented; only the default selector | Not described in enough detail |
| D15 | Benchmarks (MultiHopRAG, UltraDomain), LLM judge | Not reproduced; we include Recall@k, MRR, EM, F1 helpers and a toy evaluation | Needs datasets and paid APIs |
| D16 | 100-token overlap "as a suffix of the previous chunk" | Overlap only between chunks of the **same** section or semantic segment; no overlap across a heading or topic break | Overlap across sections would put one section's text under another's section path |
| D17 | Semantic breaks use `nomic-embed-text` sentence vectors | `split_by_meaning()` accepts any `embed_fn`; without one it uses word-count cosine. Ties at the percentile count as breaks unless all distances are equal | Phase 3 comes before the embedder; word overlap often gives many distances of exactly 1.0 |
| D18 | Chunks < 32 tokens are concatenated with the previous chunk | Merged only if the result stays within `chunk_tokens`, so a small chunk can survive when chunks are tiny (e.g. 60) | The maximum chunk size is treated as the harder limit |
| D19 | Overlap always present | Dropped for the next chunk when overlap + the next unit would go over the limit | Keeps every chunk within `chunk_tokens` |
| D20 | Section headings | The heading line is kept as the first line of its section's first chunk; a heading with no body text (e.g. `## 2. Products`) gives no chunk but stays in its children's section path | Keeps heading words searchable without empty chunks |
| D21 | Numbered headings | A line like `2.1 Datasets` counts as a heading only if it starts with a capital letter after the number, has ≤ 10 words, and does not end in `.`, `:`, `,` or `;` | Avoids treating numbered list items as headings |
| D22 | FAISS `IndexFlatIP` file | Vectors saved as `embeddings.npy` (float32, L2-normalised); search is `embeddings @ query` | Same exact scores, one fewer dependency |
| D23 | `chunks.pkl` (pickle) | `chunks.json` (readable), loaded with `np.load(allow_pickle=False)` for arrays | Readable files, and loading an index never runs pickled code |
| D24 | Prefixes for the hash embedder | Not added: prefix words would only add noise to word-hash vectors | Prefixes are an instruction for `nomic-embed-text` only |
| D25 | Entity weight "overlap in named entities" (no formula) | Jaccard: shared ÷ all entities of the two chunks; only for pairs sharing ≥ 1 entity | Standard, in [0, 1], and does not over-reward chunks with many entities |
| D26 | Hierarchy weight "shared headings or a common parent section" (no formula) | Same section = 1.0; siblings or parent–child = shared prefix ÷ longer path (e.g. 2.1 vs 2.2 = 2/3); more distant = no edge; never across documents | Graded by depth, and restricting to direct relatives keeps the graph sparse |
| D27 | Citation edges: "references, hyperlinks, footnotes" | Two kinds: `Section 2.1` / `Sec. 3` / `§4` → chunks of that section in the same document; file names (`notes.md`, Markdown links) → every chunk of that document; weight 1.0 | Footnote and bibliography links are too ambiguous without layout information |
| D28 | Utility edges: "similarity between generated questions" | Each chunk's question vectors are averaged, then linked to its `semantic_k` nearest neighbours by cosine. Per-question vectors are kept for query-time matching (Phase 7) | A full question × question matrix grows too fast; averaging keeps it `O(n²)` in chunks |
| D29 | Question vectors | Embedded as `kind="document"` (prefix `search_document:`) | At query time the user's question is the `search_query:` side |

---

## 11. Implementation phases and progress

```mermaid
flowchart LR
    P1["1. Paper analysis<br/>& architecture"]:::done --> P2["2. Package skeleton,<br/>config, loader, tokens"]:::done
    P2 --> P3["3. Chunking"]:::done
    P3 --> P4["4. Embeddings<br/>& index store"]:::done
    P4 --> P5["5. Six edge signals"]:::done
    P5 --> P6["6. Graph + spectral<br/>= SpectRAGIngestion"]:::doing
    P6 --> P7["7. BM25 + hybrid<br/>seeding (RRF)"]:::todo
    P7 --> P8["8. Expansion +<br/>diffusion re-score"]:::todo
    P8 --> P9["9. Token-budget selection<br/>= SpectRAGRetrieval"]:::todo
    P9 --> P10["10. Answer, metrics,<br/>examples, polish"]:::todo
    classDef done fill:#2e7d32,color:#fff,stroke:#1b5e20
    classDef doing fill:#f9a825,color:#000,stroke:#f57f17
    classDef todo fill:#9e9e9e,color:#fff,stroke:#616161
```

Legend: 🟩 done · 🟨 in progress · ⬜ pending

| Phase | Scope | Testable result | Status |
|---|---|---|---|
| 1 | Paper analysis, architecture, plan, diagrams | This README | 🟩 Done |
| 2 | `pyproject.toml`, `config.py`, `loader.py`, `tokens.py`, sample docs | Load sample docs, count tokens, split sentences | 🟩 Done (18 tests) |
| 3 | `sections.py`, `units.py`, `semantic_split.py`, `chunker.py` | Chunks respect size, overlap, sentences, tables | 🟩 Done (44 tests in total) |
| 4 | `embedder.py`, `store.py` | Vectors are normalised; save/load gives the same data back | 🟩 Done (57 tests in total) |
| 5 | `entities.py`, `references.py`, `questions.py`, `edge_list.py`, `edges_structure.py`, `edges_content.py`, `edges.py` | Each signal finds the expected links on a toy corpus | 🟩 Done (81 tests + 1 opt-in) |
| 6 | `graph.py`, `spectral.py`, `ingestion.py` | `SpectRAGIngestion().ingest()` writes a full index | 🟨 Next |
| 7 | `bm25.py`, `seeding.py` | Seeds contain the obvious matching chunks | ⬜ Pending |
| 8 | `expansion.py`, `diffusion.py` | Bridge chunks gain score through the graph | ⬜ Pending |
| 9 | `selection.py`, `prompt.py`, `retrieval.py` | `search()` respects the budget and diversifies documents | ⬜ Pending |
| 10 | `answer()`, metrics, examples, final README | End-to-end demo plus SpectRAG vs top-k on a toy multi-hop set | ⬜ Pending |

---

## 12. Installation and usage

Installation works now, with either `pip` or `uv`:

```bash
git clone <your-repo-url> && cd Spect-RAG

# Option A: pip
python3 -m venv .venv && source .venv/bin/activate
pip install -e .              # installs numpy
pip install -e ".[dev]"       # adds pytest
pip install -e ".[extras]"    # optional: tiktoken (exact token counts), pypdf (PDF files)

# Option B: uv (installs everything, including the optional extras)
uv sync --all-extras
```

With `uv`, put `uv run --all-extras` in front of each command below (e.g.
`uv run --all-extras pytest -q`). A plain `uv sync` removes the extras, including `pytest`.

Runnable now (Phase 2 demo):

```bash
python examples/load_example.py
```

Expected output with `tiktoken` installed:

```text
Chunk size setting: 256 tokens
Token counter: tiktoken cl100k_base

aurora_labs.md
  tokens: 315  sentences: 24
  first sentence: # Aurora Labs
helios_energy.txt
  tokens: 162  sentences: 10
  first sentence: Helios Energy is a solar power company based in Porto.
river_valley_farming.md
  tokens: 143  sentences: 12
  first sentence: # River Valley Cooperative
```

Without `tiktoken`, the counter line says `word estimate`, and the token counts are 280, 164, and 145.

Runnable now (Phase 3 demo):

```bash
python examples/chunk_example.py           # paper default: 1200 tokens, 100 overlap
python examples/chunk_example.py 60 20     # tiny chunks, to see packing and overlap
```

Expected output with the default settings and `tiktoken` (shortened):

```text
chunk_tokens = 1200  overlap_tokens = 100
8 chunks

[0] aurora_labs.md #0  structural  109 tokens
    section: Aurora Labs > 1. History
    starts : ## 1. History
[1] aurora_labs.md #1  structural  47 tokens
    section: Aurora Labs > 2. Products > 2.1 Sensors
    starts : ### 2.1 Sensors
...
[4] helios_energy.txt #0  semantic  45 tokens
    section: (no headings)
    starts : Helios Energy is a solar power company based in Porto.
[5] helios_energy.txt #1  semantic  111 tokens
...
[7] river_valley_farming.md #1  structural  85 tokens
    section: River Valley Cooperative > Irrigation
    starts : ## Irrigation
```

What this shows:

- The Markdown files use **structural** mode and keep their section path. `# Aurora Labs` and
  `## 2. Products` have no body text, so they give no chunk of their own but appear in the path.
- `helios_energy.txt` has no headings, so it uses **semantic** mode and breaks where the topic
  changes.
- The 3-row product table stays inside chunk 2.
- `## Plans` (about 30 tokens) is under the 32-token minimum, so it was merged into the
  "Irrigation" chunk.

With `60 20` you get 13 chunks, all ≤ 60 tokens. Long sections are split across several chunks,
and each next chunk starts with the last sentence(s) of the previous one.

Runnable now (Phase 5 demo):

```bash
python examples/edges_example.py                    # offline: hash embedder, no utility questions
python examples/edges_example.py ollama gemma3:4b   # nomic-embed-text + LLM utility questions
```

Actual output (offline, shortened):

```text
[2] aurora_labs.md#2           refs=['2.1']
     entities: 2017, 2018, 2020, aurora, aurora s1, aurora s2, dashboard, fieldview
[3] aurora_labs.md#3           refs=['river_valley_farming.md']
...
Edges per signal: {'semantic': 16, 'adjacency': 5, 'section': 3, 'entity': 14, 'utility': 0, 'citation': 3}

section
  0.500  aurora_labs.md#0 -- aurora_labs.md#3
  0.667  aurora_labs.md#1 -- aurora_labs.md#2
entity
  0.280  aurora_labs.md#0 -- helios_energy.txt#0      <- bridge: both mention Maya Chen
citation
  1.000  aurora_labs.md#1 -- aurora_labs.md#2          <- "described in Section 2.1"
  1.000  aurora_labs.md#3 -- river_valley_farming.md#0 <- "See river_valley_farming.md"
```

With `ollama gemma3:4b`, every chunk also gets 3 questions, for example *"Who was hired as chief
technology officer in 2019?"* for the Helios chunk, and the utility signal adds 16 edges.

Planned API (runnable from Phases 6 and 9):

```python
from spectrag import SpectRAGIngestion, SpectRAGRetrieval

# Offline: run once per document collection
ingest = SpectRAGIngestion(index_dir="my_index")
ingest.ingest("examples/docs")

# Online: run per question
search = SpectRAGRetrieval(index_dir="my_index")
result = search.search("Which company did the founder of X later join?")
print(result["context"])          # text for your LLM, grouped by document
for chunk in result["chunks"]:    # selected chunks with scores
    print(chunk["doc_id"], chunk["score"])

# Optional: generate an answer with any LLM function
answer = search.answer("...", llm_fn=my_llm)
```

To use the real `nomic-embed-text` model (optional), install [Ollama](https://ollama.com), run
`ollama pull nomic-embed-text`, and use `make_settings({"embedder": "ollama"})`.

Runnable now (Phase 4 demo):

```bash
python examples/embed_example.py           # offline hash embedder
python examples/embed_example.py ollama    # real nomic-embed-text (needs Ollama running)
```

The demo chunks the sample docs, embeds them, saves `examples/demo_index/` (`chunks.json`,
`embeddings.npy`, `settings.json`; git ignores it), loads it back, and runs a plain dense search.
Actual output:

```text
--- hash ---
Saved 8 chunks, embeddings (8, 768) to demo_index

Question: Who is the chief technology officer of Helios Energy?
  1. score 0.255  [4] helios_energy.txt  Helios Energy is a solar power company based in Po
  2. score 0.198  [0] aurora_labs.md  ## 1. History
  3. score 0.134  [7] river_valley_farming.md  ## Irrigation
--- ollama ---
Saved 8 chunks, embeddings (8, 768) to demo_index

Question: Who is the chief technology officer of Helios Energy?
  1. score 0.781  [5] helios_energy.txt  In 2019 Helios Energy hired Maya Chen as chief tec
  2. score 0.746  [4] helios_energy.txt  Helios Energy is a solar power company based in Po
  3. score 0.672  [0] aurora_labs.md  ## 1. History
```

The hash embedder matches shared *words*, so it ranks Helios's intro chunk first. `nomic-embed-text`
matches *meaning*, so it finds the exact answer chunk. The real embedder also moves the semantic
break in `helios_energy.txt` (chunk 5 now starts at "In 2019 Helios Energy hired..."), because
chunking receives it as `embed_fn`.

---

## 13. Testing strategy

- **Unit tests** (`pytest`): one test file per `utils` module, using tiny hand-written texts with
  known answers.
- **Integration tests**: from Phase 6, ingest `examples/docs` into a temporary folder; from
  Phase 9, run `search()` on it.
- **Offline by default**: no test needs the internet, an API key, or Ollama. Ollama calls are
  tested with a fake server. One live test, `test_real_nomic_embed_text`, runs only when Ollama
  is reachable on `localhost:11434`; otherwise it is skipped.
- **Opt-in LLM test**: `SPECTRAG_LIVE_LLM=1 pytest tests/test_questions.py` calls a real Ollama
  LLM (default `gemma3:4b`; change it with `SPECTRAG_LLM_MODEL=qwen3:4b`).
- Run all tests: `pytest -q`. Run one file: `pytest tests/test_edges.py -q`. See skipped tests
  with `pytest -q -rs`.

---

## 14. Known issues

- `split_sentences()` is rule-based. It handles common abbreviations (`Dr.`, `e.g.`, `Fig.`) and
  decimals (`2.1`), but an unusual abbreviation followed by a capital letter can still cause a
  wrong split.
- Without an `embed_fn`, semantic chunking measures sentence distance by word overlap, so its
  topic breaks are rough. Pass `embed_fn=make_embedder(settings)` (done automatically from
  Phase 6) to use the chosen embedder.
- `nomic-embed-text` has a 2048-token context in Ollama. Chunks are at most 1200 tokens, so they
  fit, but very long single sentences may be cut off by the model.
- The Ollama embedder makes one HTTP request per 32 texts and has no retry. A large corpus on a
  slow machine can take a while.
- With very small `chunk_tokens` (e.g. 60), a chunk can stay under the 32-token minimum when
  merging it would break the maximum (see D18).
- A single sentence longer than `chunk_tokens` becomes its own over-size chunk, because chunks
  never cut a sentence.
- Only Markdown pipe tables (`| a | b |`) are recognised as tables. Tables from PDFs come out as
  plain text.
- The regex entity extractor misses lowercase technical terms (e.g. "soil moisture") and picks up
  table header words (`product`, `year`). Pass an LLM-based `entity_fn` for better entities.
- Semantic edge weights depend on the embedder: `nomic-embed-text` cosines are usually 0.6–0.9,
  hash cosines 0.1–0.4. The edge threshold `τ` (Phase 6) therefore removes different edges for
  each embedder.
- With few chunks, `semantic_k = 10` links almost every pair. The demo uses `semantic_k = 3`.
- `section_edges()` and `knn_edges()` compare all pairs (`O(n²)`), and `entity_edges()` links
  every pair that shares a common entity. Fine for thousands of chunks, slow for millions.
- Citation edges to a file link to every chunk of that file, which can add many edges for a
  large target document.
- With `tiktoken` installed, the first call to `count_tokens()` may download the `cl100k_base`
  file. If that fails (for example, offline), the counter silently falls back to the word estimate.
- PDF text comes from `pypdf`'s `extract_text()`, so layout such as columns and tables can come
  out jumbled.
- The hashing embedder captures word overlap, not meaning (see the Phase 4 demo). Use Ollama
  `nomic-embed-text` for real semantic quality. Hash collisions between different words are rare
  at 768 dimensions but possible.
- Dense eigen-decomposition (`numpy.linalg.eigh`) is `O(n³)`. It is fine up to a few thousand
  chunks; larger corpora would need `scipy.sparse.linalg.eigsh` (a possible later option).
- Values marked ⚠️ in Section 9 are our guesses, not tuned values from the paper.

---

## 15. What is implemented so far

### Phase 2: package skeleton, settings, loader, token counter

```mermaid
flowchart LR
    F["file or folder"] --> LF["list_files()<br/>.txt .md .markdown .pdf"]
    LF --> RF["read_file()"]
    RF -- ".pdf" --> RP["read_pdf()<br/>needs pypdf"]
    RF --> LD["load_docs()<br/>skip empty files"]
    RP --> LD
    LD --> DOCS["list of docs<br/>doc_id, path, text"]
    DOCS --> CT["count_tokens()<br/>tiktoken or estimate"]
    DOCS --> SS["split_sentences()"]
    SS --> MA["merge_abbreviations()"]
```

| File | Function | What it does |
|---|---|---|
| `pyproject.toml` | — | Package metadata. `numpy` is required; `[extras]` adds `tiktoken` and `pypdf`, `[dev]` adds `pytest`. |
| `src/spectrag/__init__.py` | — | Package entry point. Exports `DEFAULTS` and `make_settings` for now. |
| `src/spectrag/config.py` | `DEFAULTS` | Every setting in one dict, from Section 9. |
| | `EDGE_WEIGHTS_CONFIG_1` / `_2` | The paper's two edge-weight configurations (Tables 9 and 10). |
| | `make_settings(overrides)` | Returns a fresh copy of `DEFAULTS` with your changes. A misspelled key raises `KeyError`, so typos are caught early. |
| `src/spectrag/utils/loader.py` | `list_files(path)` | Finds supported files in a file or folder, including subfolders, in sorted order. |
| | `read_file(path)` | Reads a text file as UTF-8, or hands PDFs to `read_pdf`. |
| | `read_pdf(path)` | Extracts PDF text with `pypdf`, with a clear error if it is not installed. |
| | `make_doc_id(file, root)` | Builds a readable id such as `sub/b.md`, relative to the folder you loaded. |
| | `load_docs(path)` | Returns `[{"doc_id", "path", "text"}, ...]` and skips empty files. |
| `src/spectrag/utils/tokens.py` | `get_encoder()` | Loads `tiktoken`'s `cl100k_base` once (the paper's tokenizer), or returns `None`. |
| | `estimate_tokens(text)` | Fallback count: number of words divided by 0.75. |
| | `count_tokens(text)` | Exact count with `tiktoken` when possible, otherwise the estimate. |
| | `split_sentences(text)` | Splits on `.`, `!`, or `?` followed by a capital letter, digit, or quote. A blank line always ends a sentence. |
| | `merge_abbreviations(parts)` | Re-joins pieces wrongly split after `Dr.`, `e.g.`, `Fig.`, and similar. |
| `examples/docs/*` | — | Three small fictional documents: two Markdown files with headings, a table, and a cross-reference, and one plain-text file. They contain a multi-hop chain (Aurora Labs → Maya Chen → Helios Energy). |
| `examples/load_example.py` | `main()` | Phase 2 demo: loads the sample docs and prints token and sentence counts. |
| `tests/test_config.py` | 4 tests | Paper defaults, weights summing to 1, overrides not changing `DEFAULTS`, typo detection. |
| `tests/test_loader.py` | 7 tests | Folder and file loading, subfolders, unsupported or empty files, missing path, Unicode, PDF (skipped without `pypdf`). |
| `tests/test_tokens.py` | 7 tests | Estimate maths, fallback without `tiktoken`, sentence splitting, abbreviations, decimals, blank lines. |

Run the Phase 2 tests:

```bash
pytest -v tests/test_config.py tests/test_loader.py tests/test_tokens.py
```

### Phase 3: chunking

```mermaid
flowchart TD
    DOC["doc text"] --> HS{"has_structure()<br/>Markdown heading, or 2+ numbered headings"}
    HS -- yes --> SBS["split_by_sections()<br/>section path, heading, body"]
    SBS --> SU1["split_units(body)"]
    SU1 --> PK1["pack_units()<br/>per section"]
    HS -- no --> SU2["split_units(text)"]
    SU2 --> SBM["split_by_meaning()<br/>break at 90th pct distance"]
    SBM --> PK2["pack_units()<br/>per segment"]
    PK1 & PK2 --> MS["merge_small_chunks()<br/>&lt; 32 tokens joins previous if it fits"]
    MS --> OUT["chunk dicts<br/>id, doc_id, position, section, mode, text, tokens"]
```

Each chunk is a plain dict:

```python
{"id": 0, "doc_id": "aurora_labs.md", "position": 0, "mode": "structural",
 "section": ["Aurora Labs", "1. History"], "tokens": 109,
 "text": "## 1. History\nAurora Labs was founded in 2014 ...\n..."}
```

`id` is unique across all documents (it becomes the graph node number later). `position` is the
order inside one document (used for adjacency edges in Phase 5). `section` is the path used for
hierarchy edges.

| File | Function | What it does |
|---|---|---|
| `utils/sections.py` | `is_heading(line)` | Returns `(level, title)` for `## Title` (level = number of `#`) or `2.1 Title` (level = number of parts), otherwise `None`. |
| | `is_numbered_heading(line)` | Accepts `2.1 Datasets` but rejects list items such as `1. Buy milk.` |
| | `heading_lines(text)` | Lists every heading, ignoring lines inside ```` ``` ```` code blocks. |
| | `has_structure(text)` | Chooses structural mode: any Markdown heading, or at least two numbered headings. |
| | `split_by_sections(text)` | Walks the lines with a stack of headings and returns `{"section", "heading", "body"}` for every section that has text. |
| | `add_section(...)` | Saves one finished section and skips heading-only sections. |
| `utils/units.py` | `split_units(text, max_tokens)` | Turns text into a list of whole sentences and whole tables, in order. |
| | `is_table_line(line)` / `is_separator_row(line)` | Recognise Markdown table rows and the `|---|` row. |
| | `add_table(...)` | Adds one table as a unit, or splits it if it is too big. |
| | `split_table(table, max_tokens)` | Splits a big table between rows and repeats the header in each part, so no row is cut. |
| `utils/semantic_split.py` | `word_cosine(a, b)` | Similarity from word counts. The default when no embedder is given. |
| | `sentence_distances(sentences, embed_fn)` | `1 − cosine` between neighbouring sentences, from `embed_fn` vectors or word counts. |
| | `find_breakpoints(distances, percentile)` | Indexes where a new segment starts: distance ≥ the 90th percentile (and not all distances equal). |
| | `split_by_meaning(sentences, ...)` | Cuts the sentence list at those breakpoints. |
| `utils/chunker.py` | `pack_units(units, max, overlap)` | Fills a chunk until the next unit would not fit, then starts a new one with an overlap from the previous one. |
| | `take_overlap(units, overlap)` | Last sentences of a chunk that fit in `overlap_tokens`. Stops at a table and never takes the first unit. |
| | `structural_pieces()` / `semantic_pieces()` | Run the two modes and return `(section_path, chunk_text)` pairs. |
| | `merge_small_chunks(chunks, min, max)` | Joins chunks under 32 tokens onto the previous chunk when the result still fits. |
| | `chunk_document(doc, settings, embed_fn)` | One document → chunk dicts with `position`. |
| | `chunk_documents(docs, settings, embed_fn)` | All documents → chunk dicts with a global `id`. |
| `examples/chunk_example.py` | `main()` | Prints every chunk. Optional arguments: `chunk_tokens overlap_tokens`. |
| `tests/test_sections.py` | 6 tests | Markdown and numbered headings, structure detection, section paths, text before the first heading, code blocks. |
| `tests/test_units.py` | 4 tests | Sentence/table order, small tables whole, big tables split by row with the header repeated, size limit. |
| `tests/test_semantic_split.py` | 6 tests | Word cosine, breakpoint placement, no breaks for equal distances, topic change, custom `embed_fn`, short inputs. |
| `tests/test_chunker.py` | 11 tests | Size limit, whole sentences, overlap is a suffix of the previous chunk, zero overlap, overlap edge cases, merging and its limit, sample-doc modes, paths, positions, table kept in one chunk, 32-token minimum. |

Run the Phase 3 tests:

```bash
pytest -v tests/test_sections.py tests/test_units.py tests/test_semantic_split.py tests/test_chunker.py
```

### Phase 4: embeddings and index store

```mermaid
flowchart LR
    S["settings['embedder']"] --> ME{"make_embedder()"}
    ME -- '"hash"' --> HE["hash_embed()<br/>words + pairs, md5 to 768 slots"]
    ME -- '"ollama"' --> OE["ollama_embed()<br/>adds search_document: / search_query:"]
    OE --> OR["ollama_request()<br/>POST /api/embed, 32 per batch"]
    ME -- "your function" --> CF["custom texts to vectors"]
    HE & OR & CF --> N["normalize()<br/>length 1, dot = cosine"]
    N --> SI["save_index()<br/>arrays .npy, rest .json"]
    SI --> F[("index folder")]
    F --> LI["load_index()"]
```

`make_embedder(settings)` returns one function, `embed(texts, kind="document")`. Every embedder
returns a float32 `numpy` array with one length-1 row per text. Later phases call `embed(...)`
without caring which model is behind it. Use `kind="query"` for questions, so `nomic-embed-text`
gets the `search_query:` prefix.

| File | Function | What it does |
|---|---|---|
| `config.py` | (new keys) | `ollama_model`, `ollama_url`. `embedder` can now also be a function. |
| `utils/embedder.py` | `normalize(vectors)` | Divides each row by its length; all-zero rows stay zero. |
| | `text_features(text)` | Lowercase words plus neighbouring word pairs (`"solar farm"`). |
| | `hash_feature(feature, dims)` | md5 → a fixed slot and a ±1 sign. Python's `hash()` changes on every run, so md5 is used to get the same vectors every time. |
| | `hash_embed(texts, dims)` | Adds ±1 into the slot of every feature, then normalises. |
| | `ollama_request(texts, model, url)` | One `POST /api/embed` call with `urllib`. Raises a clear `ConnectionError` if Ollama is not reachable. |
| | `ollama_embed(texts, kind, ...)` | Adds the task prefix, sends batches of 32, normalises. |
| | `make_embedder(settings)` | Picks hash, Ollama, or your own function. |
| `utils/store.py` | `save_json(path, data)` / `load_json(path)` | Readable UTF-8 JSON. Values JSON cannot store (e.g. a function) are saved as text. |
| | `save_index(folder, index)` | Writes each dict key as `<key>.npy` (arrays) or `<key>.json` (anything else). |
| | `load_index(folder)` | Reads every `.npy` and `.json` back into a dict. |
| `examples/embed_example.py` | `main()` | Chunk → embed → save → load → dense search. Argument: `hash` or `ollama`. |
| `tests/test_embedder.py` | 10 tests | Normalising, shape/dtype, deterministic hashing, similar texts score higher, custom function, unknown name, Ollama prefixes + batching (fake server), friendly connection error, and the live `nomic-embed-text` check (768-d; similar sentences closer). |
| `tests/test_store.py` | 3 tests | Exact round trip of chunks (with Unicode) and float32 vectors, settings holding a function, missing folder. |

Run the Phase 4 tests:

```bash
pytest -v -rs tests/test_embedder.py tests/test_store.py
```

### Phase 5: the six edge signals

```mermaid
flowchart TD
    CH["chunks + embeddings"] --> AE["add_entities()<br/>chunk.entities"]
    CH --> AR["add_references()<br/>chunk.references"]
    CH --> AQ["add_utility_questions(question_fn)<br/>chunk.utility_questions"]
    CH --> SEM["semantic_edges()<br/>kNN cosine, k = semantic_k"]
    CH --> ADJ["adjacency_edges()<br/>position n and n+1"]
    CH --> SEC["section_edges()<br/>same / sibling / parent"]
    AE --> ENT["entity_edges()<br/>Jaccard"]
    AR --> CIT["citation_edges()<br/>Section X, file names"]
    AQ --> EQ["embed_questions()"] --> UTL["utility_edges()<br/>mean question vector, kNN"]
    SEM & ADJ & SEC & ENT & CIT & UTL --> OUT["build_edge_signals()<br/>{signal: [[i, j, w], ...]}"]
```

Every signal is a sorted list of `[i, j, weight]` with `i < j` and `0 < weight ≤ 1`, where `i`
and `j` are chunk ids. Each pair appears at most once per signal. Phase 6 combines the six lists
with the weights `α` from `settings["edge_weights"]` (paper Eq. 10).

| Signal | Rule in this code | Example from the sample docs |
|---|---|---|
| semantic | `max(0, cos)` to each chunk's `semantic_k` nearest neighbours | Aurora history ↔ Helios intro (0.353) |
| adjacency | 1.0 between positions `n` and `n+1` of one document | `aurora_labs.md#0` ↔ `#1` |
| section | 1.0 same section; siblings or parent–child = shared prefix ÷ longer path | 2.1 Sensors ↔ 2.2 Software (0.667) |
| entity | Jaccard of entity sets | Aurora history ↔ Helios (0.280, "maya chen", "helios energy") |
| utility | cosine of averaged question vectors, kNN | Only when a `question_fn` is given |
| citation | 1.0 to the referenced section or document | FieldView chunk → Section 2.1 chunk |

| File | Function | What it does |
|---|---|---|
| `utils/entities.py` | `extract_entities(text)` | Sorted, lowercased capitalised phrases (`maya chen`), their words (`maya`), and years. Works line by line, so headings never merge into sentences. |
| | `sentence_phrases(sentence)` | Phrases in one sentence. Skips a lone capitalised word at the start (`Farmers check ...`) and heading numbers (`## 1. History`). |
| | `clean_phrase(phrase)` | Trims stopwords: `The Aurora S1` → `aurora s1`. |
| | `add_entities(chunks, entity_fn)` | Stores `chunk["entities"]`; `entity_fn` can replace the regex extractor. |
| `utils/references.py` | `extract_references(text)` | Finds `Section 2.1`, `Sec. 3`, `§4.2` and file names/links such as `report.pdf`. |
| | `section_numbers(path)` | `["Doc", "2. Products", "2.1 Sensors"]` → `{"2", "2.1"}`. |
| | `add_references(chunks)` | Stores `chunk["references"]` (the field from paper Table 1). |
| `utils/questions.py` | `make_question_fn(llm_fn, count=3)` | Turns any `llm_fn(prompt) -> text` into a question generator. |
| | `parse_questions(reply)` | Keeps lines ending in `?` and removes numbering and bullets. |
| | `add_utility_questions(chunks, question_fn)` | Stores `chunk["utility_questions"]` (empty list when `question_fn` is `None`). |
| | `ollama_llm(model)` | Optional `llm_fn` that calls a local Ollama model at temperature 0. |
| `utils/edge_list.py` | `add_edge(edges, i, j, w)` | Stores the pair once as `(min, max)`, keeps the larger weight, ignores self-loops and zero weights, caps at 1.0. |
| | `as_list(edges)` | Dict → sorted `[[i, j, w], ...]` (JSON-friendly). |
| `utils/edges_structure.py` | `adjacency_edges(chunks)` | Neighbouring chunks of one document. |
| | `shared_prefix()`, `section_weight()` | The section rule from the table above. |
| | `section_edges(chunks)` | Applies `section_weight` to every pair in the same document. |
| | `reference_targets()`, `citation_edges(chunks)` | Resolves each reference to chunks and links them. |
| `utils/edges_content.py` | `knn_edges(vectors, k)` | Shared kNN routine (paper Eq. 3), used by semantic and utility edges. |
| | `semantic_edges(embeddings, k)` | kNN over chunk embeddings. |
| | `jaccard()`, `entity_edges(chunks)` | Inverted index entity → chunks, then Jaccard for pairs that share an entity. |
| | `embed_questions(chunks, embed)` | All question vectors plus `owners` (the chunk each question belongs to), reused at query time in Phase 7. |
| | `utility_edges(vectors, owners, n, k)` | Averages each chunk's question vectors, then kNN. |
| `utils/edges.py` | `build_edge_signals(chunks, embeddings, embed, settings)` | Runs all six and returns `(signals, question_vectors, owners)`. |
| | `count_edges(signals)` | Edge count per signal. |
| `examples/edges_example.py` | `main()` | Prints entities, references, questions and every edge. Arguments: `[embedder] [llm-model]`. |
| `tests/test_entities.py` | 6 tests | Names, products and years; sentence-start rule; two-word names at the start; stopwords; headings kept separate; custom `entity_fn`. |
| `tests/test_references.py` | 5 tests | Section forms, files and links, duplicates, `section_numbers`, `add_references`. |
| `tests/test_questions.py` | 4 tests + 1 opt-in | Parsing, prompt and limit, off by default, Ollama request format (fake server); live LLM with `SPECTRAG_LIVE_LLM=1`. |
| `tests/test_edges.py` | 9 tests | `add_edge` rules, adjacency, every section-weight case, no cross-document section edges, citation by section and by file, kNN limits and negatives, Jaccard, utility averaging, and the full pipeline on sample docs (all 6 signals present, valid weights, the Maya Chen bridge, the Section 2.1 citation). |

Run the Phase 5 tests:

```bash
pytest -v -rs tests/test_entities.py tests/test_references.py tests/test_questions.py tests/test_edges.py
```
