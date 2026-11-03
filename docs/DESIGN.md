# Design notes: the paper, and every decision this implementation made

SpectRAG is described in Sahinoglu et al., *SpectRAG: Weighted Chunk Graphs for Token-Budgeted
Context Selection in Multi-Hop Retrieval-Augmented Generation*, arXiv:2610.36991v2
([PDF](2610.36991v2.pdf)). The paper leaves many details open; each place where this code had to
choose is listed below (D1, D2, …), with the reason. Measured results are in
[PROGRESS.md](PROGRESS.md) and [../benchmarks/results/multihoprag/REPORT.md](../benchmarks/results/multihoprag/REPORT.md).

## 1. Paper analysis

### 1.1 The core idea (Eq. 8–9)

The paper sets retrieval up as choosing a context set `C` that maximises

```
F(C, q) = Σ Rel(c, q)  −  λ Σ Red(ci, cj)  +  γ Cov(C, q)  +  η Conn(C; G)
          relevance       redundancy          coverage        graph connectivity
```

With `λ = γ = η = 0` this reduces to plain top-k. Picking the best set exactly is too slow, so
the paper approximates it with the pipeline below and a greedy selection rule (Eq. 21–22).

### 1.2 Offline pipeline (paper Table 1), run once per document collection

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

### 1.3 Online pipeline (paper Table 2 and Algorithm 1), run per question

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

### 1.4 Main reported results (for context only; we do not reproduce the benchmarks)

- On MultiHopRAG: highest overall recall (91.8) and Recall@10 (84.0). Strong hybrid retrieval is
  still better on early precision (R@2, R@5, MRR).
- Answer accuracy is highest of the five systems, by a small margin (EM 71.2 vs 70.6). Gains are
  largest on *inference* questions.
- The graph has 2.8–6.6× fewer nodes than LightRAG, and the index is about 6× smaller (46 MB vs 274 MB).
- In LLM-judge comparisons, **diversity** is the most consistent win.

### 1.5 Inconsistencies found in the paper, and how we resolve them

| Where | Inconsistency | Our choice |
|---|---|---|
| Eq. 2 vs Eq. 10 | Eq. 2 lists 5 signals, Eq. 10 lists 6 | Use the 6 signals of Eq. 10 |
| Eq. 15 vs Table 2 / Alg. 1 | Eq. 15 seeds by dense similarity only; Table 2 uses RRF of three rankings | Follow Table 2 / Algorithm 1 (hybrid RRF) |
| Section 3.2.3 vs Table 1 | "Smallest *non-trivial*" vs "32 smallest" eigenvectors | Skip the trivial first eigenvector (Eq. 13: `u2 … u_{m+1}`) |
| Eq. 17 vs Table 2 | Heat-kernel diffusion vs Personalized PageRank | Use PPR (Eq. 18) as the default |
| Eq. 9 vs Eq. 21 | `Cov` and `Conn` in the objective vs `ClusterGain`, `DocGain`, `Prox` in the greedy rule | Implement Eq. 21–22, the rule the paper actually uses |
| Section 6 vs Table 8 | The text says SpectRAG is best on null questions; the table shows it is not | Does not affect the implementation |

---


## 2. Assumptions and deviations from the paper

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
| D11 | "Spectral eigenspace closeness" (not defined) | Each coordinate `z_k` is scaled by `exp(−t·λ_k)` (the heat kernel of Eq. 17, `t = diffusion_t = 1`); then cosine between the candidate and the relevance-weighted centre of the seeds, mapped to [0, 1] | The scaling makes smooth community directions count most; without it, small corpora (where the 32 coordinates span the whole spectrum) give near-random closeness |
| D12 | ClusterGain needs spectral clusters (method not given) | Normalised spectral clustering (Ng–Jordan–Weiss, as in von Luxburg [5]): for k clusters, take the trivial eigenvector (`√degree`, rebuilt from the degrees) plus the first k − 1 non-trivial coordinates, i.e. the k smoothest eigenvectors; scale rows to length 1; k-means. Start centres chosen far apart, so no randomness | Using all 32 coordinates mixes in high-frequency eigenvectors and hides communities (seen on the sample docs). Leaving out the trivial vector makes chunks near `u2 = 0` fall to a random side |
| D13 | Isolated nodes (degree 0) make `D^-1/2` undefined | Use 0 for those entries (their row of `L_sym` is the identity row) | Keeps the Laplacian well defined |
| D14 | Alternative selectors (`mmr`, `protected_mmr`, `anchor_expand_ce`) | Not implemented; only the default selector | Not described in enough detail |
| D15 | Benchmarks (MultiHopRAG, UltraDomain), LLM judge | Not reproduced; we include Recall@k, MRR, EM, F1 helpers and a toy evaluation | Needs datasets and paid APIs |
| D16 | 100-token overlap "as a suffix of the previous chunk" | Overlap only between chunks of the **same** section or semantic segment; no overlap across a heading or topic break | Overlap across sections would put one section's text under another's section path |
| D17 | Semantic breaks use `nomic-embed-text` sentence vectors | `split_by_meaning()` accepts any `embed_fn`; without one it uses word-count cosine. Ties at the percentile count as breaks unless all distances are equal | Phase 3 comes before the embedder; word overlap often gives many distances of exactly 1.0 |
| D18 | Chunks < 32 tokens are concatenated with the previous chunk | Merged only if the result stays within `chunk_tokens`, so a small chunk can survive when chunks are tiny (e.g. 60) | The maximum chunk size is treated as the harder limit |
| D19 | Overlap always present | Dropped for the next chunk when overlap + the next unit would go over the limit | Keeps every chunk within `chunk_tokens` |
| D20 | Section headings | The heading line is kept as the first line of its section's first chunk. A heading with no body text of its own (e.g. `## 2. Products`) is carried forward and put in front of the next section's heading; headings at the very end are added to the last section. In Markdown files only `#` headings count, so numbered lists (`1. Install`) stay normal text | No line is ever lost, there are no empty chunks, and heading words stay searchable |
| D21 | Numbered headings | A line like `2.1 Datasets` counts as a heading only if it starts with a capital letter after the number, has ≤ 10 words, and does not end in `.`, `:`, `,` or `;` | Avoids treating numbered list items as headings |
| D22 | FAISS `IndexFlatIP` file | Vectors saved as `embeddings.npy` (float32, L2-normalised); search is `embeddings @ query` | Same exact scores, one fewer dependency |
| D23 | `chunks.pkl` (pickle) | `chunks.json` (readable), loaded with `np.load(allow_pickle=False)` for arrays | Readable files, and loading an index never runs pickled code |
| D24 | Prefixes for the hash embedder | Not added: prefix words would only add noise to word-hash vectors | Prefixes are an instruction for `nomic-embed-text` only |
| D25 | Entity weight "overlap in named entities" (no formula) | Jaccard: shared ÷ all entities of the two chunks; only for pairs sharing ≥ 1 entity | Standard, in [0, 1], and does not over-reward chunks with many entities |
| D26 | Hierarchy weight "shared headings or a common parent section" (no formula) | Same section = 1.0; siblings or parent–child = shared prefix ÷ longer path (e.g. 2.1 vs 2.2 = 2/3); more distant = no edge; never across documents | Graded by depth, and restricting to direct relatives keeps the graph sparse |
| D27 | Citation edges: "references, hyperlinks, footnotes" | Two kinds: `Section 2.1` / `Sec. 3` / `§4` → chunks of that section in the same document; file names (`notes.md`, Markdown links) → every chunk of that document; weight 1.0 | Footnote and bibliography links are too ambiguous without layout information |
| D28 | Utility edges: "similarity between generated questions" | Each chunk's question vectors are averaged, then linked to its `semantic_k` nearest neighbours by cosine. Per-question vectors are kept for query-time matching (Phase 7) | A full question × question matrix grows too fast; averaging keeps it `O(n²)` in chunks |
| D29 | Question vectors | Embedded as `kind="document"` (prefix `search_document:`) | At query time the user's question is the `search_query:` side |
| D30 | Utility weight when no LLM is used | A signal with **no edges** gets weight 0, and the other weights are rescaled to sum to 1 (`active_weights()`) | Otherwise 8.3% of the weight would be lost and every fused edge would be weaker against `τ` |
| D31 | Neo4j write with per-signal attributes | `signals.json` keeps every signal's own edge list; `edges.json` holds the fused graph | Same information, readable without a database |
| D32 | Eigen-solver | `numpy.linalg.eigh` on the dense `n × n` Laplacian | Exact and simple; fine for thousands of chunks |
| D33 | Small corpora | `spectral_dims` is capped at `n − 1` (the number of non-trivial eigenvectors) | A graph with `n` nodes has only `n − 1` of them |
| D34 | "Never cuts mid-sentence" | A single "sentence" longer than a whole chunk (common in PDF text without full stops) is cut between words, as a last resort, with exact token checks | Later phases fill a token budget, so every chunk must really be ≤ `chunk_tokens` |
| D35 | Sparse graph, `\|E\| = O(k_g n)` (Section 3.2.2), Table 4 shows 0.8–5.7 edges per node | After `τ`, each chunk keeps at most its `graph_k = 10` strongest edges (an edge survives if it is in the top 10 of either end). Adjacency edges are always kept, so each document stays one connected chain | Without this, long sections (all pairs weight 1.0) and common entities gave 144 edges per chunk on a 3,465-chunk test; with it, about 7 |
| D36 | Entity edges | An entity in more than 1% of the chunks (and more than 20 chunks) adds no entity edges (5% before the production audit); each chunk also keeps only its `graph_k` strongest entity and section edges | It does not tell chunks apart (like a stopword), and counting its pairs is `O(df²)`. Small corpora are never affected |
| D37 | Large graphs | Above 2,000 chunks, if `scipy` is installed, eigenvectors come from `scipy.sparse.linalg.eigsh` on `D^-1/2 W D^-1/2` (its largest eigenvalues are `1 −` the smallest of `L_sym`) with a fixed start vector | The paper suggests approximate eigensolvers for scale (Section 7); results match the dense solver (tested) |
| D38 | BM25 settings and tokenizer (not given) | Lowercase `\w+` words, no stemming and no stopword list; `k1 = 1.5`, `b = 0.75`; Lucene-style `idf = ln(1 + (N − df + 0.5) / (df + 0.5))` | Standard choices; this IDF is never negative, so common words cannot push scores down |
| D39 | RRF lists | Dense ranks **all** chunks; BM25 ranks only chunks that contain a query word; utility ranks only chunks that have questions, and is left out when there are none. RRF constant `k = 60`, ranks start at 1 | A chunk with zero keyword overlap was not "retrieved" by BM25, so it should not earn BM25 RRF credit |
| D40 | Utility matching "query to stored questions" | A chunk's utility score is the **best** cosine between the query and any of its questions | One matching question is enough to show the chunk answers the query |
| D41 | "The full relevance vector … is hybrid (dense & BM25)" (Table 2), no formula | `relevance = (1 − w)·minmax(dense) + w·minmax(BM25)` with `w = bm25_weight = 0.5`, over all chunks | Puts both on the same 0..1 scale before mixing; the weight can be changed |
| D42 | FAISS exact search for dense seeds and utility questions (Section 6.2) | `embeddings @ query` and `question_vectors @ query` in `numpy` | Exactly the same result as `IndexFlatIP` |
| D43 | Two-hop BFS "with a cap of 200 nodes" (order inside a hop not given) | Seeds first; then each hop's new chunks, strongest link to the previous hop first, until the cap is reached | When the cap cuts a hop, the most strongly connected chunks are kept |
| D44 | PPR "from the seeds over the real weighted subgraph" | Eq. 18 on the subgraph of the candidates only; the restart vector is the seeds' hybrid relevance, summing to 1 (equal shares if all are 0); 30 steps | Seeds that match the question better spread more relevance |
| D45 | Proximity is "the mean of spectral eigenspace closeness and Personalized PageRank" | PPR is min-max rescaled over the candidates (spectral closeness is already 0..1), then `proximity = (ppr + spectral) / 2` | The two must be on the same scale before averaging |
| D46 | Eq. 21 terms are named but not defined | `s_c` = the re-scored score; `Prox` = Phase 8 proximity; `ClusterGain = 1 / (1 + chunks already picked from c's cluster)`; `DocGain` = the same per document; `Red` = the largest cosine between c and any picked chunk | Each term is in [0, 1], rewards what is not yet covered, and is cheap to update after each pick |
| D47 | Protected core | The `core_size` best-scored candidates that fit, picked in score order (the floor does not apply to them) | "Strongly relevant evidence is not removed by later diversity or budget constraints" (Section 3.8) |
| D48 | Greedy tail | Repeatedly pick the candidate with the best `U / tokens^ρ` among candidates with `score ≥ precision_floor × best score` that fit the budget; a candidate is only picked if `U > 0` | With `U ≤ 0`, dividing by length would favour long redundant chunks |
| D49 | Backfill "remaining high-quality candidates" | All remaining candidates in score order, each one if it still fits | Fills the budget without any further rules |
| D50 | Budget accounting (Eq. 19) | Counts chunk tokens only; the `### Source:` headers and section labels added by `format_context()` are not counted (a few tokens per chunk) | Eq. 19 sums `ℓ(c)` over the chunks |
| D51 | "Format C grouped by source document" | Documents ordered by their best chunk score; chunks inside a document in reading order; the section label is shown only when the chunk does not start with its heading | Keeps the most relevant source first and each document readable |
| D52 | Generation with `gpt-4o-mini`, temperature 0 | `answer(question, llm_fn)` with any function; `ollama_llm()` uses temperature 0 and strips `<think>` blocks of reasoning models | Any provider works; no API key needed for the demos |
| D53 | Exact Match on short gold answers (MultiHopRAG) | Answer evaluation uses a **short-answer** prompt ("reply with the answer only"); `answer()` keeps the full grounded prompt | With the full prompt, LLMs write sentences and EM was 0.00 for every method |
| D54 | Benchmarks MultiHopRAG and UltraDomain, `gpt-4o` judge | A 14-document, 12-question toy set (9 multi-hop) with exact evidence snippets; the same budget for every method; scaled-down settings (`chunk_tokens = 128`, `seed_count = 5`, 400-token budget) for a corpus of 26–28 chunks | Runs offline in seconds; results show tendencies, not benchmark numbers |
| D55 | Retrieval recall (Eq. 23) over gold chunks | Recall = share of evidence **snippets** found (a snippet counts if any selected chunk contains it); "complete" = all snippets found; precision and MRR over chunks that contain a snippet | Chunk overlap can put one snippet into two chunks; counting snippets avoids penalising that |

---


## 3. Decisions added in the production upgrade

| # | Topic | This implementation | Why |
|---|---|---|---|
| D56 | `mmr` selector (named in Table 3, not defined) | Classic MMR (Carbonell and Goldstein [4]) over the re-scored candidates: repeatedly pick `argmax λ·score − (1 − λ)·max cosine to the picks`, `λ = mmr_lambda = 0.7`, until the budget is full | The standard definition |
| D57 | `protected_mmr` | The `core_size` best candidates first (as in the default selector), then MMR | "Protected core" + MMR, as the name says |
| D58 | `anchor_expand_ce` | Anchors = `core_size` best candidates; then the anchors' graph neighbours among the candidates, ranked by a cross-encoder (`rerank_fn(question, texts)`) if one is given, else by score; then backfill | "ce" read as cross-encoder; no model is bundled, so it is a plug-in |
| D59 | `heat` proximity (Eq. 17) | `exp(−t L) s` from the stored eigenpairs: `u1 (u1·s) + U e^(−tΛ) Uᵀ s` | Exact with the full eigenbasis (tested against `scipy.linalg.expm`) |
| D60 | Graph sparsity on large corpora | Entity and section signals keep each chunk's `graph_k` strongest edges before fusion; common-entity cap 1% | 2.79 M entity edges on 10k MultiHopRAG chunks became 70k, with the same fused graph size |
| D61 | Approximate nearest neighbours (Section 6.2) | `ann_backend = auto`: exact up to 20,000 chunks, then FAISS HNSW, then hnswlib. Query-time dense scores exact for the top `dense_top_n` only | Measured: ANN gives the same kNN graph 6.8× faster at 100k, but loses tail recall at query time |
| D62 | Semantic chunking cost | `semantic_embedder` may differ from `embedder` (e.g. `"hash"` for the sentence distances); default `None` = the same model, as in the paper | Embedding about 60,000 sentences through Ollama took over an hour on MultiHopRAG |
| D63 | Index files | Format 2: `.npy` / `.npz` for edges, signals and vectors, compact JSON for chunks; `format_version` in `info.json`; written atomically | 159 MB → 49 MB, and no half-written index after a crash |
| D64 | Defaults for 0.1.0 | Paper-based defaults are kept. On MultiHopRAG the validation-tuned settings were better (see PROGRESS.md, Phase 3); tuned defaults are planned for a later release | Tuned on one dataset only |
| D65 | Which embedder built an index (not in the paper) | `settings.json` saves your own embed function as `"custom:<name>"`. The index also keeps a fingerprint, `embedder_probe.npy`: the vector of one fixed sentence. Opening the index embeds that sentence again and raises `EmbedderError` if the size differs or the cosine is below 0.99 (assumed). Indexes without the file (made before 0.1.2) open with a log message | Two embedders of the same size (e.g. `hash` and `nomic-embed-text`, both 768-d) cannot be told apart by size, and mixing them returns nonsense with no error. 0.99 rather than exact equality, because Ollama on a GPU can return slightly different numbers each run. It costs one embedding call when an index is opened |
