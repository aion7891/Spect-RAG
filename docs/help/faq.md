---
description: Common questions about using SpectraFuse.
---

# FAQ

## General

### When should I use SpectraFuse instead of plain vector search?

When questions need facts from more than one place: *"Which company hired the co-founder of
Aurora Labs, and where is it based?"* Plain top-k retrieval ranks chunks one by one, so the
second fact, which may share no words with the question, often never reaches the LLM. SpectraFuse
walks a graph of your chunks to reach it, and picks a varied set of chunks for a fixed token
budget. For simple look-up questions, hybrid dense + BM25 search is usually as good and simpler.

### How is it different from GraphRAG or LightRAG?

Those systems use an LLM to extract entities and relations from every document, then retrieve
over that knowledge graph. SpectraFuse links the *chunks* themselves, with signals computed without
an LLM (embeddings, document structure, rule-based entities, references), and returns the
original text. Building the index needs no LLM calls, unless you choose to generate utility
questions. In the benchmark, LightRAG needed about 225 seconds of local-LLM time per article to
index; the comparison was too small to say which retrieves better ([details](../results/benchmarks.md#smaller-experiments)).

### Is this the paper authors' code?

No. It is an implementation of the method in *SpectRAG: Weighted Chunk Graphs for
Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation* by Zafer
Sahinoglu, Abdulazeez Mohammed, Buse Yaren Kazangirler and Cahit Berkay Kazangirler
([arXiv:2610.36991](https://arxiv.org/abs/2610.36991)), written from the paper. Where the paper
leaves a detail open, the choice made here is recorded in
[DESIGN.md](https://github.com/aion7891/SpectraFuse/blob/main/docs/DESIGN.md).

### Do I need a GPU, an API key or an internet connection?

No. The core package needs only numpy and runs offline. A GPU or an API only matters for the
embedding model or LLM you choose to plug in.

## Setup

### Which embedder should I use?

Anything but the default for real use. The default `hash` embedder matches words, not meaning;
it exists so the package works offline and in tests. `embedder="ollama"` uses
`nomic-embed-text` (the paper's model) locally, or pass
[your own function](../guides/custom-models.md#embedders), such as a sentence-transformers or an
OpenAI model.

### Which settings should I start with?

On MultiHopRAG, `selector="protected_mmr"` alone took evidence recall from 0.803 (defaults) to
0.845. Start there, then [tune](../guides/tuning.md) on 30 or more of your own questions.

### How big should the token budget be?

As big as your LLM's context and your cost allow, minus room for the instructions and the answer.
The paper and the benchmark use 12,000 tokens. Smaller budgets make the selector's choices
matter more.

### What chunk size should I use?

The default is the paper's 1,200 tokens with 100 overlap. Smaller chunks (for example 256–512
tokens) let the budget hold more, smaller pieces, at the cost of more chunks to embed and link.
Chunk settings need a new index, so compare a few by running [`tune()` on each index](../guides/tuning.md).

## Using it

### Can I add documents to an existing index?

Not yet: ingestion builds the whole index, because the graph and its spectral coordinates depend
on every chunk. Re-run ingestion on the full document set; the old index is replaced only once
the new one is complete.

### How do I use it with LangChain, LlamaIndex or my own framework?

There is no built-in integration. Call `search()` inside your framework's custom retriever and
return `result["chunks"]` (each has `text`, `doc_id` and `score`), or pass `result["context"]`
straight into your prompt.

### Does it work with languages other than English?

Tokenisation, BM25 and the rule-based entity extractor work on any language written with spaces
between words, and capital letters help the entity rules. Testing so far has been in English. For
other languages, use a multilingual embedder and an `entity_fn` suited to the language.

### Why does SpectraFuse return fewer chunks than the budget would allow?

Chunks are only chosen from the candidates the graph walk collected (at most
`max_candidates`, 200), and a chunk is only added if it fits in the tokens left. When every
remaining candidate is too long, or there are no candidates left, selection stops below the
budget. `result["tokens"]` tells you how much of the budget was used.

### Why is a chunk I expected missing?

Look at the retrieval path: `spectrafuse graph --index my_index --question "..."` shows the seeds,
the chunks the graph walk reached, and those selected. If the chunk was never reached, raise
`seed_count` or `hops`, or check that it is linked to the seeds (shared entities, sections or
meaning). If it was reached but not selected, raise the budget or try another `selector`.

## Project

### Is this the official SpectRAG?

No. SpectraFuse is an independent implementation inspired by SpectRAG, written from the paper and
not affiliated with or endorsed by its authors (Zafer Sahinoglu, Abdulazeez Mohammed, Buse Yaren
Kazangirler and Cahit Berkay Kazangirler). SpectRAG's official implementation and releases are at
[SightAndCite/SpectRAG](https://github.com/SightAndCite/SpectRAG). SpectraFuse contains no code
from it. Results and choices described here (for example the defaults the paper leaves open,
listed in the design notes) are SpectraFuse's, not SpectRAG's.

### Why was the package renamed from `spectrag`?

So that it can't be mistaken for an official SpectRAG release. The authors maintain SpectRAG's
releases in their repository and keep the name for that project. Up to 0.1.4 this package was
published as `spectrag`; from 0.2.0 it is `spectrafuse` (`pip install spectrafuse`,
`import spectrafuse`, the `spectrafuse` command). Only the names changed: indexes built with
`spectrag` 0.1.x open with `spectrafuse`.

### How do I cite SpectRAG?

SpectRAG is the authors' method, so please cite their paper:

```bibtex
@article{sahinoglu2026spectrag,
  title   = {SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in
             Multi-Hop Retrieval-Augmented Generation},
  author  = {Sahinoglu, Zafer and Mohammed, Abdulazeez and Kazangirler, Buse Yaren and
             Kazangirler, Cahit Berkay},
  journal = {arXiv preprint arXiv:2610.36991},
  year    = {2026}
}
```

### Where do I report a bug or ask for a feature?

On [GitHub issues](https://github.com/aion7891/SpectraFuse/issues). Pull requests are welcome; the
test suite runs offline in under 10 seconds (`pytest -q`).

### What changed between versions?

See the [changelog](https://github.com/aion7891/SpectraFuse/blob/main/CHANGELOG.md).
