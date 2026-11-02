# MultiHopRAG dataset (copy used for the SpectRAG benchmarks)

- **Source:** Tang and Yang, *MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for
  Multi-Hop Queries*, arXiv:2401.15391 (2024). This is reference [17] in the SpectRAG paper.
- **Downloaded from:** https://huggingface.co/datasets/yixuantt/MultiHopRAG (files `corpus.json`
  and `MultiHopRAG.json`, downloaded 2026-11-03).
- **License:** Open Data Commons Attribution License (ODC-BY). The files are unchanged.
- **Checksums:** `SHA256SUMS` in this folder (`shasum -a 256 -c SHA256SUMS`).

| File | Content |
|---|---|
| `corpus.json` | 609 news articles: `title`, `author`, `source`, `published_at`, `category`, `url`, `body` (1.06 M words) |
| `MultiHopRAG.json` | 2,556 questions: `query`, `answer`, `question_type` (comparison 856, inference 816, temporal 583, null 301), `evidence_list` (0–4 facts, each with the article `title` and the exact `fact` text) |

All 6,084 evidence facts appear word-for-word in their articles, so chunk-level gold labels are
exact. If the files are missing, `python benchmarks/download_multihoprag.py` downloads them
again and checks the checksums.
