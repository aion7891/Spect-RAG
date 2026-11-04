"""Retrieval methods and metrics for the MultiHopRAG benchmark.

Every method returns a ranked list of chunk ids that fits the SAME token budget. Baselines:
  dense      - standard RAG / NaiveRAG: top-k by dense cosine
  bm25       - top-k by BM25
  hybrid     - HybridRAG-style: dense + BM25 fused with RRF (k = 60)
  dense_mmr  - MMR (lambda 0.7) over the dense top-200
  hyde       - HyDE: dense search with the embedding of an LLM-written hypothetical answer
SpectraFuse variants are run through SpectRAGRetrieval.retrieve() with their settings.
"""

import time

import numpy as np

from spectrafuse.utils.bm25 import bm25_scores
from spectrafuse.utils.evaluation import fill_budget
from spectrafuse.utils.metrics import reciprocal_rank
from spectrafuse.utils.seeding import min_max, ranking, rrf_fuse
from spectrafuse.utils.selectors import select

KS = (2, 5, 10)
BASELINES = ("dense", "bm25", "hybrid", "dense_mmr")


def dense_order(search, vector):
    return np.argsort(-(search.index["embeddings"] @ vector), kind="stable")


def baseline_ids(search, name, question, vector, budget):
    """Ranked chunk ids of one baseline under the token budget."""
    chunks = search.index["chunks"]
    if name in ("dense", "hyde"):
        return fill_budget(dense_order(search, vector), chunks, budget)
    lexical = bm25_scores(search.bm25, question, search.settings["bm25_k1"], search.settings["bm25_b"])
    if name == "bm25":
        return fill_budget(ranking(lexical, keep=lexical > 0), chunks, budget)
    if name == "hybrid":
        fused = rrf_fuse([dense_order(search, vector).tolist(), ranking(lexical, keep=lexical > 0)],
                         60, size=len(chunks))
        return fill_budget(np.lexsort((np.arange(len(fused)), -fused)), chunks, budget)
    if name == "dense_mmr":
        pool = dense_order(search, vector)[:200].tolist()
        dense = search.index["embeddings"][pool] @ vector
        scored = {"candidates": pool, "scores": min_max(dense), "proximity": np.zeros(len(pool))}
        settings = dict(search.settings, selector="mmr", mmr_lambda=0.7, token_budget=budget)
        return [pool[k] for k, _ in select(scored, search.index, settings)]
    raise ValueError(name)


def spectrag_ids(search, question, vector, budget, overrides):
    _, scored, picks = search.retrieve(question, dict(overrides, token_budget=budget),
                                       query_vector=vector)
    return [scored["candidates"][k] for k, _ in picks]


def run(search, method, question, vector, budget):
    """(ranked ids, seconds). method = baseline name or ("spectrag", overrides)."""
    start = time.perf_counter()
    if isinstance(method, str):
        ids = baseline_ids(search, method, question, vector, budget)
    else:
        ids = spectrag_ids(search, question, vector, budget, method[1])
    return ids, time.perf_counter() - start


def metrics(ids, gold, chunks, ks=KS):
    """Paper Section 4.5 metrics for one question (gold = {fact: chunk ids})."""
    gold_ids = set().union(*gold.values())
    row = {"chunks": len(ids), "tokens": sum(chunks[i]["tokens"] for i in ids)}
    for k in ks:
        top = set(ids[:k])
        recall = np.mean([1.0 if ids_ & top else 0.0 for ids_ in gold.values()])
        precision = len(top & gold_ids) / max(1, min(k, len(ids)))
        row["recall@%d" % k] = float(recall)
        row["precision@%d" % k] = precision
        row["f1@%d" % k] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    found = [1.0 if ids_ & set(ids) else 0.0 for ids_ in gold.values()]
    row.update({"recall": float(np.mean(found)), "complete": float(min(found)),
                "mrr": reciprocal_rank(ids, gold_ids)})
    return row


def average(rows):
    """Mean of every numeric metric over a list of per-question rows."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))] if rows else []
    return {k: float(np.mean([row[k] for row in rows])) for k in keys}
