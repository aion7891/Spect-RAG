"""Hybrid seed retrieval (paper Table 2 stage 1, Algorithm 1 lines 1-3).

Three rankings - dense cosine, BM25, and utility-question matching - are fused with
Reciprocal Rank Fusion (RRF) into the seed set S0. A hybrid (dense + BM25) relevance score
is also returned for every chunk, because later stages need it.
"""

import numpy as np

from spectrag.utils.bm25 import bm25_scores


def dense_scores(embeddings, query_vector):
    """Cosine similarity of every chunk to the query (vectors are normalised) - exact search."""
    return embeddings @ query_vector


def utility_scores(question_vectors, owners, query_vector, chunk_count):
    """For each chunk, its best-matching utility question. Chunks without questions get 0."""
    scores = np.zeros(chunk_count)
    if len(owners) == 0:
        return scores
    best = {}
    for similarity, owner in zip(question_vectors @ query_vector, owners):
        best[owner] = max(best.get(owner, -1.0), float(similarity))
    for owner, value in best.items():
        scores[owner] = value
    return scores


def ranking(scores, keep=None):
    """Chunk ids from best to worst score. 'keep' (True/False per chunk) leaves some out.
    Equal scores keep the chunk order, so the result is the same every run."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    return [int(i) for i in order if keep is None or keep[i]]


def rrf_fuse(rankings, k=60):
    """Reciprocal Rank Fusion: a chunk at rank r in a list earns 1 / (k + r)."""
    fused = {}
    for ranked in rankings:
        for rank, chunk_id in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def top_seeds(fused, count):
    """The 'count' chunks with the highest fused score (ties: lower id first)."""
    return sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:count]


def min_max(scores):
    """Rescale scores to 0..1. If all scores are equal, they all become 0."""
    scores = np.asarray(scores, dtype=float)
    low, high = scores.min(), scores.max()
    return (scores - low) / (high - low) if high > low else np.zeros_like(scores)


def hybrid_relevance(dense, lexical, bm25_weight):
    """Relevance for every chunk: (1 - w) * dense + w * BM25, both rescaled to 0..1."""
    return (1 - bm25_weight) * min_max(dense) + bm25_weight * min_max(lexical)


def find_seeds(query, query_vector, index, bm25, settings):
    """Run stage 1. Returns the seed ids plus every score, so each step can be inspected."""
    chunk_count = len(index["chunks"])
    owners = index["question_owners"]
    dense = dense_scores(index["embeddings"], query_vector)
    lexical = bm25_scores(bm25, query, settings["bm25_k1"], settings["bm25_b"])
    utility = utility_scores(index["question_vectors"], owners, query_vector, chunk_count)
    rankings = [ranking(dense), ranking(lexical, keep=lexical > 0)]
    if owners:  # utility questions exist only when an LLM made them at ingestion
        has_questions = set(owners)
        rankings.append(ranking(utility, keep=[i in has_questions for i in range(chunk_count)]))
    fused = rrf_fuse(rankings, settings["rrf_k"])
    return {"seeds": top_seeds(fused, settings["seed_count"]),
            "relevance": hybrid_relevance(dense, lexical, settings["bm25_weight"]),
            "dense": dense, "bm25": lexical, "utility": utility, "rrf": fused,
            "rankings": rankings}
