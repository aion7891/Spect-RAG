"""Algorithm 1, steps 1-3: exact dense/question scores, BM25, and RRF seeds."""

import math

import numpy as np

from .indexing import tokenize


def retrieval_settings(
    *, token_budget=12_000, seed_k=20, max_candidates=200, rrf_k=60,
    alpha=0.4, core_size=5, precision_floor=0.35, backfill=True,
) -> dict:
    """Validate caller settings; numerical defaults not fixed by the paper are explicit."""
    values = {
        "token_budget": token_budget, "seed_k": seed_k, "max_candidates": max_candidates,
        "rrf_k": rrf_k, "core_size": core_size,
    }
    for name, value in values.items():
        if type(value) is not int or value < (1 if name in {"seed_k", "max_candidates"} else 0):
            raise ValueError(f"{name} must be an integer within its permitted range.")
    if seed_k > max_candidates:
        raise ValueError("max_candidates must be at least seed_k so every seed can be preserved.")
    for name, value in (("alpha", alpha), ("precision_floor", precision_floor)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and lie in [0, 1].")
    if type(backfill) is not bool:
        raise TypeError("backfill must be a boolean.")
    return {
        **values, "alpha": float(alpha), "precision_floor": float(precision_floor),
        "backfill": backfill, "dense_weight": 0.5, "spectral_weight": 0.5,
        "ppr_alpha": 0.85, "coverage_tolerance": 1e-9,
    }


def normalize_positive(scores) -> np.ndarray:
    """Positive max scaling; an all-zero vector stays zero, including singleton cases."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite one-dimensional vector.")
    scores = np.maximum(scores, 0)
    maximum = float(scores.max()) if len(scores) else 0.0
    return scores / maximum if maximum > 0 else np.zeros_like(scores)


def fuse_rankings(channels: dict[str, np.ndarray], rrf_k: int) -> tuple[np.ndarray, dict]:
    """Only positive entries vote; absent/missing channels contribute nothing."""
    count = len(next(iter(channels.values())))
    fused = np.zeros(count)
    rankings = {}
    for name, values in channels.items():
        if len(values) != count:
            raise ValueError("Retrieval channels must share the same chunk order.")
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Retrieval channel scores must be finite.")
        rows = [int(row) for row in np.argsort(-values, kind="stable") if values[row] > 0]
        rankings[name] = rows
        for rank, row in enumerate(rows, start=1):
            fused[row] += 1.0 / (rrf_k + rank)
    return fused, rankings


def initial_retrieval(index: dict, query: str, query_vector: np.ndarray, *, seed_k=20, rrf_k=60) -> dict:
    """Retrieve all exact scores so graph-expanded chunks retain hybrid relevance."""
    count = len(index["chunks"])
    vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
    dense_raw = np.zeros(count)
    scores, rows = index["dense_index"].search(vector, count)
    dense_raw[rows[0]] = scores[0]
    bm25_raw = (
        np.asarray(index["bm25"].get_scores(tokenize(query)), dtype=np.float64)
        if index["bm25"] is not None else np.zeros(count)
    )
    utility_raw = np.zeros(count)
    question_count = index["question_index"].ntotal
    if question_count:
        scores, rows = index["question_index"].search(vector, question_count)
        np.maximum.at(
            utility_raw, index["question_owners"][rows[0]], np.clip(scores[0], 0, 1),
        )
    channels = {
        "dense": np.clip(dense_raw, 0, 1),
        "bm25": np.maximum(bm25_raw, 0),
        "utility": utility_raw,
    }
    fused, rankings = fuse_rankings(channels, rrf_k)
    seeds = [int(row) for row in np.argsort(-fused, kind="stable") if fused[row] > 0][:seed_k]
    dense, lexical = normalize_positive(channels["dense"]), normalize_positive(channels["bm25"])
    active = int(np.any(dense > 0)) + int(np.any(lexical > 0))
    # Equal dense/BM25 relevance weight; renormalize when one has no positive signal.
    relevance = (dense + lexical) / active if active else np.zeros(count)
    return {
        "seed_rows": seeds, "relevance": relevance, "rrf": fused, "rankings": rankings,
        "raw": {"dense": dense_raw, "bm25": bm25_raw, "utility": utility_raw},
    }
