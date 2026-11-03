import numpy as np

from spectrag.config import make_settings
from spectrag.utils.selection import select_context


def make_case(tokens, scores, clusters=None, docs=None, vectors=None, proximity=None):
    """A tiny fake index + scored candidates (candidate k is chunk k)."""
    n = len(tokens)
    clusters = clusters or [0] * n
    docs = docs or ["d%d" % i for i in range(n)]
    vectors = np.eye(n) if vectors is None else np.array(vectors, dtype=float)
    index = {"chunks": [{"tokens": t, "cluster": c, "doc_id": d}
                        for t, c, d in zip(tokens, clusters, docs)],
             "embeddings": vectors}
    scored = {"candidates": list(range(n)), "scores": np.array(scores, dtype=float),
              "proximity": np.array(proximity or [0.0] * n, dtype=float)}
    return index, scored


def settings(**overrides):
    return make_settings(dict({"token_budget": 1000, "core_size": 1, "length_power": 0.0},
                              **overrides))


def ids(picks):
    return [k for k, _ in picks]


def test_budget_is_never_exceeded():
    index, scored = make_case([400, 300, 300, 200, 100], [0.9, 0.8, 0.7, 0.6, 0.5])
    picks = select_context(scored, index, settings(token_budget=750))
    assert sum(index["chunks"][k]["tokens"] for k in ids(picks)) <= 750


def test_core_is_the_best_scores():
    index, scored = make_case([10] * 5, [0.2, 0.9, 0.5, 0.8, 0.1])
    picks = select_context(scored, index, settings(core_size=2))
    assert picks[:2] == [(1, "core"), (3, "core")]


def test_precision_floor_and_backfill():
    index, scored = make_case([10, 10, 10], [1.0, 0.9, 0.1])
    picks = select_context(scored, index, settings(precision_floor=0.5))
    assert picks == [(0, "core"), (1, "tail"), (2, "backfill")]   # 0.1 < 0.5 floor -> backfill


def test_redundancy_penalty_prefers_new_information():
    # chunk 1 is a near-copy of chunk 0; chunk 2 is different but scores a bit lower
    vectors = [[1, 0], [0.99, 0.14], [0, 1]]
    index, scored = make_case([10, 10, 10], [1.0, 0.8, 0.7], vectors=vectors)
    picks = select_context(scored, index, settings(token_budget=20))
    assert ids(picks) == [0, 2]


def test_cluster_gain_prefers_new_clusters():
    index, scored = make_case([10, 10, 10], [1.0, 0.75, 0.7], clusters=[0, 0, 1],
                              docs=["a", "a", "a"])
    picks = select_context(scored, index, settings(token_budget=20))
    assert ids(picks) == [0, 2]


def test_length_power_prefers_short_chunks():
    index, scored = make_case([10, 400, 40, 40], [1.0, 0.8, 0.7, 0.7])
    long_first = select_context(scored, index, settings(token_budget=410, length_power=0.0))
    short_first = select_context(scored, index, settings(token_budget=410, length_power=1.0))
    assert ids(long_first)[1] == 1                                 # rho = 0: utility only
    assert ids(short_first)[1:3] == [2, 3]                         # rho = 1: utility per token


def test_too_large_chunks_are_skipped_and_empty_input():
    index, scored = make_case([5000, 10], [1.0, 0.5])
    assert ids(select_context(scored, index, settings())) == [1]
    empty = {"candidates": [], "scores": np.array([]), "proximity": np.array([])}
    assert select_context(empty, index, settings()) == []
