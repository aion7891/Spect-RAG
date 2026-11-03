"""The fast (vectorised) code must give exactly the same results as the simple loop versions."""

import numpy as np
import pytest

from reference import ref_bm25, ref_expand_graph, ref_rrf_top, ref_select_context
from spectrag.config import make_settings
from spectrag.utils.bm25 import bm25_scores, build_bm25
from spectrag.utils.expansion import expand_graph
from spectrag.utils.graph import neighbors
from spectrag.utils.seeding import ranking, rrf_fuse, top_seeds
from spectrag.utils.selection import select_context


def random_edges(rng, n, m):
    edges = {}
    for _ in range(m):
        i, j = sorted(rng.choice(n, 2, replace=False).tolist())
        edges[(i, j)] = round(float(rng.choice([0.2, 0.4, 0.6, 0.8])), 4)  # ties on purpose
    return [[i, j, w] for (i, j), w in sorted(edges.items())]


@pytest.mark.parametrize("seed", range(20))
def test_expansion_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 80))
    edges = random_edges(rng, n, int(rng.integers(0, 4 * n)))
    seeds = rng.choice(n, int(rng.integers(1, 6)), replace=False).tolist()
    for hops, cap in [(1, 200), (2, 200), (2, 10), (3, 25)]:
        fast = expand_graph(seeds, neighbors(edges, n), hops, cap)[0]
        assert fast == ref_expand_graph(seeds, edges, hops, cap)


@pytest.mark.parametrize("seed", range(20))
def test_rrf_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 60))
    scores = [rng.choice([0.1, 0.5, 0.9], n) for _ in range(3)]          # many ties
    rankings = [ranking(scores[0]), ranking(scores[1], keep=scores[1] > 0.4), ranking(scores[2])]
    expected_top, expected = ref_rrf_top(rankings, 60, 7)
    fused = rrf_fuse(rankings, 60, size=n)
    assert top_seeds(fused, 7) == expected_top
    for chunk_id, value in expected.items():
        assert fused[chunk_id] == pytest.approx(value)


def test_bm25_matches_reference():
    rng = np.random.default_rng(0)
    words = ["solar", "farm", "grape", "sensor", "water", "panel", "aurora", "helios"]
    texts = [" ".join(rng.choice(words, int(rng.integers(1, 15)))) for _ in range(40)]
    for query in ["solar farm", "grape water sensor", "aurora", "unknown words"]:
        assert np.allclose(bm25_scores(build_bm25(texts), query), ref_bm25(texts, query))


@pytest.mark.parametrize("seed", range(30))
def test_selection_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    vectors = rng.normal(size=(n, 6))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = {"embeddings": vectors,
             "chunks": [{"tokens": int(rng.integers(5, 120)), "cluster": int(rng.integers(0, 4)),
                         "doc_id": "d%d" % rng.integers(0, 5)} for _ in range(n)]}
    scored = {"candidates": list(range(n)), "scores": rng.choice([0.2, 0.5, 0.7, 1.0], n),
              "proximity": rng.random(n)}
    for budget, core, floor, rho in [(300, 3, 0.4, 0.5), (150, 1, 0.0, 1.0), (1000, 0, 0.6, 0.0)]:
        settings = make_settings({"token_budget": budget, "core_size": core,
                                  "precision_floor": floor, "length_power": rho})
        fast = [(int(k), reason) for k, reason in select_context(scored, index, settings)]
        assert fast == ref_select_context(scored, index, settings)
