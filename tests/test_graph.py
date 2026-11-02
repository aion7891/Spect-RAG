import numpy as np
import pytest

from spectrag.utils.graph import (active_weights, build_matrix, degrees, fuse_edges,
                                  keep_strongest, neighbors)

WEIGHTS = {"semantic": 0.3, "section": 0.25, "entity": 0.2, "adjacency": 0.084,
           "utility": 0.083, "citation": 0.083}


def test_active_weights_share_empty_signal_weight():
    signals = {"semantic": [[0, 1, 1.0]], "section": [[0, 1, 1.0]], "entity": [[0, 1, 1.0]],
               "adjacency": [[0, 1, 1.0]], "utility": [], "citation": [[0, 1, 1.0]]}
    weights = active_weights(WEIGHTS, signals)
    assert weights["utility"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["semantic"] == pytest.approx(0.3 / (1 - 0.083))


def test_active_weights_all_empty():
    assert set(active_weights(WEIGHTS, {}).values()) == {0.0}


def test_fuse_edges_weighted_sum_and_threshold():
    signals = {"semantic": [[0, 1, 0.5], [1, 2, 0.1]], "adjacency": [[0, 1, 1.0]]}
    weights = {"semantic": 0.5, "adjacency": 0.5}
    edges = fuse_edges(signals, weights, threshold=0.06)
    assert edges == [[0, 1, 0.75]]                 # 0.5*0.5 + 0.5*1.0; edge 1-2 is 0.05 < 0.06


def test_matrix_neighbors_and_degrees():
    edges = [[0, 1, 0.5], [1, 2, 0.25]]
    matrix = build_matrix(edges, 3)
    assert np.allclose(matrix, matrix.T)
    assert matrix[1, 0] == 0.5
    assert neighbors(edges, 3) == {0: {1: 0.5}, 1: {0: 0.5, 2: 0.25}, 2: {1: 0.25}}
    assert np.allclose(degrees(edges, 3), [0.5, 0.75, 0.25])


def test_keep_strongest_bounds_degree_and_keeps_always_edges():
    star = [[0, j, 1.0 / j] for j in range(1, 8)] + [[6, 7, 0.01]]   # node 0 has 7 edges
    kept = keep_strongest(star, k=2)
    pairs = {(i, j) for i, j, _ in kept}
    assert (0, 1) in pairs and (0, 2) in pairs                      # node 0's two strongest
    assert (0, 5) in pairs                                          # node 5's only edge
    assert len(kept) <= 2 * 8
    kept = keep_strongest(star, k=1, always=[(6, 7)])
    assert (6, 7) in {(i, j) for i, j, _ in kept}                   # weak, but always kept


def test_keep_strongest_is_noop_when_k_is_large():
    edges = [[0, 1, 0.5], [1, 2, 0.25]]
    assert keep_strongest(edges, k=10) == edges
