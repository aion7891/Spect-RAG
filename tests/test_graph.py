import numpy as np
import pytest

from spectrag.utils.graph import active_weights, build_matrix, degrees, fuse_edges, neighbors

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
