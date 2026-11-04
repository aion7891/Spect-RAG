import numpy as np

from spectrafuse.utils.expansion import expand_graph, next_layer, subgraph_matrix
from spectrafuse.utils.graph import neighbors

PATH = neighbors([[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0], [3, 4, 1.0]], 5)   # 0-1-2-3-4


def test_two_hops_on_a_path():
    candidates, hop = expand_graph([0], PATH, hops=2, max_candidates=200)
    assert candidates == [0, 1, 2]
    assert hop == {0: 0, 1: 1, 2: 2}


def test_seeds_first_and_no_duplicates():
    candidates, hop = expand_graph([2, 0], PATH, hops=1, max_candidates=200)
    assert candidates == [2, 0, 1, 3]                  # seeds first, in seed order
    assert hop[1] == 1 and hop[3] == 1


def test_cap_keeps_strongest_links():
    star = neighbors([[0, 1, 0.1], [0, 2, 0.9], [0, 3, 0.5], [0, 4, 0.7]], 5)
    candidates, _ = expand_graph([0], star, hops=2, max_candidates=3)
    assert candidates == [0, 2, 4]
    found = np.array([True, False, False, False, False])
    assert next_layer([0], star, found)[0] == [2, 4, 3, 1]


def test_isolated_seed_and_zero_hops():
    table = neighbors([[0, 1, 1.0]], 3)
    assert expand_graph([2], table, hops=2)[0] == [2]
    assert expand_graph([0], table, hops=0)[0] == [0]


def test_subgraph_matrix_keeps_only_candidates():
    matrix = subgraph_matrix(PATH, [1, 2, 4])
    assert np.allclose(matrix, matrix.T)
    assert matrix[0, 1] == 1.0                          # edge 1-2
    assert not matrix[:, 2].any()                       # 4 has no edge inside {1, 2, 4}
