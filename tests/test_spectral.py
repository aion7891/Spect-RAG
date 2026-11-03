import numpy as np
import pytest

from spectrag.utils.graph import build_matrix
from spectrag.utils.spectral import (compute_spectral, normalized_laplacian, spectral_coords,
                                     sparse_spectral_coords)

# Two triangles (0,1,2) and (3,4,5) joined by one weak bridge edge 2-3.
TWO_COMMUNITIES = [[0, 1, 1.0], [0, 2, 1.0], [1, 2, 1.0],
                   [3, 4, 1.0], [3, 5, 1.0], [4, 5, 1.0], [2, 3, 0.1]]


def random_graph(node_count, edge_count, seed):
    rng = np.random.default_rng(seed)
    edges = {}
    for _ in range(edge_count):
        i, j = sorted(rng.choice(node_count, 2, replace=False).tolist())
        edges[(i, j)] = float(rng.uniform(0.05, 1))
    for i in range(node_count - 1):                                  # keep it connected
        edges.setdefault((i, i + 1), 0.5)
    return [[i, j, w] for (i, j), w in sorted(edges.items())]


def test_laplacian_properties():
    laplacian = normalized_laplacian(build_matrix(TWO_COMMUNITIES, 6))
    values = np.linalg.eigvalsh(laplacian)
    assert np.allclose(laplacian, laplacian.T)
    assert np.allclose(np.diag(laplacian), 1.0)
    assert abs(values[0]) < 1e-9                 # connected graph: smallest eigenvalue is 0
    assert values.min() > -1e-9 and values.max() <= 2 + 1e-9


def test_isolated_node_does_not_break_laplacian():
    laplacian = normalized_laplacian(build_matrix([[0, 1, 1.0]], 3))
    assert np.all(np.isfinite(laplacian))
    assert laplacian[2, 2] == 1.0 and not laplacian[2, :2].any()


def test_spectral_coords_skip_trivial_and_separate_communities():
    laplacian = normalized_laplacian(build_matrix(TWO_COMMUNITIES, 6))
    coords, values = spectral_coords(laplacian, dims=32)
    assert coords.shape == (6, 5)                # capped at n - 1 for a small graph
    assert values[0] > 1e-6                      # the trivial eigenvalue 0 was skipped
    fiedler = coords[:, 0]                       # u2 splits the two triangles by sign
    assert np.sign(fiedler[0]) == np.sign(fiedler[1]) == np.sign(fiedler[2])
    assert np.sign(fiedler[3]) == np.sign(fiedler[4]) == np.sign(fiedler[5])
    assert np.sign(fiedler[0]) != np.sign(fiedler[3])


def test_spectral_coords_tiny_graphs():
    coords, values = spectral_coords(normalized_laplacian(np.zeros((1, 1))), 32)
    assert coords.shape == (1, 0) and len(values) == 0
    coords, values = compute_spectral([], 1, 32)
    assert coords.shape == (1, 0)


def test_sparse_solver_matches_dense():
    pytest.importorskip("scipy")
    edges = random_graph(300, 900, seed=0)
    dense_coords, dense_values = spectral_coords(normalized_laplacian(build_matrix(edges, 300)), 8)
    sparse_coords, sparse_values = sparse_spectral_coords(edges, 300, 8)
    assert np.allclose(dense_values, sparse_values, atol=1e-6)
    laplacian = normalized_laplacian(build_matrix(edges, 300))
    for column, value in zip(sparse_coords.T, sparse_values):   # L u = lambda u
        assert np.allclose(laplacian @ column, value * column, atol=1e-5)


def test_compute_spectral_uses_sparse_above_the_limit():
    pytest.importorskip("scipy")
    edges = random_graph(120, 300, seed=3)
    dense = compute_spectral(edges, 120, 6, dense_limit=1000)
    sparse = compute_spectral(edges, 120, 6, dense_limit=50)
    assert np.allclose(dense[1], sparse[1], atol=1e-6)
    assert np.allclose(np.abs(dense[0]), np.abs(sparse[0]), atol=1e-4)
