import numpy as np

from spectrag.utils.graph import build_matrix
from spectrag.utils.spectral import (auto_cluster_count, cluster_coords, kmeans,
                                     normalized_laplacian, spectral_coords)

# Two triangles (0,1,2) and (3,4,5) joined by one weak bridge edge 2-3.
TWO_COMMUNITIES = [[0, 1, 1.0], [0, 2, 1.0], [1, 2, 1.0],
                   [3, 4, 1.0], [3, 5, 1.0], [4, 5, 1.0], [2, 3, 0.1]]


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


def test_clusters_match_communities():
    laplacian = normalized_laplacian(build_matrix(TWO_COMMUNITIES, 6))
    coords, _ = spectral_coords(laplacian, 32)
    labels = cluster_coords(coords, count=2)
    assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_kmeans_and_cluster_count():
    points = np.array([[0, 0], [0, 0.1], [5, 5], [5, 5.1]])
    labels = kmeans(points, 2)
    assert labels[0] == labels[1] != labels[2] == labels[3]
    assert auto_cluster_count(8) == 2
    assert auto_cluster_count(200) == 10
    assert auto_cluster_count(1) == 1
    assert list(cluster_coords(np.zeros((3, 0)))) == [0, 0, 0]
