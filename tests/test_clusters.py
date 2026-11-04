import numpy as np

from spectrafuse.utils.clusters import (auto_cluster_count, cluster_coords, kmeans,
                                     trivial_eigenvector)
from spectrafuse.utils.graph import build_matrix, degrees
from spectrafuse.utils.spectral import normalized_laplacian, spectral_coords

# Two triangles (0,1,2) and (3,4,5) joined by one weak bridge edge 2-3.
TWO_COMMUNITIES = [[0, 1, 1.0], [0, 2, 1.0], [1, 2, 1.0],
                   [3, 4, 1.0], [3, 5, 1.0], [4, 5, 1.0], [2, 3, 0.1]]


def test_clusters_match_communities_with_and_without_degree():
    coords, _ = spectral_coords(normalized_laplacian(build_matrix(TWO_COMMUNITIES, 6)), 32)
    for degree in [None, degrees(TWO_COMMUNITIES, 6)]:
        labels = cluster_coords(coords, count=2, degree=degree)
        assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1
        assert labels[0] != labels[3]


def test_trivial_eigenvector_is_the_first_eigenvector():
    laplacian = normalized_laplacian(build_matrix(TWO_COMMUNITIES, 6))
    vector = trivial_eigenvector(degrees(TWO_COMMUNITIES, 6))
    assert np.allclose(laplacian @ vector, 0)                      # eigenvalue 0
    assert np.isclose(np.linalg.norm(vector), 1)


def test_three_communities():
    edges = []
    for start in [0, 4, 8]:                                         # three groups of 4
        for a in range(start, start + 4):
            for b in range(a + 1, start + 4):
                edges.append([a, b, 1.0])
    edges += [[3, 4, 0.1], [7, 8, 0.1]]
    coords, _ = spectral_coords(normalized_laplacian(build_matrix(edges, 12)), 32)
    labels = cluster_coords(coords, count=3, degree=degrees(edges, 12))
    groups = [set(labels[s:s + 4].tolist()) for s in [0, 4, 8]]
    assert all(len(g) == 1 for g in groups)
    assert len(set(labels.tolist())) == 3


def test_kmeans_and_cluster_count():
    points = np.array([[0, 0], [0, 0.1], [5, 5], [5, 5.1]])
    labels = kmeans(points, 2)
    assert labels[0] == labels[1] != labels[2] == labels[3]
    assert auto_cluster_count(8) == 2
    assert auto_cluster_count(200) == 10
    assert auto_cluster_count(1) == 1
    assert list(cluster_coords(np.zeros((3, 0)))) == [0, 0, 0]
