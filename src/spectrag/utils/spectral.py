"""Spectral coordinates and clusters (paper Eq. 11-13, Section 3.2.3 and 3.4)."""

import numpy as np


def normalized_laplacian(matrix):
    """L_sym = I - D^-1/2 W D^-1/2. Nodes without edges get 0 instead of 1/sqrt(0)."""
    degree = matrix.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    scaled = inv_sqrt[:, None] * matrix * inv_sqrt[None, :]
    return np.eye(len(matrix)) - scaled


def spectral_coords(laplacian, dims):
    """Eigenvectors 2 .. dims+1 of L_sym (the first one is trivial), one row per chunk.

    Returns (coords, eigenvalues). Small graphs give fewer than 'dims' columns.
    """
    values, vectors = np.linalg.eigh(laplacian)  # sorted from smallest to largest
    used = min(dims, len(values) - 1)
    if used <= 0:
        return np.zeros((len(values), 0)), values[:0]
    return vectors[:, 1:used + 1], values[1:used + 1]


def auto_cluster_count(node_count):
    """Assumed rule: about sqrt(n / 2) clusters, at least 2, at most n."""
    return max(1, min(node_count, max(2, round((node_count / 2) ** 0.5))))


def farthest_point_start(points, count):
    """Pick starting centres that are far apart (deterministic, no randomness)."""
    centres = [points[0]]
    while len(centres) < count:
        distances = np.min([np.sum((points - c) ** 2, axis=1) for c in centres], axis=0)
        centres.append(points[int(np.argmax(distances))])
    return np.array(centres)


def kmeans(points, count, steps=50):
    """Plain k-means: assign each point to its nearest centre, move centres, repeat."""
    centres = farthest_point_start(points, count)
    labels = np.zeros(len(points), dtype=int)
    for step in range(steps):
        distances = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if step > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(count):
            if np.any(labels == c):
                centres[c] = points[labels == c].mean(axis=0)
    return labels


def cluster_coords(coords, count=None):
    """Spectral clusters for ClusterGain (paper Eq. 21).

    Standard spectral clustering uses the k smoothest eigenvectors for k clusters. The trivial
    one is already left out of 'coords', so we take the first k - 1 columns
    (for 2 clusters that is just u2, the classic Fiedler split). Rows are scaled to length 1.
    """
    node_count = len(coords)
    count = min(count or auto_cluster_count(node_count), node_count)
    if node_count == 0 or coords.shape[1] == 0 or count < 2:
        return np.zeros(node_count, dtype=int)
    coords = coords[:, :count - 1]  # high-frequency eigenvectors would hide the communities
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    points = coords / np.where(norms == 0, 1, norms)
    return kmeans(points, count)
