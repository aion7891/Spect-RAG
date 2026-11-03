"""Spectral clusters for ClusterGain (paper Eq. 21).

Normalised spectral clustering (Ng, Jordan and Weiss; see von Luxburg [5]): for k clusters,
take the k smoothest eigenvectors of L_sym, scale every row to length 1, and run k-means.
"""

import numpy as np


def auto_cluster_count(node_count):
    """Assumed rule: about sqrt(n / 2) clusters, at least 2, at most n."""
    return max(1, min(node_count, max(2, round((node_count / 2) ** 0.5))))


def farthest_point_start(points, count):
    """Pick starting centres that are far apart (deterministic, no randomness)."""
    centres = [points[0]]
    nearest = np.sum((points - points[0]) ** 2, axis=1)  # distance to the closest centre
    while len(centres) < count:
        centres.append(points[int(np.argmax(nearest))])
        nearest = np.minimum(nearest, np.sum((points - centres[-1]) ** 2, axis=1))
    return np.array(centres, dtype=float)


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


def trivial_eigenvector(degree):
    """The first eigenvector of L_sym is sqrt(degree), scaled to length 1."""
    vector = np.sqrt(np.maximum(degree, 0))
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def cluster_coords(coords, count=None, degree=None):
    """Cluster label for every chunk.

    'coords' leaves out the trivial eigenvector, so we put it back from 'degree' and add the
    next k - 1 columns (for 2 clusters: the trivial one plus u2, the Fiedler vector).
    """
    node_count = len(coords)
    count = min(count or auto_cluster_count(node_count), node_count)
    if node_count == 0 or coords.shape[1] == 0 or count < 2:
        return np.zeros(node_count, dtype=int)
    columns = [coords[:, :count - 1]]  # high-frequency eigenvectors would hide communities
    if degree is not None:
        columns.insert(0, trivial_eigenvector(degree)[:, None])
    points = np.hstack(columns)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return kmeans(points / np.where(norms == 0, 1, norms), count)
