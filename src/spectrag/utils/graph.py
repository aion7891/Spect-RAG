"""Combine the six edge signals into one weighted graph (paper Eq. 10, Section 3.2.2).

Edge lists are [[i, j, weight], ...]. They may also come from a saved index as a float array
of shape (E, 3), so chunk ids are always turned back into int before they are used.
"""

import numpy as np


def active_weights(edge_weights, signals):
    """Give the weight of empty signals (e.g. utility without an LLM) to the others.

    The weights of the signals that have edges are rescaled so they sum to 1 again.
    """
    used = {name: w for name, w in edge_weights.items() if len(signals.get(name, [])) > 0}
    total = sum(used.values())
    if total == 0:
        return {name: 0.0 for name in edge_weights}
    return {name: used.get(name, 0.0) / total for name in edge_weights}


def fuse_edges(signals, weights, threshold):
    """w_ij = sum of alpha_r * w_r,ij over all signals; keep edges with w_ij >= threshold."""
    total = {}
    for name, edges in signals.items():
        alpha = weights.get(name, 0.0)
        if alpha == 0:
            continue
        for i, j, weight in edges:
            key = (int(i), int(j))
            total[key] = total.get(key, 0.0) + alpha * float(weight)
    kept = [[i, j, round(w, 4)] for (i, j), w in sorted(total.items())
            if w > 0 and w >= threshold]
    return kept


def keep_strongest(edges, k, always=()):
    """Keep an edge only if it is among the k strongest edges of at least one of its chunks.

    This keeps |E| <= k * n, as the paper requires (|E| = O(k_g n), Section 3.2.2), even when
    a long section or a common entity would otherwise link every chunk to every other chunk.
    Pairs in 'always' (the adjacency edges) are never dropped, so each document stays connected.
    """
    always = {(int(i), int(j)) for i, j in always}
    by_node = {}  # node -> [(weight, edge number), ...]
    keep = set()
    for number, (i, j, weight) in enumerate(edges):
        i, j = int(i), int(j)
        by_node.setdefault(i, []).append((weight, number))
        by_node.setdefault(j, []).append((weight, number))
        if (i, j) in always:
            keep.add(number)
    for items in by_node.values():
        items.sort(key=lambda item: (-item[0], item[1]))  # strongest first, ties by order
        keep.update(number for _, number in items[:k])
    return [edges[number] for number in sorted(keep)]


def build_matrix(edges, node_count):
    """Symmetric weighted adjacency matrix W (n x n) from an edge list (dense solver only)."""
    matrix = np.zeros((node_count, node_count))
    for i, j, weight in edges:
        matrix[int(i), int(j)] = weight
        matrix[int(j), int(i)] = weight
    return matrix


def neighbors(edges, node_count):
    """The graph as compressed sparse rows (CSR), built once per loaded index.

    The neighbours of node v are indices[indptr[v]:indptr[v + 1]], with the same slice of
    weights. Both directions of every edge are stored, sorted by node and then neighbour.
    """
    edges = np.asarray(edges, dtype=float).reshape(-1, 3)
    rows = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    cols = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    weights = np.concatenate([edges[:, 2], edges[:, 2]])
    order = np.lexsort((cols, rows))
    indptr = np.zeros(node_count + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=node_count), out=indptr[1:])
    return {"indptr": indptr, "indices": cols[order], "weights": weights[order]}


def degrees(edges, node_count):
    """Weighted degree of every node: the sum of its edge weights (paper Eq. 28)."""
    degree = np.zeros(node_count)
    for i, j, weight in edges:
        degree[int(i)] += weight
        degree[int(j)] += weight
    return degree
