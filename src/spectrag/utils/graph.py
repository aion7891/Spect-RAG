"""Combine the six edge signals into one weighted graph (paper Eq. 10, Section 3.2.2)."""

import numpy as np


def active_weights(edge_weights, signals):
    """Give the weight of empty signals (e.g. utility without an LLM) to the others.

    The weights of the signals that have edges are rescaled so they sum to 1 again.
    """
    used = {name: w for name, w in edge_weights.items() if signals.get(name)}
    total = sum(used.values())
    if total == 0:
        return {name: 0.0 for name in edge_weights}
    return {name: used.get(name, 0.0) / total for name in edge_weights}


def fuse_edges(signals, weights, threshold):
    """w_ij = sum of alpha_r * w_r,ij over all signals; keep edges with w_ij >= threshold."""
    total = {}
    for name, edges in signals.items():
        alpha = weights.get(name, 0.0)
        for i, j, weight in edges:
            total[(i, j)] = total.get((i, j), 0.0) + alpha * weight
    kept = [[i, j, round(w, 4)] for (i, j), w in sorted(total.items())
            if w > 0 and w >= threshold]
    return kept


def keep_strongest(edges, k, always=()):
    """Keep an edge only if it is among the k strongest edges of at least one of its chunks.

    This keeps |E| <= k * n, as the paper requires (|E| = O(k_g n), Section 3.2.2), even when
    a long section or a common entity would otherwise link every chunk to every other chunk.
    Pairs in 'always' (the adjacency edges) are never dropped, so each document stays connected.
    """
    always = {(i, j) for i, j in always}
    by_node = {}  # node -> [(weight, edge number), ...]
    keep = set()
    for number, (i, j, weight) in enumerate(edges):
        by_node.setdefault(i, []).append((weight, number))
        by_node.setdefault(j, []).append((weight, number))
        if (i, j) in always:
            keep.add(number)
    for items in by_node.values():
        items.sort(key=lambda item: (-item[0], item[1]))  # strongest first, ties by order
        keep.update(number for _, number in items[:k])
    return [edges[number] for number in sorted(keep)]


def build_matrix(edges, node_count):
    """Symmetric weighted adjacency matrix W (n x n) from an edge list."""
    matrix = np.zeros((node_count, node_count))
    for i, j, weight in edges:
        matrix[i, j] = weight
        matrix[j, i] = weight
    return matrix


def neighbors(edges, node_count):
    """Neighbour lists {node: {other_node: weight}} for walking the graph."""
    table = {node: {} for node in range(node_count)}
    for i, j, weight in edges:
        table[i][j] = weight
        table[j][i] = weight
    return table


def degrees(edges, node_count):
    """Weighted degree of every node: the sum of its edge weights (paper Eq. 28)."""
    degree = np.zeros(node_count)
    for i, j, weight in edges:
        degree[i] += weight
        degree[j] += weight
    return degree
