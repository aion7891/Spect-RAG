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
    kept = [[i, j, round(w, 4)] for (i, j), w in sorted(total.items()) if w >= threshold]
    return kept


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
    return build_matrix(edges, node_count).sum(axis=1)
