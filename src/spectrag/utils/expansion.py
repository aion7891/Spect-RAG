"""Graph expansion (paper Section 3.6, Table 2 stage 2, Algorithm 1 lines 4-5).

Breadth-first search from the seeds S0, at most 'hops' steps away and at most
'max_candidates' chunks in total. The result S1 can contain chunks that are not similar to
the question but are linked to chunks that are: the "bridge" evidence.
"""

import numpy as np


def next_layer(frontier, table, found):
    """New neighbours of the frontier, strongest link first: {node: best edge weight}."""
    strength = {}
    for node in frontier:
        for other, weight in table[node].items():
            if other not in found:
                strength[other] = max(strength.get(other, 0.0), weight)
    return sorted(strength, key=lambda node: (-strength[node], node)), strength


def expand_graph(seeds, table, hops=2, max_candidates=200):
    """Return (candidates, hop) where hop[c] = 0 for seeds, 1 for their neighbours, ...

    If the cap is reached inside a layer, chunks with stronger links to the previous
    layer are kept first. Candidates are listed seeds first, then hop 1, then hop 2.
    """
    hop = {seed: 0 for seed in seeds[:max_candidates]}
    frontier = list(hop)
    for distance in range(1, hops + 1):
        layer, _ = next_layer(frontier, table, hop)
        layer = layer[:max_candidates - len(hop)]
        for node in layer:
            hop[node] = distance
        if not layer:
            break
        frontier = layer
    return list(hop), hop


def subgraph_matrix(table, candidates):
    """Weighted adjacency matrix of the candidates only (the induced subgraph)."""
    position = {node: k for k, node in enumerate(candidates)}
    matrix = np.zeros((len(candidates), len(candidates)))
    for node in candidates:
        for other, weight in table[node].items():
            if other in position:
                matrix[position[node], position[other]] = weight
    return matrix
