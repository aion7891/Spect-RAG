"""Graph expansion (paper Section 3.6, Table 2 stage 2, Algorithm 1 lines 4-5).

Breadth-first search from the seeds S0, at most 'hops' steps away and at most
'max_candidates' chunks in total. The result S1 can contain chunks that are not similar to
the question but are linked to chunks that are: the "bridge" evidence.

'graph' is the CSR form from graph.neighbors(), so each step is a few numpy operations.
"""

import numpy as np


def gather(graph, nodes):
    """All (neighbour, weight) pairs of the given nodes, as two arrays."""
    indptr = graph["indptr"]
    slices = [np.arange(indptr[node], indptr[node + 1]) for node in nodes]
    positions = np.concatenate(slices) if slices else np.zeros(0, dtype=np.int64)
    return graph["indices"][positions], graph["weights"][positions]


def next_layer(frontier, graph, found):
    """New neighbours of the frontier, strongest link first (ties: lower id).

    'found' is a True/False array of the chunks already in the candidate set.
    """
    others, weights = gather(graph, frontier)
    new = ~found[others]
    strength = np.zeros(len(found))
    np.maximum.at(strength, others[new], weights[new])  # best link to the frontier
    layer = np.flatnonzero(strength > 0)
    return layer[np.lexsort((layer, -strength[layer]))].tolist(), strength


def expand_graph(seeds, graph, hops=2, max_candidates=200):
    """Return (candidates, hop) where hop[c] = 0 for seeds, 1 for their neighbours, ...

    If the cap is reached inside a layer, chunks with stronger links to the previous
    layer are kept first. Candidates are listed seeds first, then hop 1, then hop 2.
    """
    found = np.zeros(len(graph["indptr"]) - 1, dtype=bool)
    hop = {}
    for seed in seeds[:max_candidates]:
        hop[int(seed)] = 0
        found[seed] = True
    frontier = list(hop)
    for distance in range(1, hops + 1):
        room = max_candidates - len(hop)
        if room <= 0 or not frontier:
            break  # the cap is reached: no need to look further
        layer = next_layer(frontier, graph, found)[0][:room]
        for node in layer:
            hop[node] = distance
        found[layer] = True
        frontier = layer
    return list(hop), hop


def subgraph_matrix(graph, candidates):
    """Weighted adjacency matrix of the candidates only (the induced subgraph)."""
    position = np.full(len(graph["indptr"]) - 1, -1)
    position[candidates] = np.arange(len(candidates))
    matrix = np.zeros((len(candidates), len(candidates)))
    for row, node in enumerate(candidates):
        start, end = graph["indptr"][node], graph["indptr"][node + 1]
        columns = position[graph["indices"][start:end]]
        inside = columns >= 0
        matrix[row, columns[inside]] = graph["weights"][start:end][inside]
    return matrix
