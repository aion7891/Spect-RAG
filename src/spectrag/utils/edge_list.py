"""Tiny helpers for edge lists.

While building, edges live in a dict {(i, j): weight} with i < j, so each pair is stored once.
The final result is a sorted list of [i, j, weight], which can be saved as JSON.
"""


def add_edge(edges, i, j, weight):
    """Add an undirected edge, keeping the larger weight if it already exists."""
    if i == j or weight <= 0:
        return
    key = (min(i, j), max(i, j))
    edges[key] = max(edges.get(key, 0.0), min(float(weight), 1.0))


def as_list(edges):
    """Turn the edge dict into a sorted list of [i, j, weight]."""
    return [[i, j, round(weight, 4)] for (i, j), weight in sorted(edges.items())]
