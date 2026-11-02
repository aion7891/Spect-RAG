"""Fuse six signals into a sparse undirected weighted chunk graph (Equation 10)."""

import math

import networkx as nx

from .graph_signals import (
    adjacency_edges, citation_edges, entity_edges, section_edges, semantic_edges, utility_edges,
)
from .validation import validate_chunks


DEFAULT_EDGE_WEIGHTS = {
    "semantic": 0.300, "adjacency": 0.084, "section": 0.250,
    "entity": 0.200, "utility": 0.083, "citation": 0.083,
}


def graph_settings(
    *, graph_neighbors: int = 10, relation_neighbors: int = 20,
    edge_threshold: float = 0.05, edge_weights: dict | None = None,
) -> dict:
    """Validate explicit implementation defaults; weights override named signals."""
    for value in (graph_neighbors, relation_neighbors):
        if type(value) is not int or value < 0:
            raise ValueError("Graph neighbor limits must be non-negative integers.")
    if isinstance(edge_threshold, bool) or not math.isfinite(edge_threshold) or edge_threshold < 0:
        raise ValueError("Edge threshold must be finite and non-negative.")
    weights = {**DEFAULT_EDGE_WEIGHTS, **(edge_weights or {})}
    if set(weights) != set(DEFAULT_EDGE_WEIGHTS):
        raise ValueError("Unknown graph edge-signal name.")
    if any(isinstance(w, bool) or not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Signal weights must be finite and non-negative.")
    if sum(weights.values()) <= 0:
        raise ValueError("At least one edge-signal weight must be positive.")
    return {
        "graph_neighbors": graph_neighbors, "relation_neighbors": relation_neighbors,
        "edge_threshold": float(edge_threshold),
        "edge_weights": {key: float(value) for key, value in weights.items()},
    }


def build_chunk_graph(chunks: list[dict], **options) -> nx.Graph:
    """Keep every chunk, combine duplicate signal pairs by max, then threshold."""
    settings = graph_settings(**options)
    graph = nx.Graph()
    graph.add_nodes_from(chunk["chunk_id"] for chunk in chunks)
    graph.graph["settings"] = settings
    pairs = {}
    if chunks:
        embeddings, questions, owners = validate_chunks(chunks)
        k, limit = settings["graph_neighbors"], settings["relation_neighbors"]
        signals = {
            "semantic": lambda: semantic_edges(embeddings, k),
            "adjacency": lambda: adjacency_edges(chunks),
            "section": lambda: section_edges(chunks, limit),
            "entity": lambda: entity_edges(chunks, limit),
            "utility": lambda: utility_edges(chunks, questions, owners, k),
            "citation": lambda: citation_edges(chunks, limit),
        }
        for signal, extract in signals.items():
            if settings["edge_weights"][signal] == 0:
                continue
            if signal != "adjacency" and (k if signal in {"semantic", "utility"} else limit) == 0:
                continue
            for left, right, value in extract():
                pair = tuple(sorted((left, right)))
                attributes = pairs.setdefault(pair, {name: 0.0 for name in DEFAULT_EDGE_WEIGHTS})
                attributes[signal] = max(attributes[signal], float(value))
    for (left, right), attributes in sorted(pairs.items()):
        weight = sum(settings["edge_weights"][name] * value for name, value in attributes.items())
        if weight > 0 and weight >= settings["edge_threshold"]:
            graph.add_edge(chunks[left]["chunk_id"], chunks[right]["chunk_id"],
                           weight=float(weight), **attributes)
    graph.graph["stats"] = {
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "candidate_edges": len(pairs), "isolated_nodes": len(list(nx.isolates(graph))),
        "components": nx.number_connected_components(graph),
        "signal_edges": {
            name: sum(data[name] > 0 for _, _, data in graph.edges(data=True))
            for name in DEFAULT_EDGE_WEIGHTS
        },
    }
    return graph
