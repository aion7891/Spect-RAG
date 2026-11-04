"""Build all six edge signals at once (paper Section 3.2.1 and Table 1, "Edge signals").

Before calling build_edge_signals(), each chunk needs:
  chunk["entities"]          - from entities.add_entities()
  chunk["references"]        - from references.add_references()
  chunk["utility_questions"] - from questions.add_utility_questions() (may be empty)
"""

from spectrafuse.utils.edges_content import (embed_questions, entity_edges, semantic_edges,
                                          utility_edges)
from spectrafuse.utils.edges_structure import adjacency_edges, citation_edges, section_edges
from spectrafuse.utils.graph import keep_strongest

SIGNALS = ["semantic", "adjacency", "section", "entity", "utility", "citation"]


def build_edge_signals(chunks, embeddings, embed, settings, vector_index=None):
    """Return ({signal name: [[i, j, weight], ...]}, question_vectors, question_owners).

    Section and entity signals can link many chunks to each other (a long section, a name
    that appears often), so each chunk keeps only its graph_k strongest edges of that signal.
    """
    k = settings["semantic_k"]
    question_vectors, owners = embed_questions(chunks, embed)
    signals = {
        "semantic": semantic_edges(embeddings, k, vector_index),
        "adjacency": adjacency_edges(chunks),
        "section": keep_strongest(section_edges(chunks), settings["graph_k"]),
        "entity": keep_strongest(entity_edges(chunks), settings["graph_k"]),
        "utility": utility_edges(question_vectors, owners, len(chunks), k),
        "citation": citation_edges(chunks),
    }
    return signals, question_vectors, owners


def count_edges(signals):
    """Number of edges per signal, e.g. {'semantic': 40, ...}."""
    return {name: len(signals[name]) for name in SIGNALS}
