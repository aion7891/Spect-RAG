"""Build all six edge signals at once (paper Section 3.2.1 and Table 1, "Edge signals").

Before calling build_edge_signals(), each chunk needs:
  chunk["entities"]          - from entities.add_entities()
  chunk["references"]        - from references.add_references()
  chunk["utility_questions"] - from questions.add_utility_questions() (may be empty)
"""

from spectrag.utils.edges_content import (embed_questions, entity_edges, semantic_edges,
                                          utility_edges)
from spectrag.utils.edges_structure import adjacency_edges, citation_edges, section_edges

SIGNALS = ["semantic", "adjacency", "section", "entity", "utility", "citation"]


def build_edge_signals(chunks, embeddings, embed, settings):
    """Return ({signal name: [[i, j, weight], ...]}, question_vectors, question_owners)."""
    k = settings["semantic_k"]
    question_vectors, owners = embed_questions(chunks, embed)
    signals = {
        "semantic": semantic_edges(embeddings, k),
        "adjacency": adjacency_edges(chunks),
        "section": section_edges(chunks),
        "entity": entity_edges(chunks),
        "utility": utility_edges(question_vectors, owners, len(chunks), k),
        "citation": citation_edges(chunks),
    }
    return signals, question_vectors, owners


def count_edges(signals):
    """Number of edges per signal, e.g. {'semantic': 40, ...}."""
    return {name: len(signals[name]) for name in SIGNALS}
