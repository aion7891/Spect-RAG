from pathlib import Path

import numpy as np

from spectrag.config import make_settings
from spectrag.utils.chunker import chunk_documents
from spectrag.utils.edge_list import add_edge, as_list
from spectrag.utils.edges import SIGNALS, build_edge_signals, count_edges
from spectrag.utils.edges_content import entity_edges, jaccard, knn_edges, utility_edges
from spectrag.utils.edges_structure import (adjacency_edges, citation_edges, section_edges,
                                            section_weight)
from spectrag.utils.embedder import make_embedder
from spectrag.utils.entities import add_entities
from spectrag.utils.loader import load_docs
from spectrag.utils.questions import add_utility_questions
from spectrag.utils.references import add_references

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"


def chunk(id, doc, position, section=(), text="", **extra):
    return dict(id=id, doc_id=doc, position=position, section=list(section), text=text, **extra)


def test_add_edge_keeps_max_and_ignores_self_loops():
    edges = {}
    add_edge(edges, 2, 1, 0.3)
    add_edge(edges, 1, 2, 0.7)
    add_edge(edges, 1, 2, 0.5)
    add_edge(edges, 3, 3, 1.0)
    add_edge(edges, 4, 5, 0.0)
    add_edge(edges, 6, 7, 5.0)
    assert as_list(edges) == [[1, 2, 0.7], [6, 7, 1.0]]


def test_adjacency_only_within_a_document():
    chunks = [chunk(0, "a", 0), chunk(1, "a", 1), chunk(2, "b", 0), chunk(3, "b", 1)]
    assert adjacency_edges(chunks) == [[0, 1, 1.0], [2, 3, 1.0]]


def test_section_weight_rules():
    assert section_weight(["A", "1"], ["A", "1"]) == 1.0                    # same section
    assert section_weight(["A", "2", "2.1"], ["A", "2", "2.2"]) == 2 / 3   # siblings
    assert section_weight(["A", "2"], ["A", "2", "2.1"]) == 2 / 3          # parent and child
    assert section_weight(["A", "1"], ["A", "2", "2.1"]) == 0.0            # too distant
    assert section_weight([], ["A"]) == 0.0                                # semantic chunks


def test_section_edges_never_cross_documents():
    chunks = [chunk(0, "a", 0, ["X"]), chunk(1, "b", 0, ["X"]), chunk(2, "a", 1, ["X"])]
    assert section_edges(chunks) == [[0, 2, 1.0]]


def test_citation_edges_for_sections_and_files():
    chunks = [
        chunk(0, "a.md", 0, ["Doc", "2. Products", "2.1 Sensors"]),
        chunk(1, "a.md", 1, ["Doc", "2. Products", "2.2 Software"],
              references=[{"type": "section", "target": "2.1"}]),
        chunk(2, "a.md", 2, ["Doc", "3. Partners"], references=[{"type": "doc", "target": "b.md"}]),
        chunk(3, "sub/b.md", 0),
    ]
    assert citation_edges(chunks) == [[0, 1, 1.0], [2, 3, 1.0]]


def test_knn_edges_limits_neighbours_and_drops_negative():
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    edges = knn_edges(vectors, k=1)
    pairs = [(i, j) for i, j, _ in edges]
    assert (0, 1) in pairs
    assert all(w > 0 for _, _, w in edges)
    assert (0, 3) not in pairs                      # cosine -1 -> no edge


def test_entity_edges_use_jaccard():
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    chunks = [chunk(0, "a", 0, entities=["maya chen", "helios"]),
              chunk(1, "b", 0, entities=["maya chen"]),
              chunk(2, "c", 0, entities=["grapes"])]
    assert entity_edges(chunks) == [[0, 1, 0.5]]


def test_utility_edges_average_question_vectors():
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    edges = utility_edges(vectors, owners=[0, 1, 2], chunk_count=3, k=1)
    assert [0, 1, 1.0] in edges
    assert utility_edges(np.zeros((0, 0)), [], 3, 5) == []


def test_build_edge_signals_on_sample_docs():
    settings = make_settings({"semantic_k": 3})
    embed = make_embedder(settings)
    chunks = chunk_documents(load_docs(EXAMPLES), settings, embed)
    add_entities(chunks)
    add_references(chunks)
    add_utility_questions(chunks, lambda text: ["What is " + text.split()[1] + "?"])
    embeddings = embed([c["text"] for c in chunks])
    signals, question_vectors, owners = build_edge_signals(chunks, embeddings, embed, settings)
    assert sorted(signals) == sorted(SIGNALS)
    assert len(owners) == len(chunks) and question_vectors.shape[0] == len(chunks)
    for name in SIGNALS:
        assert count_edges(signals)[name] > 0
        for i, j, weight in signals[name]:
            assert i < j and 0 < weight <= 1
    entity_pairs = [(i, j) for i, j, _ in signals["entity"]]
    assert (0, 4) in entity_pairs                   # bridge: Aurora history <-> Helios (Maya Chen)
    assert [1, 2, 1.0] in signals["citation"]       # "Section 2.1" in the FieldView chunk
