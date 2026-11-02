"""Six bounded edge-signal extractors (paper Section 3.2.1).

Edges use row indices into the input chunks. Nearest-neighbor computations are
exact and use O(n) temporary scores, not a materialized n-by-n matrix.
"""

from collections import defaultdict

import numpy as np


def _best(scores, source: int, neighbors: int):
    """Positive top neighbors with ties broken by corpus row order."""
    return [
        (int(target), float(scores[target]))
        for target in np.argsort(-scores, kind="stable")
        if target != source and scores[target] > 0
    ][:neighbors]


def semantic_edges(embeddings: np.ndarray, neighbors: int):
    for source, vector in enumerate(embeddings):
        scores = np.clip(embeddings @ vector, 0, 1)
        for target, weight in _best(scores, source, neighbors):
            yield source, target, weight


def adjacency_edges(chunks: list[dict]):
    documents = defaultdict(list)
    for row, chunk in enumerate(chunks):
        documents[chunk["doc_id"]].append((chunk["position"], row))
    for rows in documents.values():
        ordered = [row for _, row in sorted(rows)]
        for left, right in zip(ordered, ordered[1:]):
            yield left, right, 1.0


def section_edges(chunks: list[dict], neighbors: int):
    """Common section-prefix depth divided by maximum path depth, within a document."""
    roots = defaultdict(list)
    for row, chunk in enumerate(chunks):
        if chunk["section_path"]:
            roots[(chunk["doc_id"], chunk["section_path"][0])].append(row)
    for members in roots.values():
        for source in members:
            left = chunks[source]["section_path"]
            candidates = []
            for target in members:
                if source == target:
                    continue
                right = chunks[target]["section_path"]
                common = 0
                for a, b in zip(left, right):
                    if a != b:
                        break
                    common += 1
                candidates.append((common / max(len(left), len(right)), target))
            for weight, target in sorted(candidates, key=lambda pair: (-pair[0], pair[1]))[:neighbors]:
                yield source, target, weight


def entity_edges(chunks: list[dict], neighbors: int):
    """Case-insensitive concept-set Jaccard similarity."""
    concepts = [{value.casefold() for value in chunk["concepts"]} for chunk in chunks]
    inverted = defaultdict(set)
    for row, values in enumerate(concepts):
        for value in values:
            inverted[value].add(row)
    for source, values in enumerate(concepts):
        candidates = set()
        for value in values:
            candidates.update(inverted[value])
        scored = []
        for target in candidates - {source}:
            other = concepts[target]
            scored.append((len(values & other) / len(values | other), target))
        for weight, target in sorted(scored, key=lambda pair: (-pair[0], pair[1]))[:neighbors]:
            yield source, target, weight


def utility_edges(chunks: list[dict], question_vectors: np.ndarray, owners: np.ndarray, neighbors: int):
    """Maximum pairwise question cosine per chunk pair, truncated at zero."""
    for source, chunk in enumerate(chunks):
        own = chunk["utility_question_embeddings"]
        if len(own) == 0:
            continue
        per_question = np.clip(np.asarray(own) @ question_vectors.T, 0, 1).max(axis=0)
        scores = np.zeros(len(chunks))
        np.maximum.at(scores, owners, per_question)
        for target, weight in _best(scores, source, neighbors):
            yield source, target, weight


def citation_edges(chunks: list[dict], neighbors: int):
    """Resolved targets only; cap per citing chunk in stable corpus order."""
    rows = {chunk["chunk_id"]: row for row, chunk in enumerate(chunks)}
    for source, chunk in enumerate(chunks):
        targets = set()
        for citation in chunk["citations"]:
            if citation["status"] == "resolved":
                for target in citation["target_chunk_ids"]:
                    if target not in rows:
                        raise ValueError(f"Citation target is not a corpus chunk: {target}")
                    targets.add(rows[target])
        for target in sorted(targets - {source})[:neighbors]:
            yield source, target, 1.0
