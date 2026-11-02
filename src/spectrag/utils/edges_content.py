"""Edge signals that come from chunk content (paper Section 3.2.1).

semantic - cosine similarity of chunk embeddings, only to each chunk's k nearest neighbours
entity   - Jaccard overlap of the chunks' entity sets
utility  - cosine similarity of the chunks' utility-question embeddings (k nearest neighbours)
"""

import numpy as np

from spectrag.utils.edge_list import add_edge, as_list
from spectrag.utils.embedder import normalize


def knn_edges(vectors, k):
    """Link each row to its k most similar rows, weight max(0, cosine) (paper Eq. 3)."""
    edges = {}
    if len(vectors) < 2:
        return as_list(edges)
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -1.0)  # never pick yourself
    for i in range(len(vectors)):
        for j in np.argsort(-similarities[i])[:k]:
            add_edge(edges, i, int(j), max(0.0, float(similarities[i, j])))
    return as_list(edges)


def semantic_edges(embeddings, k):
    """Semantic signal: k-nearest-neighbour cosine graph over chunk embeddings."""
    return knn_edges(normalize(embeddings), k)


def jaccard(set_a, set_b):
    """Shared items divided by all items."""
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def entity_edges(chunks):
    """Entity signal: Jaccard overlap, only for chunks that share at least one entity."""
    owners = {}  # entity -> ids of the chunks that mention it
    for chunk in chunks:
        for entity in chunk.get("entities", []):
            owners.setdefault(entity, []).append(chunk["id"])
    pairs = set()
    for ids in owners.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pairs.add((ids[a], ids[b]))
    sets = {chunk["id"]: set(chunk.get("entities", [])) for chunk in chunks}
    edges = {}
    for i, j in pairs:
        add_edge(edges, i, j, jaccard(sets[i], sets[j]))
    return as_list(edges)


def embed_questions(chunks, embed):
    """Embed every utility question. Returns (vectors, owners): owners[q] = chunk id."""
    texts, owners = [], []
    for chunk in chunks:
        for question in chunk.get("utility_questions", []):
            texts.append(question)
            owners.append(chunk["id"])
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), []
    return embed(texts, kind="document"), owners


def utility_edges(question_vectors, owners, chunk_count, k):
    """Utility signal: average each chunk's question vectors, then link nearest neighbours."""
    if len(owners) == 0:
        return []
    chunk_vectors = np.zeros((chunk_count, question_vectors.shape[1]), dtype=np.float32)
    for vector, owner in zip(question_vectors, owners):
        chunk_vectors[owner] += vector
    return knn_edges(normalize(chunk_vectors), k)
