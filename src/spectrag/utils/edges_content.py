"""Edge signals that come from chunk content (paper Section 3.2.1).

semantic - cosine similarity of chunk embeddings, only to each chunk's k nearest neighbours
entity   - Jaccard overlap of the chunks' entity sets
utility  - cosine similarity of the chunks' utility-question embeddings (k nearest neighbours)
"""

import numpy as np

from spectrag.utils.edge_list import add_edge, as_list
from spectrag.utils.embedder import normalize

# An entity found in more than 1% of the chunks (and more than 20 chunks) is too common to
# say two chunks are related, like a stopword. Skipping it also avoids n x n pair counting.
COMMON_ENTITY_SHARE = 0.01
COMMON_ENTITY_MIN = 20


def knn_edges(vectors, k, block_size=1024):
    """Link each row to its k most similar rows, weight max(0, cosine) (paper Eq. 3).

    Rows are handled in blocks, so memory stays at block_size x n instead of n x n.
    """
    edges = {}
    k = min(k, len(vectors) - 1)
    if k < 1:
        return as_list(edges)
    for start in range(0, len(vectors), block_size):
        similarities = vectors[start:start + block_size] @ vectors.T
        rows = np.arange(len(similarities))
        similarities[rows, start + rows] = -1.0  # never pick yourself
        # argpartition finds the k largest without sorting the whole row.
        nearest = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
        for row, columns in enumerate(nearest):
            for j in columns:
                add_edge(edges, start + row, int(j), max(0.0, float(similarities[row, j])))
    return as_list(edges)


def semantic_edges(embeddings, k):
    """Semantic signal: k-nearest-neighbour cosine graph over chunk embeddings."""
    return knn_edges(normalize(embeddings), k)


def entity_edges(chunks):
    """Entity signal: Jaccard overlap, only for chunks that share at least one entity.

    Shared entities are counted while pairs are found, so no set is intersected twice:
    Jaccard = shared / (size_a + size_b - shared).
    """
    owners = {}  # entity -> ids of the chunks that mention it
    sizes = {}
    for chunk in chunks:
        entities = set(chunk.get("entities", []))
        sizes[chunk["id"]] = len(entities)
        for entity in entities:
            owners.setdefault(entity, []).append(chunk["id"])
    too_common = max(COMMON_ENTITY_MIN, COMMON_ENTITY_SHARE * len(chunks))
    shared = {}  # (i, j) -> number of shared entities
    for ids in owners.values():
        if len(ids) > too_common:
            continue
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pair = (ids[a], ids[b])
                shared[pair] = shared.get(pair, 0) + 1
    edges = {}
    for (i, j), common in shared.items():
        add_edge(edges, i, j, common / (sizes[i] + sizes[j] - common))
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
