"""SpectRAGIngestion: the offline pipeline (paper Table 1). Run it once per document set.

    ingest = SpectRAGIngestion("my_index")
    info = ingest.ingest("examples/docs")
"""

from spectrag.config import make_settings
from spectrag.utils.chunker import chunk_documents
from spectrag.utils.edges import build_edge_signals, count_edges
from spectrag.utils.embedder import make_embedder
from spectrag.utils.entities import add_entities
from spectrag.utils.graph import active_weights, build_matrix, fuse_edges
from spectrag.utils.loader import load_docs
from spectrag.utils.questions import add_utility_questions
from spectrag.utils.references import add_references
from spectrag.utils.spectral import cluster_coords, normalized_laplacian, spectral_coords
from spectrag.utils.store import save_index


class SpectRAGIngestion:
    """Load -> chunk -> embed -> edge signals -> weighted graph -> spectral -> save."""

    def __init__(self, index_dir, settings=None, entity_fn=None, question_fn=None):
        self.index_dir = index_dir
        self.settings = make_settings(settings)
        self.embed = make_embedder(self.settings)
        self.entity_fn = entity_fn        # optional: text -> list of entities
        self.question_fn = question_fn    # optional: text -> list of utility questions

    def make_chunks(self, path):
        """Load the documents and split them into chunks with entities, references, questions."""
        docs = load_docs(path)
        if not docs:
            raise ValueError("No documents with text found in " + str(path))
        chunks = chunk_documents(docs, self.settings, embed_fn=self.embed)
        add_entities(chunks, self.entity_fn)
        add_references(chunks)
        add_utility_questions(chunks, self.question_fn)
        return chunks

    def make_graph(self, signals):
        """Fuse the signals (Eq. 10) and drop weak edges. Returns (edges, weights used)."""
        weights = active_weights(self.settings["edge_weights"], signals)
        edges = fuse_edges(signals, weights, self.settings["edge_threshold"])
        return edges, weights

    def make_spectral(self, edges, node_count):
        """Laplacian (Eq. 11) -> spectral coordinates (Eq. 12-13) -> clusters."""
        laplacian = normalized_laplacian(build_matrix(edges, node_count))
        coords, eigenvalues = spectral_coords(laplacian, self.settings["spectral_dims"])
        clusters = cluster_coords(coords, self.settings["n_clusters"])
        return coords, eigenvalues, clusters

    def ingest(self, path):
        """Run the whole offline pipeline, save the index folder, and return a summary."""
        chunks = self.make_chunks(path)
        embeddings = self.embed([chunk["text"] for chunk in chunks], kind="document")
        signals, question_vectors, owners = build_edge_signals(
            chunks, embeddings, self.embed, self.settings)
        edges, weights = self.make_graph(signals)
        coords, eigenvalues, clusters = self.make_spectral(edges, len(chunks))
        for chunk, cluster in zip(chunks, clusters):
            chunk["cluster"] = int(cluster)
        info = {"documents": len({c["doc_id"] for c in chunks}), "chunks": len(chunks),
                "signal_edges": count_edges(signals), "graph_edges": len(edges),
                "weights_used": {name: round(w, 4) for name, w in weights.items()},
                "spectral_dims": coords.shape[1],
                "clusters": len(set(clusters.tolist()))}
        save_index(self.index_dir, {
            "chunks": chunks, "embeddings": embeddings, "signals": signals, "edges": edges,
            "spectral": coords, "eigenvalues": eigenvalues,
            "question_vectors": question_vectors, "question_owners": owners,
            "settings": self.settings, "info": info,
        })
        return info
