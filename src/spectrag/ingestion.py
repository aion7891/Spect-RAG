"""SpectRAGIngestion: the offline pipeline (paper Table 1). Run it once per document set.

    ingest = SpectRAGIngestion("my_index")
    info = ingest.ingest("examples/docs")
"""

import shutil
import time
from pathlib import Path

import numpy as np

from spectrag._version import INDEX_FORMAT, __version__
from spectrag.config import make_settings
from spectrag.errors import DocumentError, IndexError_, NoDocumentsError
from spectrag.log import log_step, logger
from spectrag.utils.chunker import chunk_documents
from spectrag.utils.edges import build_edge_signals, count_edges
from spectrag.utils.embedder import make_embedder
from spectrag.utils.embedder_id import saved_settings
from spectrag.utils.entities import add_entities
from spectrag.utils.graph_build import build_graph, build_spectral
from spectrag.utils.loader import load_docs
from spectrag.utils.questions import add_utility_questions
from spectrag.utils.references import add_references
from spectrag.utils.store import edges_to_array, save_index
from spectrag.utils.vector_index import build_vector_index, save_vector_index


class SpectRAGIngestion:
    """Load -> chunk -> embed -> edge signals -> weighted graph -> spectral -> save."""

    def __init__(self, index_dir, settings=None, entity_fn=None, question_fn=None):
        self.index_dir = index_dir
        self.settings = make_settings(settings)
        self.embed = make_embedder(self.settings)
        # Semantic chunking can use a faster embedder for sentence distances (None = the same).
        split_choice = self.settings["semantic_embedder"]
        self.split_embed = self.embed if split_choice is None else \
            make_embedder(dict(self.settings, embedder=split_choice))
        self.entity_fn = entity_fn        # optional: text -> list of entities
        self.question_fn = question_fn    # optional: text -> list of utility questions

    def make_chunks(self, docs):
        """Split documents into chunks with entities, references and utility questions."""
        chunks = chunk_documents(docs, self.settings, embed_fn=self.split_embed)
        add_entities(chunks, self.entity_fn)
        add_references(chunks)
        add_utility_questions(chunks, self.question_fn)
        return chunks

    def make_graph(self, signals):
        """Fuse the signals (Eq. 10), drop weak edges, keep each chunk's strongest graph_k."""
        return build_graph(signals, self.settings)

    def make_spectral(self, edges, node_count):
        """Laplacian (Eq. 11) -> spectral coordinates (Eq. 12-13) -> clusters."""
        return build_spectral(edges, node_count, self.settings)

    def ingest(self, path):
        """Ingest a file or a folder (.txt, .md, .pdf), save the index, return a summary."""
        docs = load_docs(path)
        if not docs:
            raise NoDocumentsError("No .txt, .md or .pdf file with text found in " + str(path))
        return self.ingest_documents(docs)

    def ingest_documents(self, docs):
        """Ingest documents already in memory: [{"doc_id": ..., "text": ...}, ...]."""
        check_documents(docs)
        start = begin = time.time()
        chunks = self.make_chunks(docs)
        start = log_step("chunking", start, documents=len(docs), chunks=len(chunks))
        embeddings = self.embed([chunk["text"] for chunk in chunks], kind="document")
        vector_index = build_vector_index(embeddings, self.settings)  # exact, FAISS or HNSW
        start = log_step("embedding", start, backend=vector_index["backend"])
        signals, question_vectors, owners = build_edge_signals(
            chunks, embeddings, self.embed, self.settings, vector_index)
        start = log_step("edge signals", start, **count_edges(signals))
        edges, weights = self.make_graph(signals)
        coords, eigenvalues, clusters = self.make_spectral(edges, len(chunks))
        start = log_step("graph and spectral", start, edges=len(edges), dims=coords.shape[1])
        for chunk, cluster in zip(chunks, clusters):
            chunk["cluster"] = int(cluster)
        info = {"documents": len({c["doc_id"] for c in chunks}), "chunks": len(chunks),
                "signal_edges": count_edges(signals), "graph_edges": len(edges),
                "weights_used": {name: round(w, 4) for name, w in weights.items()},
                "spectral_dims": coords.shape[1], "ann_backend": vector_index["backend"],
                "clusters": len(set(clusters.tolist())),
                "format_version": INDEX_FORMAT, "spectrag_version": __version__,
                "seconds": round(time.time() - begin, 2)}
        partial = prepare_folder(self.index_dir)
        save_index(partial, {
            "chunks": chunks, "embeddings": embeddings, "edges": edges_to_array(edges),
            "signals": {name: edges_to_array(found) for name, found in signals.items()},
            "spectral": coords, "eigenvalues": eigenvalues,
            "question_vectors": question_vectors,
            "question_owners": np.asarray(owners, dtype=np.int64),
            "settings": saved_settings(self.settings), "info": info,  # functions saved by name
        })
        save_vector_index(vector_index, partial)
        replace_folder(partial, self.index_dir)
        log_step("saved index", start, folder=self.index_dir)
        return info


def check_documents(docs):
    """Every document needs a text and a unique doc_id."""
    if not docs:
        raise NoDocumentsError("No documents with text to ingest")
    seen = set()
    for number, doc in enumerate(docs):
        if not isinstance(doc, dict) or not isinstance(doc.get("text"), str) or "doc_id" not in doc:
            raise DocumentError("Document %d must be a dict with 'doc_id' and 'text' (a string)" % number)
        if doc["doc_id"] in seen:
            raise DocumentError("Two documents have the same doc_id: %r" % (doc["doc_id"],))
        seen.add(doc["doc_id"])
    if not any(doc["text"].strip() for doc in docs):
        raise NoDocumentsError("Every document is empty")


def is_index(folder):
    return (folder / "info.json").is_file() and (folder / "chunks.json").is_file()


def prepare_folder(index_dir):
    """A fresh '<index_dir>.partial' folder to write into, so a crash never leaves half an index."""
    target = Path(index_dir)
    if target.exists() and (not target.is_dir() or (any(target.iterdir()) and not is_index(target))):
        raise IndexError_("%s already exists and is not a SpectRAG index; choose another folder" % target)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    return partial


def replace_folder(partial, index_dir):
    """Move the finished index into place (replacing an older SpectRAG index)."""
    target = Path(index_dir)
    if target.exists():
        logger.info("replacing the existing index in %s", target)
        shutil.rmtree(target)
    partial.rename(target)
