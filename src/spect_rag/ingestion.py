"""Offline pipeline from the paper's Table 1.

Offline ingestion: prepare -> enrich -> graph -> spectrum -> save -> verify.
"""

from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

from .utility.chunking import chunk_document
from .utility.citations import resolve_citations
from .utility.embeddings import embed_texts
from .utility.extraction import extract_metadata
from .utility.loading import load_documents
from .utility.graph import build_chunk_graph, graph_settings
from .utility.indexing import load_index, save_index
from .utility.spectral import spectral_coordinates


class SpectRAGIngestion:
    """Load documents and build a reusable graph and spectral index.

    Pipeline: load -> chunk -> embed -> extract signals -> build graph
    -> compute spectral coordinates -> save index.
    """

    def __init__(
        self, index_dir: str | Path = "indexes/default", *,
        embed_fn: Callable | None = None, embedding_model: str | None = None,
        embedding_dimension: int = 768, document_prefix: str = "search_document: ",
        extract_fn: Callable | None = None, extraction_model: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embed_fn = embed_fn
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.document_prefix = document_prefix
        self.extract_fn = extract_fn
        self.extraction_model = extraction_model
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def prepare(
        self, document_paths: Iterable[str | Path], *,
        chunk_size: int = 1_200, chunk_overlap: int = 100,
        min_chunk_tokens: int = 32, semantic_chunking: bool = False,
    ) -> list[dict]:
        """Load and chunk; optional semantic boundaries use the embedding callback.

        By default this preserves Phase 2's model-free preparation. Semantic
        preparation may write model outputs only when cache_dir is configured.
        """
        if type(semantic_chunking) is not bool:
            raise TypeError("semantic_chunking must be a boolean.")
        if semantic_chunking and (self.embed_fn is None or not self.embedding_model):
            raise ValueError("Semantic chunking requires embed_fn and embedding_model.")
        sentence_embed_fn = partial(
            embed_texts, embed_fn=self.embed_fn, model_id=self.embedding_model,
            dimension=self.embedding_dimension, prefix=self.document_prefix,
            cache_dir=self.cache_dir,
        ) if semantic_chunking else None
        documents = load_documents(document_paths)
        chunks = []
        for document in documents:
            chunks.extend(chunk_document(
                document, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                min_chunk_tokens=min_chunk_tokens,
                sentence_embed_fn=sentence_embed_fn,
                embedding_dimension=self.embedding_dimension,
            ))
        return chunks

    def enrich(
        self, document_paths: Iterable[str | Path], *,
        chunk_size: int = 1_200, chunk_overlap: int = 100,
        min_chunk_tokens: int = 32, semantic_chunking: bool = True,
    ) -> list[dict]:
        """Prepare, embed, extract concepts/questions, and resolve local citations.

        No graph/index is built. Extraction is explicitly disabled if extract_fn
        is absent; no questions or concepts are fabricated as a fallback.
        """
        if self.embed_fn is None or not self.embedding_model:
            raise ValueError("Enrichment requires embed_fn and embedding_model.")
        if self.extract_fn is not None and not self.extraction_model:
            raise ValueError("Extraction requires an explicit extraction_model identity.")
        chunks = self.prepare(
            document_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            min_chunk_tokens=min_chunk_tokens, semantic_chunking=semantic_chunking,
        )
        embed = partial(
            embed_texts, embed_fn=self.embed_fn, model_id=self.embedding_model,
            dimension=self.embedding_dimension, prefix=self.document_prefix,
            cache_dir=self.cache_dir,
        )
        vectors = embed([chunk["text"] for chunk in chunks])
        metadata = [
            extract_metadata(
                chunk["text"], self.extract_fn, model_id=self.extraction_model,
                cache_dir=self.cache_dir,
            ) if self.extract_fn is not None else {"concepts": [], "utility_questions": []}
            for chunk in chunks
        ]
        question_vectors = embed([question for item in metadata for question in item["utility_questions"]])
        enriched = []
        offset = 0
        for chunk, vector, item in zip(chunks, vectors, metadata):
            count = len(item["utility_questions"])
            enriched.append({
                **chunk, **item, "embedding": vector,
                "utility_question_embeddings": question_vectors[offset:offset + count],
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dimension,
                "embedding_prefix": self.document_prefix,
                "extraction_model": self.extraction_model if self.extract_fn is not None else None,
                "metadata_extraction": "enabled" if self.extract_fn is not None else "disabled",
            })
            offset += count
        return resolve_citations(enriched)

    def ingest(
        self, document_paths: Iterable[str | Path], *,
        chunk_size: int = 1_200, chunk_overlap: int = 100, min_chunk_tokens: int = 32,
        semantic_chunking: bool = True, graph_neighbors: int = 10,
        relation_neighbors: int = 20, edge_threshold: float = 0.05,
        edge_weights: dict | None = None, spectral_dimension: int = 32,
    ) -> dict:
        """Build a complete local index and reload it to verify saved artifacts."""
        settings = graph_settings(
            graph_neighbors=graph_neighbors, relation_neighbors=relation_neighbors,
            edge_threshold=edge_threshold, edge_weights=edge_weights,
        )
        if type(spectral_dimension) is not int or spectral_dimension < 0:
            raise ValueError("Spectral dimension must be a non-negative integer.")
        chunks = self.enrich(
            document_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            min_chunk_tokens=min_chunk_tokens, semantic_chunking=semantic_chunking,
        )
        graph = build_chunk_graph(chunks, **settings)
        spectrum = spectral_coordinates(graph, dimension=spectral_dimension)
        save_index(
            self.index_dir, chunks, graph, spectrum,
            chunking_settings={
                "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                "min_chunk_tokens": min_chunk_tokens, "semantic_chunking": semantic_chunking,
            },
        )
        return load_index(
            self.index_dir, embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension, document_prefix=self.document_prefix,
        )
