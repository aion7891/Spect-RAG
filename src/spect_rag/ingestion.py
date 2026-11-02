"""Offline pipeline from the paper's Table 1.

Phase 2 implements inspectable document preparation. Later phases add embedding,
graph construction, spectral coordinates, and index persistence.
"""

from collections.abc import Iterable
from pathlib import Path

from .utility.chunking import chunk_document
from .utility.loading import load_documents


class SpectRAGIngestion:
    """Load documents and build a reusable graph and spectral index.

    Planned flow: load -> chunk -> embed -> extract signals -> build graph
    -> compute spectral coordinates -> save index.
    """

    def __init__(self, index_dir: str | Path = "indexes/default") -> None:
        self.index_dir = Path(index_dir)

    def prepare(
        self, document_paths: Iterable[str | Path], *,
        chunk_size: int = 1_200, chunk_overlap: int = 100,
        min_chunk_tokens: int = 32,
    ) -> list[dict]:
        """Load and chunk documents without models or writing index artifacts."""
        documents = load_documents(document_paths)
        chunks = []
        for document in documents:
            chunks.extend(chunk_document(
                document, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                min_chunk_tokens=min_chunk_tokens,
            ))
        return chunks

    def ingest(self, document_paths: Iterable[str | Path]) -> None:
        """Build a full index; implemented once Phases 3-4 are complete."""
        raise NotImplementedError(
            "Use prepare() to load and inspect document chunks in Phase 2. "
            "Full indexing follows in Phases 3-4."
        )
