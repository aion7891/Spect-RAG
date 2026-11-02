"""Offline pipeline from the paper's Table 1.

Phase 1 defines the interface only. Later phases implement each indexing step
as a utility function and call those functions here in pipeline order.
"""

from collections.abc import Iterable
from pathlib import Path


class SpectRAGIngestion:
    """Load documents and build a reusable graph and spectral index.

    Planned flow: load -> chunk -> embed -> extract signals -> build graph
    -> compute spectral coordinates -> save index.
    """

    def __init__(self, index_dir: str | Path = "indexes/default") -> None:
        self.index_dir = Path(index_dir)

    def ingest(self, document_paths: Iterable[str | Path]) -> None:
        """Build an index from explicit file paths; not implemented in Phase 1."""
        raise NotImplementedError(
            "Phase 1 provides the package interface only. "
            "Document loading starts in Phase 2; indexing follows in Phases 3-4."
        )
