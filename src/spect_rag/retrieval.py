"""Online pipeline from the paper's Table 2 and Algorithm 1.

Retrieval is planned for Phase 5. It will expose context independently
of optional answer generation so it can be inspected and tested on its own.
"""

from pathlib import Path
from typing import Any

from .utility.indexing import load_index


class SpectRAGRetrieval:
    """Retrieve a token-budgeted evidence set from a persisted index.

    Planned flow: hybrid seeds -> graph expansion -> proximity scoring
    -> context selection -> return ranked chunks with source metadata.
    """

    def __init__(self, index_dir: str | Path = "indexes/default") -> None:
        self.index_dir = Path(index_dir)

    def load(self) -> dict:
        """Load and verify the Phase 4 index without running query retrieval."""
        return load_index(self.index_dir)

    def retrieve(
        self, query: str, *, token_budget: int = 12_000
    ) -> list[dict[str, Any]]:
        """Return evidence records; not implemented until Phase 5.

        Planned record fields include chunk_id, doc_id, text, score,
        token_count, source path, page, and section path.
        """
        raise NotImplementedError(
            "Hybrid and graph-spectral retrieval will be implemented in Phase 5. "
            "Use load() to inspect a saved graph/spectral index now."
        )
