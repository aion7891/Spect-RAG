"""Online pipeline from the paper's Table 2 and Algorithm 1.

Phase 1 defines the interface only. Retrieval will expose context independently
of optional answer generation so it can be inspected and tested on its own.
"""

from pathlib import Path
from typing import Any


class SpectRAGRetrieval:
    """Retrieve a token-budgeted evidence set from a persisted index.

    Planned flow: hybrid seeds -> graph expansion -> proximity scoring
    -> context selection -> return ranked chunks with source metadata.
    """

    def __init__(self, index_dir: str | Path = "indexes/default") -> None:
        self.index_dir = Path(index_dir)

    def retrieve(
        self, query: str, *, token_budget: int = 12_000
    ) -> list[dict[str, Any]]:
        """Return evidence records; not implemented until Phase 5.

        Planned record fields include chunk_id, doc_id, text, score,
        token_count, source path, page, and section path.
        """
        raise NotImplementedError(
            "Phase 1 provides the package interface only. "
            "Hybrid and graph-spectral retrieval will be implemented in Phase 5."
        )
