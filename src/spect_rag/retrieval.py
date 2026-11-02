"""Online pipeline from the paper's Table 2 and Algorithm 1.

Phase 5 returns ranked, budgeted evidence independently of answer generation.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .utility.embeddings import embed_texts
from .utility.expansion import expand_candidates
from .utility.indexing import load_index
from .utility.proximity import score_candidates, spectral_support
from .utility.ranking import initial_retrieval, retrieval_settings
from .utility.retrieval_trace import evidence_records, retrieval_trace
from .utility.selection import select_context


class SpectRAGRetrieval:
    """Retrieve a token-budgeted evidence set from a persisted index.

    Pipeline: hybrid seeds -> graph expansion -> proximity scoring
    -> context selection -> return ranked chunks with source metadata.
    """

    def __init__(
        self, index_dir: str | Path = "indexes/default", *,
        embed_fn: Callable | None = None, embedding_model: str | None = None,
        embedding_dimension: int | None = None, query_prefix: str = "search_query: ",
        document_prefix: str | None = None, cache_dir: str | Path | None = None,
    ) -> None:
        if embedding_dimension is not None and (
            type(embedding_dimension) is not int or embedding_dimension <= 0
        ):
            raise ValueError("embedding_dimension must be a positive integer or None.")
        self.index_dir = Path(index_dir)
        self.embed_fn = embed_fn
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.last_trace: dict | None = None
        self._index = None
        self._spectral_support = None

    def load(self) -> dict:
        """Verify/reload CURRENT, then pin that snapshot for subsequent queries."""
        index = load_index(
            self.index_dir, embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension, document_prefix=self.document_prefix,
        )
        self._spectral_support = spectral_support(index["graph"], index["spectral_coords"])
        self._index = index
        self.last_trace = None
        return index

    def retrieve(
        self, query: str, *, token_budget: int = 12_000,
        seed_k: int = 20, max_candidates: int = 200, rrf_k: int = 60,
        alpha: float = 0.4, core_size: int = 5, precision_floor: float = 0.35,
        backfill: bool = True,
    ) -> list[dict[str, Any]]:
        """Return scored evidence under sum(chunk token counts); expose last_trace.

        No generation LLM is called. The embedding identity must match the index.
        Backfill can admit below-floor positive scores; core is also protected.
        """
        self.last_trace = None
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        settings = retrieval_settings(
            token_budget=token_budget, seed_k=seed_k, max_candidates=max_candidates,
            rrf_k=rrf_k, alpha=alpha, core_size=core_size,
            precision_floor=precision_floor, backfill=backfill,
        )
        if token_budget == 0:
            self.last_trace = {
                "query": query, "settings": settings, "status": "empty_budget",
                "token_budget": 0, "used_tokens": 0, "unused_tokens": 0,
                "seed_chunk_ids": [], "candidate_chunk_ids": [], "selected_chunk_ids": [],
                "candidate_scores": [], "selection_order": [], "snapshot_generation": None,
                "active_channels": [], "channel_rankings": {}, "candidate_cap_reached": False,
            }
            return []
        if self.embed_fn is None or not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ValueError("Retrieval requires embed_fn and an explicit embedding_model identity.")
        index = self._index if self._index is not None else self.load()
        expected = index["manifest"]["embedding"]
        if self.embedding_model != expected["model"] or (
            self.embedding_dimension is not None and self.embedding_dimension != expected["dimension"]
        ) or (self.document_prefix is not None and self.document_prefix != expected["document_prefix"]):
            raise ValueError("Retrieval embedding settings do not match the loaded index.")
        vector = embed_texts(
            [query], self.embed_fn, model_id=self.embedding_model,
            dimension=expected["dimension"], prefix=self.query_prefix, cache_dir=self.cache_dir,
        )[0]
        initial = initial_retrieval(index, query, vector, seed_k=seed_k, rrf_k=rrf_k)
        ids = [chunk["chunk_id"] for chunk in index["chunks"]]
        relevance = dict(zip(ids, (float(value) for value in initial["relevance"])))
        seeds = [ids[row] for row in initial["seed_rows"]]
        candidates, subgraph, hops = expand_candidates(
            index["graph"], seeds, relevance, max_candidates=max_candidates,
        )
        coordinates = dict(zip(ids, index["spectral_coords"]))
        scores, weights = score_candidates(
            subgraph, seeds, candidates, relevance, coordinates, self._spectral_support, alpha=alpha,
        )
        selection = select_context(index["chunks"], scores, settings)
        self.last_trace = retrieval_trace(
            query, index, initial, candidates, hops, scores, weights, selection, settings,
        )
        return evidence_records(index["chunks"], scores, selection)
