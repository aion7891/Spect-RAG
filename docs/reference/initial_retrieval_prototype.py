from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Callable, Any
import numpy as np
import networkx as nx


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    embedding: np.ndarray
    utility_questions: List[np.ndarray] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpectRAG:
    """
    Implementation of Algorithm 1: Online SpectRAG retrieval and generation
    (arXiv:2610.36991v2).
    """

    def __init__(
        self,
        chunks: Dict[str, Chunk],
        graph: nx.Graph,
        spectral_coords: Dict[str, np.ndarray],
        embed_fn: Callable[[str], np.ndarray],
        bm25_fn: Callable[[str, List[str]], Dict[str, float]],
        llm_fn: Callable[[str], str],
        seed_k: int = 20,
        max_candidates: int = 200,
        alpha: float = 0.4,          # Blend weight between initial relevance r(c) and proximity p(c)
        beta_ppr: float = 0.5,       # Blend weight between Spectral Proximity and PPR inside p(c)
        core_size: int = 5,          # Protected core size
        precision_floor: float = 0.35,
        coverage_bonus: float = 0.08,
        rrf_k: int = 60,
    ):
        """
        Args:
            chunks: Mapping of chunk_id -> Chunk object (V).
            graph: Weighted chunk graph G = (V, E, W) where edges hold 'weight'.
            spectral_coords: Precomputed d-dim Laplacian spectral coordinates Z (chunk_id -> np.ndarray).
            embed_fn: Function mapping text -> L2-normalized dense vector.
            bm25_fn: Function mapping (query, chunk_ids) -> {chunk_id: bm25_score}.
            llm_fn: Function mapping grounded prompt -> generated answer string.
        """
        self.chunks = chunks
        self.graph = graph
        self.Z = spectral_coords
        self.embed_fn = embed_fn
        self.bm25_fn = bm25_fn
        self.llm_fn = llm_fn

        self.seed_k = seed_k
        self.max_candidates = max_candidates
        self.alpha = alpha
        self.beta_ppr = beta_ppr
        self.core_size = core_size
        self.precision_floor = precision_floor
        self.coverage_bonus = coverage_bonus
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Helper: Min-Max Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = np.array(list(scores.values()), dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if np.isclose(vmin, vmax):
            return {k: 1.0 for k in scores}
        return {k: float((v - vmin) / (vmax - vmin)) for k, v in scores.items()}

    # ------------------------------------------------------------------
    # Steps 1-3: Dense, BM25, Utility-Question Scoring & RRF Seed Fusion
    # ------------------------------------------------------------------
    def _initial_retrieval(
        self, query: str
    ) -> Tuple[List[str], Dict[str, float]]:
        q_emb = self.embed_fn(query)
        all_ids = list(self.chunks.keys())

        # Step 1: Compute dense FAISS / cosine scores
        dense_scores = {
            cid: float(np.dot(q_emb, self.chunks[cid].embedding))
            for cid in all_ids
        }

        # Step 2a: Compute BM25 lexical scores
        bm25_scores = self.bm25_fn(query, all_ids)

        # Step 2b: Compute utility-question matching scores (max cosine over stored questions)
        uq_scores = {}
        for cid in all_ids:
            uqs = self.chunks[cid].utility_questions
            uq_scores[cid] = (
                float(max(np.dot(q_emb, uq_emb) for uq_emb in uqs))
                if uqs
                else dense_scores[cid]
            )

        # Step 3: Fuse rankings with Reciprocal Rank Fusion (RRF) to form seed set S0
        def get_ranks(score_dict: Dict[str, float]) -> Dict[str, int]:
            sorted_ids = sorted(score_dict, key=score_dict.get, reverse=True)
            return {cid: rank + 1 for rank, cid in enumerate(sorted_ids)}

        r_dense = get_ranks(dense_scores)
        r_bm25 = get_ranks(bm25_scores)
        r_uq = get_ranks(uq_scores)

        rrf_scores = {
            cid: (1.0 / (self.rrf_k + r_dense[cid]))
            + (1.0 / (self.rrf_k + r_bm25[cid]))
            + (1.0 / (self.rrf_k + r_uq[cid]))
            for cid in all_ids
        }

        seed_set_S0 = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[
            : self.seed_k
        ]

        # Hybrid relevance vector r(c) consumed downstream (normalized dense + BM25)
        norm_dense = self._normalize(dense_scores)
        norm_bm25 = self._normalize(bm25_scores)
        hybrid_r = {
            cid: 0.5 * norm_dense[cid] + 0.5 * norm_bm25[cid]
            for cid in all_ids
        }

        return seed_set_S0, hybrid_r

    # ------------------------------------------------------------------
    # Steps 4-5: Two-Hop BFS Subgraph Expansion (Neo4j equivalent)
    # ------------------------------------------------------------------
    def _expand_Subgraph(
        self, seed_set_S0: List[str], r_scores: Dict[str, float]
    ) -> Tuple[List[str], nx.Graph]:
        visited: Set[str] = set(seed_set_S0)
        frontier: Set[str] = set(seed_set_S0)

        # Step 4: Two-hop BFS expansion from S0
        for _ in range(2):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node in self.graph:
                    for nbr in self.graph.neighbors(node):
                        if nbr not in visited:
                            next_frontier.add(nbr)
            visited.update(next_frontier)
            frontier = next_frontier

        # Step 5: Cap expanded candidate set S1 at max_candidates (200 chunks)
        # Keep all seeds S0, prioritize highest initial relevance/edge weight for remainder
        non_seeds = [c for c in visited if c not in seed_set_S0]
        non_seeds.sort(key=lambda c: r_scores.get(c, 0.0), reverse=True)
        S1 = (seed_set_S0 + non_seeds)[: self.max_candidates]

        subgraph = self.graph.subgraph(S1).copy()
        return S1, subgraph

    # ------------------------------------------------------------------
    # Steps 6-7: Spectral Proximity, Personalized PageRank & Re-scoring
    # ------------------------------------------------------------------
    def _compute_proximity_and_rescore(
        self,
        S0: List[str],
        S1: List[str],
        subgraph: nx.Graph,
        r_scores: Dict[str, float],
    ) -> Dict[str, float]:
        # Personalization distribution over seeds weighted by r(c)
        seed_weights = {s: max(r_scores.get(s, 1e-4), 1e-4) for s in S0}
        total_sw = sum(seed_weights.values())
        personalization = {
            c: (seed_weights[c] / total_sw if c in seed_weights else 0.0)
            for c in S1
        }

        # Step 6a: Personalized PageRank on the weighted candidate subgraph
        if subgraph.number_of_edges() > 0:
            ppr_raw = nx.pagerank(
                subgraph,
                alpha=0.85,
                personalization=personalization,
                weight="weight",
            )
        else:
            ppr_raw = personalization
        ppr_norm = self._normalize(ppr_raw)

        # Step 6b: Spectral proximity from Laplacian coordinates Z
        # Compute seed-weighted proximity in spectral eigenspace (RBF/cosine kernel on Z)
        spec_raw = {}
        for c in S1:
            z_c = self.Z[c]
            prox = 0.0
            for s in S0:
                z_s = self.Z[s]
                dist_sq = float(np.sum((z_c - z_s) ** 2))
                prox += personalization[s] * np.exp(-dist_sq)
            spec_raw[c] = prox
        spec_norm = self._normalize(spec_raw)

        # Combine spectral proximity and PPR into hybrid proximity p(c)
        p_scores = {
            c: (1.0 - self.beta_ppr) * spec_norm[c] + self.beta_ppr * ppr_norm[c]
            for c in S1
        }

        # Step 7: Re-score candidates: s2(c) <- (1 - alpha) * r(c) + alpha * p(c)
        r_sub_norm = self._normalize({c: r_scores[c] for c in S1})
        s2_scores = {
            c: (1.0 - self.alpha) * r_sub_norm[c] + self.alpha * p_scores[c]
            for c in S1
        }
        return s2_scores

    # ------------------------------------------------------------------
    # Steps 8-12: Token-Budgeted Context Selection
    # ------------------------------------------------------------------
    def _select_context(
        self, S1: List[str], s2_scores: Dict[str, float], token_budget: int
    ) -> List[str]:
        ranked = sorted(S1, key=s2_scores.get, reverse=True)
        selected_C: List[str] = []
        covered_docs: Set[str] = set()
        current_tokens = 0

        # Step 8: Initialize protected core C from highest-scoring candidates
        for cid in ranked[: self.core_size]:
            t_count = self.chunks[cid].token_count
            if current_tokens + t_count <= token_budget:
                selected_C.append(cid)
                covered_docs.add(self.chunks[cid].doc_id)
                current_tokens += t_count

        remaining = [c for c in ranked if c not in selected_C]

        # Steps 9-11: Greedily add candidates satisfying precision floor + coverage tie-breaks
        while remaining and current_tokens < token_budget:
            eligible = [
                c for c in remaining
                if s2_scores[c] >= self.precision_floor
                and current_tokens + self.chunks[c].token_count <= token_budget
            ]
            if not eligible:
                break

            # Score with document/cluster coverage tie-break bonus for uncovered source docs
            def selection_objective(cid: str) -> float:
                doc_is_new = 1.0 if self.chunks[cid].doc_id not in covered_docs else 0.0
                return s2_scores[cid] + self.coverage_bonus * doc_is_new

            best_c = max(eligible, key=selection_objective)
            selected_C.append(best_c)
            covered_docs.add(self.chunks[best_c].doc_id)
            current_tokens += self.chunks[best_c].token_count
            remaining.remove(best_c)

        # Step 12: Backfill with highest-scoring remaining chunks if budget remains
        for cid in sorted(remaining, key=s2_scores.get, reverse=True):
            t_count = self.chunks[cid].token_count
            if current_tokens + t_count <= token_budget:
                selected_C.append(cid)
                current_tokens += t_count

        return selected_C

    # ------------------------------------------------------------------
    # Step 13-15: Document Grouping, Prompting & Full Pipeline Execution
    # ------------------------------------------------------------------
    def retrieve_and_generate(
        self, query: str, token_budget: int = 12000
    ) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        # Steps 1-3
        S0, r_scores = self._initial_retrieval(query)

        # Steps 4-5
        S1, subgraph = self._expand_Subgraph(S0, r_scores)

        # Steps 6-7
        s2_scores = self._compute_proximity_and_rescore(S0, S1, subgraph, r_scores)

        # Steps 8-12
        context_C = self._select_context(S1, s2_scores, token_budget)

        # Step 13: Format C grouped by source document and generate answer
        grouped_docs: Dict[str, List[Chunk]] = {}
        for cid in context_C:
            chunk = self.chunks[cid]
            grouped_docs.setdefault(chunk.doc_id, []).append(chunk)

        formatted_blocks = []
        for doc_id, doc_chunks in grouped_docs.items():
            joined_text = "\n".join(f"  [{c.chunk_id}] {c.text}" for c in doc_chunks)
            formatted_blocks.append(f"=== Source Document: {doc_id} ===\n{joined_text}")

        context_str = "\n\n".join(formatted_blocks)
        grounded_prompt = (
            "Answer the query strictly using the grouped source context below. "
            "Cite chunk IDs in brackets.\n\n"
            f"{context_str}\n\nQuery: {query}\nAnswer:"
        )
        answer = self.llm_fn(grounded_prompt)

        # Step 15: Return Answer, ranked source metadata, and final context set C
        ranked_metadata = [
            {
                "chunk_id": cid,
                "doc_id": self.chunks[cid].doc_id,
                "s2_score": round(s2_scores[cid], 4),
                "token_count": self.chunks[cid].token_count,
                **self.chunks[cid].metadata,
            }
            for cid in sorted(context_C, key=s2_scores.get, reverse=True)
        ]

        return answer, ranked_metadata, context_C