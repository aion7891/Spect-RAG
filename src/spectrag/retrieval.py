"""SpectRAGRetrieval: the online pipeline (paper Table 2, Algorithm 1). Run it per question.

    search = SpectRAGRetrieval("my_index")
    result = search.search("Which company did the co-founder of Aurora Labs join?")
    print(result["context"])
"""

import time
from pathlib import Path

from spectrag._version import INDEX_FORMAT
from spectrag.config import make_settings
from spectrag.errors import EmbedderError, IndexError_, SettingsError
from spectrag.log import logger
from spectrag.utils.bm25 import build_bm25
from spectrag.utils.diffusion import score_candidates
from spectrag.utils.embedder import make_embedder
from spectrag.utils.embedder_id import (CUSTOM, check_embedder, is_saved_function,
                                        read_saved_settings)
from spectrag.utils.graph import degrees, neighbors
from spectrag.utils.graph_build import build_graph, build_spectral
from spectrag.utils.prompt import build_prompt, format_context
from spectrag.utils.seeding import find_seeds
from spectrag.utils.selectors import select
from spectrag.utils.store import array_to_edges, check_index_folder, load_index
from spectrag.utils.vector_index import load_vector_index

# These change the graph itself, so they need rebuild_graph() instead of a per-call override.
GRAPH_SETTINGS = ("edge_weights", "edge_threshold", "graph_k", "spectral_dims", "n_clusters")


class SpectRAGRetrieval:
    """Seeds -> graph expansion -> proximity re-scoring -> token-budget selection -> context."""

    def __init__(self, index_dir, settings=None, embed_fn=None, rerank_fn=None):
        """Load an index made by SpectRAGIngestion.

        settings:  settings to change, e.g. {"token_budget": 4000, "selector": "mmr"}.
                   Graph settings (edge_threshold, graph_k, ...) rebuild the graph from the
                   saved signals, without embedding anything again.
        embed_fn:  pass the same function again if the index was built with your own embedder.
        rerank_fn: optional (question, texts) -> scores, e.g. a cross-encoder, used by the
                   "anchor_expand_ce" selector.
        """
        self.index_dir = Path(index_dir)
        check_index_folder(self.index_dir)
        self.index = load_index(index_dir, skip=("signals",))  # raw signals are loaded on demand
        made_with = self.index["info"].get("format_version", 1)
        if made_with > INDEX_FORMAT:
            raise IndexError_("This index has format %d; this SpectRAG version reads up to %d. "
                              "Upgrade spectrag." % (made_with, INDEX_FORMAT))
        self.index["edges"] = array_to_edges(self.index["edges"])
        self.index["question_owners"] = [int(owner) for owner in self.index["question_owners"]]
        built_with = make_settings(read_saved_settings(self.index["settings"]))
        self.settings = make_settings(dict(built_with, **(settings or {})))
        if embed_fn is not None:
            self.settings["embedder"] = embed_fn
        elif is_saved_function(self.settings["embedder"]):
            raise SettingsError("This index was built with your own embed function (%s). "
                                "Pass it again: SpectRAGRetrieval(index_dir, embed_fn=...)"
                                % self.settings["embedder"][len(CUSTOM):])
        self.embed = make_embedder(self.settings)
        check_embedder(self.embed, self.index.get("embedder_probe"), built_with["embedder"])
        self.rerank_fn = rerank_fn
        self.bm25 = build_bm25([chunk["text"] for chunk in self.index["chunks"]])
        self.vector_index = load_vector_index(self.index_dir, self.index["embeddings"], self.settings)
        if any(self.settings[key] != built_with[key] for key in GRAPH_SETTINGS):
            self.rebuild_graph()
        self.refresh_graph()

    def refresh_graph(self):
        """Neighbour table (CSR) and degrees for the current edges."""
        count = len(self.index["chunks"])
        self.table = neighbors(self.index["edges"], count)
        self.index["degree"] = degrees(self.index["edges"], count)

    def rebuild_graph(self, overrides=None):
        """Change graph settings without re-embedding: fuse the saved signals again (Eq. 10)
        and recompute spectral coordinates and clusters (Eq. 11-13). In memory only."""
        self.settings = make_settings(dict(self.settings, **(overrides or {})))
        saved = load_index(self.index_dir, only=("signals",))["signals"]
        signals = {name: array_to_edges(edges) for name, edges in saved.items()}
        edges, _ = build_graph(signals, self.settings)
        coords, eigenvalues, clusters = build_spectral(edges, len(self.index["chunks"]), self.settings)
        self.index.update({"edges": edges, "spectral": coords, "eigenvalues": eigenvalues})
        for chunk, cluster in zip(self.index["chunks"], clusters):
            chunk["cluster"] = int(cluster)
        self.refresh_graph()

    def settings_for(self, overrides):
        """The settings for one call: the current settings plus per-call overrides."""
        if not overrides:
            return self.settings
        changed = [key for key in GRAPH_SETTINGS
                   if key in overrides and overrides[key] != self.settings[key]]
        if changed:
            raise SettingsError("Use rebuild_graph() to change graph settings: " + ", ".join(changed))
        return make_settings(dict(self.settings, **overrides))

    def retrieve(self, question, settings=None, query_vector=None):
        """Stages 1-4. Returns (seed result, scored candidates, picks)."""
        settings = self.settings_for(settings)
        start = time.time()
        if query_vector is None:
            query_vector = self.embed([question], kind="query")[0]
        if len(query_vector) != self.index["embeddings"].shape[1]:
            raise EmbedderError("The query vector has %d dimensions but the index has %d: search "
                                "with the same embedder the index was built with."
                                % (len(query_vector), self.index["embeddings"].shape[1]))
        seeds = find_seeds(question, query_vector, self.index, self.bm25, settings,
                           self.vector_index)
        scored = score_candidates(seeds["seeds"], seeds["relevance"], self.index,
                                  self.table, settings)
        rerank = None
        if settings["selector"] == "anchor_expand_ce" and self.rerank_fn is not None:
            texts = [self.index["chunks"][c]["text"] for c in scored["candidates"]]
            rerank = self.rerank_fn(question, texts)
        picks = select(scored, self.index, settings, self.table, rerank)
        logger.debug("query %r: %d seeds, %d candidates, %d picked in %.1f ms", question[:60],
                     len(seeds["seeds"]), len(scored["candidates"]), len(picks),
                     1000 * (time.time() - start))
        return seeds, scored, picks

    def describe(self, k, reason, rank, scored):
        """One selected chunk with all its scores (the 'ranked source metadata')."""
        chunk = self.index["chunks"][scored["candidates"][k]]
        info = {key: chunk[key] for key in ["id", "doc_id", "position", "section", "tokens", "text"]}
        info.update({"rank": rank, "reason": reason, "hop": scored["hop"][k],
                     "score": float(scored["scores"][k]),
                     "relevance": float(scored["relevance"][k]),
                     "proximity": float(scored["proximity"][k])})
        return info

    def search(self, question, token_budget=None, settings=None, query_vector=None):
        """Find the context for a question. Returns a dict with 'context', 'chunks', ...

        settings: per-call overrides, e.g. {"selector": "mmr", "alpha": 0.5}.
        """
        overrides = dict(settings or {})
        if token_budget is not None:
            overrides["token_budget"] = token_budget
        seeds, scored, picks = self.retrieve(question, overrides, query_vector)
        chunks = [self.describe(k, reason, rank, scored)
                  for rank, (k, reason) in enumerate(picks, start=1)]
        context = format_context(chunks)
        return {"question": question, "context": context, "chunks": chunks,
                "prompt": build_prompt(question, context),
                "tokens": sum(chunk["tokens"] for chunk in chunks),
                "seeds": seeds["seeds"], "candidates": scored["candidates"]}

    def answer(self, question, llm_fn, token_budget=None, settings=None):
        """search() + generation (Algorithm 1 line 13). llm_fn is any function: prompt -> text.

        Returns the search result with an extra "answer" key; result["chunks"] are the sources.
        """
        result = self.search(question, token_budget, settings)
        result["answer"] = llm_fn(result["prompt"]).strip()
        return result
