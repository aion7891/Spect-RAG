"""SpectRAGRetrieval: the online pipeline (paper Table 2, Algorithm 1). Run it per question.

    search = SpectRAGRetrieval("my_index")
    result = search.search("Which company did the co-founder of Aurora Labs join?")
    print(result["context"])
"""

from spectrag.config import make_settings
from spectrag.utils.bm25 import build_bm25
from spectrag.utils.diffusion import score_candidates
from spectrag.utils.embedder import make_embedder
from spectrag.utils.graph import neighbors
from spectrag.utils.prompt import build_prompt, format_context
from spectrag.utils.seeding import find_seeds
from spectrag.utils.selection import select_context
from spectrag.utils.store import load_index


class SpectRAGRetrieval:
    """Seeds -> graph expansion -> proximity re-scoring -> token-budget selection -> context."""

    def __init__(self, index_dir, settings=None, embed_fn=None):
        """Load an index made by SpectRAGIngestion.

        settings: retrieval settings to change, e.g. {"token_budget": 4000, "seed_count": 10}.
        embed_fn: pass the same function again if the index was built with your own embedder.
        """
        self.index = load_index(index_dir)
        self.settings = make_settings(dict(self.index["settings"], **(settings or {})))
        if embed_fn is not None:
            self.settings["embedder"] = embed_fn
        elif str(self.settings["embedder"]).startswith("<function"):
            raise ValueError("This index was built with your own embed function. "
                             "Pass it again: SpectRAGRetrieval(index_dir, embed_fn=...)")
        self.embed = make_embedder(self.settings)
        self.bm25 = build_bm25([chunk["text"] for chunk in self.index["chunks"]])
        self.table = neighbors(self.index["edges"], len(self.index["chunks"]))

    def retrieve(self, question, settings):
        """Stages 1-4. Returns (seed result, scored candidates, picks)."""
        query_vector = self.embed([question], kind="query")[0]
        seeds = find_seeds(question, query_vector, self.index, self.bm25, settings)
        scored = score_candidates(seeds["seeds"], seeds["relevance"], self.index,
                                  self.table, settings)
        return seeds, scored, select_context(scored, self.index, settings)

    def describe(self, k, reason, rank, scored):
        """One selected chunk with all its scores (the 'ranked source metadata')."""
        chunk = self.index["chunks"][scored["candidates"][k]]
        info = {key: chunk[key] for key in ["id", "doc_id", "position", "section", "tokens", "text"]}
        info.update({"rank": rank, "reason": reason, "hop": scored["hop"][k],
                     "score": float(scored["scores"][k]),
                     "relevance": float(scored["relevance"][k]),
                     "proximity": float(scored["proximity"][k])})
        return info

    def search(self, question, token_budget=None):
        """Find the context for a question. Returns a dict with 'context', 'chunks', ..."""
        settings = dict(self.settings)
        if token_budget is not None:
            settings["token_budget"] = token_budget
        seeds, scored, picks = self.retrieve(question, settings)
        chunks = [self.describe(k, reason, rank, scored)
                  for rank, (k, reason) in enumerate(picks, start=1)]
        context = format_context(chunks)
        return {"question": question, "context": context, "chunks": chunks,
                "prompt": build_prompt(question, context),
                "tokens": sum(chunk["tokens"] for chunk in chunks),
                "seeds": seeds["seeds"], "candidates": scored["candidates"]}
