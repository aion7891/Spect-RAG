"""SpectRAG quick start: ingest documents, build the index, ask questions.

Run:  python quickstart.py
No arguments are needed. Everything you may want to change is in the block below.
It works offline by default; set USE_OLLAMA = True for real embeddings and LLM answers.
"""

from pathlib import Path

import spectrag
from spectrag import SpectRAGIngestion, SpectRAGRetrieval

# ----------------------------------------------------------------------------------------
# Change these freely.
# ----------------------------------------------------------------------------------------
DOCUMENTS = Path(__file__).parent / "examples" / "docs"     # a file or a folder (.txt .md .pdf)
INDEX_DIR = Path(__file__).parent / "quickstart_index"      # where the index is saved
QUESTIONS = [
    "Which company did the co-founder of Aurora Labs join after leaving?",
    "How much water does River Valley Cooperative save?",
]
USE_OLLAMA = False      # True: nomic-embed-text embeddings + answers from LLM_MODEL (needs Ollama)
LLM_MODEL = "gemma3:4b"

INGEST_SETTINGS = {      # used once, when the index is built (see spectrag.DEFAULTS for all)
    "chunk_tokens": 1200,        # paper: 1,200-token chunks ...
    "overlap_tokens": 100,       # ... with 100 tokens of overlap
    "edge_threshold": 0.05,      # tau: weaker graph edges are dropped
    "graph_k": 10,               # each chunk keeps at most its 10 strongest edges
}
SEARCH_SETTINGS = {      # can be changed for every search
    "token_budget": 12000,       # tokens of context per question
    "seed_count": 20,            # seed chunks before graph expansion
    "alpha": 0.3,                # weight of graph proximity vs. relevance
    "selector": "spectrag",      # or "mmr", "protected_mmr", "anchor_expand_ce", "topk"
    "proximity_mode": "hybrid",  # or "ppr", "spectral", "heat", "none"
    "precision_floor": 0.4,
}
SHOW_PROGRESS = True     # print one line per pipeline step
# ----------------------------------------------------------------------------------------


def main():
    if SHOW_PROGRESS:
        spectrag.enable_logging()
    ingest_settings = dict(INGEST_SETTINGS, embedder="ollama" if USE_OLLAMA else "hash")

    # 1. Ingest and index (offline, once per document set).
    info = SpectRAGIngestion(INDEX_DIR, ingest_settings).ingest(DOCUMENTS)
    print("\nIndexed %d documents into %d chunks and %d graph edges (%s)\n"
          % (info["documents"], info["chunks"], info["graph_edges"], INDEX_DIR.name))

    # 2. Query (online, per question).
    search = SpectRAGRetrieval(INDEX_DIR, SEARCH_SETTINGS)
    llm = None
    if USE_OLLAMA:
        from spectrag.utils.llm import ollama_llm
        llm = ollama_llm(LLM_MODEL)
    for question in QUESTIONS:
        result = search.answer(question, llm) if llm else search.search(question)
        print("=" * 78 + "\nQ: " + question)
        if llm:
            print("A: " + result["answer"])
        print("Context: %d chunks, %d tokens" % (len(result["chunks"]), result["tokens"]))
        for chunk in result["chunks"][:5]:
            first_line = chunk["text"].splitlines()[0][:60]
            print("  %d. [%s] score %.2f  %s" % (chunk["rank"], chunk["doc_id"], chunk["score"], first_line))
    print("\nTry: change SEARCH_SETTINGS at the top of this file and run it again.")


if __name__ == "__main__":
    main()
