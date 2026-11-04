"""Phase 9 demo: the full SpectraFuse pipeline with the two main classes.

Run:  python examples/search_example.py                (offline hash embedder, 250-token budget)
      python examples/search_example.py ollama 400     (nomic-embed-text, 400-token budget)
"""

import sys
from pathlib import Path

from spectrafuse import SpectRAGIngestion, SpectRAGRetrieval

HERE = Path(__file__).parent
QUESTIONS = ["Which company did the co-founder of Aurora Labs join after leaving?",
             "How much water does the cooperative save?"]


def main():
    embedder = sys.argv[1] if len(sys.argv) > 1 else "hash"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 250   # paper: about 12,000 tokens

    # Offline: once per document set.
    SpectRAGIngestion(HERE / "demo_index", {"embedder": embedder}).ingest(HERE / "docs")

    # Online: per question.
    search = SpectRAGRetrieval(HERE / "demo_index")
    for question in QUESTIONS:
        result = search.search(question, token_budget=budget)
        print("=" * 70)
        print("Q:", question)
        print("Used %d of %d tokens, %d chunks:" % (result["tokens"], budget, len(result["chunks"])))
        for chunk in result["chunks"]:
            print("  %d. [%d] %-24s %-8s hop=%d score=%.2f tokens=%d" % (
                chunk["rank"], chunk["id"], chunk["doc_id"], chunk["reason"], chunk["hop"],
                chunk["score"], chunk["tokens"]))
        print("\n--- context for the LLM ---\n" + result["context"])


if __name__ == "__main__":
    main()
