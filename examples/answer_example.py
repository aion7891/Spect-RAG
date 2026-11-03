"""Phase 10 demo: ask a question and get an answer with sources (needs Ollama for the LLM).

Run:  python examples/answer_example.py                       (nomic-embed-text + gemma3:4b)
      python examples/answer_example.py ollama qwen3:4b       (nomic-embed-text + qwen3:4b)
      python examples/answer_example.py hash gemma3:4b        (offline hash embedder)
"""

import sys
from pathlib import Path

from spectrag import SpectRAGIngestion, SpectRAGRetrieval
from spectrag.utils.llm import ollama_llm

HERE = Path(__file__).parent
QUESTION = "Which company did the co-founder of Aurora Labs join, and what does it build?"


def main():
    embedder = sys.argv[1] if len(sys.argv) > 1 else "ollama"  # Ollama is needed for the LLM anyway
    model = sys.argv[2] if len(sys.argv) > 2 else "gemma3:4b"
    SpectRAGIngestion(HERE / "demo_index", {"embedder": embedder}).ingest(HERE / "docs")
    search = SpectRAGRetrieval(HERE / "demo_index")

    result = search.answer(QUESTION, ollama_llm(model), token_budget=400)
    print("Q:", QUESTION)
    print("\nA:", result["answer"])
    print("\nSources (%d tokens of context):" % result["tokens"])
    for chunk in result["chunks"]:
        print("  %d. %s#%d  score=%.2f" % (chunk["rank"], chunk["doc_id"], chunk["position"],
                                          chunk["score"]))


if __name__ == "__main__":
    main()
