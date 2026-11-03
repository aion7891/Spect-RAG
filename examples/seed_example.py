"""Phase 7 demo: hybrid seed retrieval (dense + BM25 + utility questions, fused with RRF).

Run:  python examples/seed_example.py                     (offline hash embedder)
      python examples/seed_example.py ollama              (nomic-embed-text)
      python examples/seed_example.py ollama gemma3:4b    (+ LLM utility questions)
"""

import sys
from pathlib import Path

from spectrag import SpectRAGIngestion, make_settings
from spectrag.utils.bm25 import build_bm25
from spectrag.utils.embedder import make_embedder
from spectrag.utils.questions import make_question_fn, ollama_llm
from spectrag.utils.seeding import find_seeds
from spectrag.utils.store import load_index

HERE = Path(__file__).parent
QUESTIONS = ["Which company did the co-founder of Aurora Labs join after leaving?",
             "How much water does the cooperative save?"]


def label(index, chunk_id):
    chunk = index["chunks"][chunk_id]
    return "%d:%s#%d" % (chunk_id, chunk["doc_id"].split(".")[0], chunk["position"])


def main():
    embedder = sys.argv[1] if len(sys.argv) > 1 else "hash"
    question_fn = make_question_fn(ollama_llm(sys.argv[2])) if len(sys.argv) > 2 else None
    SpectRAGIngestion(HERE / "demo_index", {"embedder": embedder},
                      question_fn=question_fn).ingest(HERE / "docs")

    index = load_index(HERE / "demo_index")
    settings = make_settings(index["settings"])   # the same settings the index was built with
    settings["seed_count"] = 4                     # the paper uses 20; 4 is easier to read here
    embed = make_embedder(settings)
    bm25 = build_bm25([chunk["text"] for chunk in index["chunks"]])
    for question in QUESTIONS:
        result = find_seeds(question, embed([question], kind="query")[0], index, bm25, settings)
        print("\nQ:", question)
        for name, ranked in zip(["dense", "bm25", "utility"], result["rankings"]):
            print("  %-8s" % name, ", ".join(label(index, c) for c in ranked[:4]))
        print("  SEEDS   ", ", ".join(label(index, c) for c in result["seeds"]))
        best = sorted(range(len(index["chunks"])), key=lambda c: -result["relevance"][c])[:3]
        print("  relevance top 3:", ", ".join("%s=%.2f" % (label(index, c), result["relevance"][c])
                                              for c in best))


if __name__ == "__main__":
    main()
