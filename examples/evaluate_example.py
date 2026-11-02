"""Phase 10 demo: SpectRAG vs dense top-k vs hybrid top-k on 12 questions (9 multi-hop).

Run:  python examples/evaluate_example.py                       (offline hash embedder)
      python examples/evaluate_example.py ollama                (nomic-embed-text)
      python examples/evaluate_example.py ollama gemma3:4b      (+ answers, EM / F1 / contains)
"""

import sys
from pathlib import Path

from spectrag import SpectRAGIngestion, SpectRAGRetrieval
from spectrag.utils.evaluation import evaluate_answers, evaluate_retrieval, load_questions
from spectrag.utils.llm import ollama_llm

HERE = Path(__file__).parent / "eval"
BUDGET = 400          # tokens of context per question (the paper uses ~12,000 on big corpora)
SEEDS = 5             # the paper uses 20 seeds; this corpus has only ~27 chunks


def print_summary(summary, keys):
    print("method     " + "".join("%-10s" % key for key in keys))
    for method, values in summary.items():
        print("%-10s " % method + "".join("%-10.2f" % values[key] for key in keys))


def main():
    embedder = sys.argv[1] if len(sys.argv) > 1 else "hash"
    ingest = SpectRAGIngestion(HERE / "index", {"embedder": embedder, "chunk_tokens": 128,
                                                "overlap_tokens": 32})
    info = ingest.ingest(HERE / "docs")
    print("Corpus: %d documents, %d chunks, %d graph edges" % (info["documents"], info["chunks"],
                                                              info["graph_edges"]))
    questions = load_questions(HERE / "questions.json")
    search = SpectRAGRetrieval(HERE / "index", {"seed_count": SEEDS})

    result = evaluate_retrieval(search, questions, BUDGET)
    print("\nRetrieval, %d questions, %d-token budget for every method:" % (len(questions), BUDGET))
    print_summary(result["summary"], ["recall", "complete", "precision", "f1", "mrr"])
    print("\nAll evidence found? (1 = yes)            dense hybrid spectrag")
    for row in result["rows"]:
        print("  %-40s %d     %d      %d" % (row["question"][:40], row["dense"]["complete"],
                                           row["hybrid"]["complete"], row["spectrag"]["complete"]))

    if len(sys.argv) > 2:
        answers = evaluate_answers(search, questions, BUDGET, ollama_llm(sys.argv[2]))
        print("\nAnswers by %s:" % sys.argv[2])
        print_summary(answers["summary"], ["em", "f1", "contains"])
        for row in answers["rows"]:
            print("  Q:", row["question"][:60])
            for method in ["dense", "hybrid", "spectrag"]:
                print("     %-9s %s" % (method, row[method]["answer"][:70].replace("\n", " ")))


if __name__ == "__main__":
    main()
