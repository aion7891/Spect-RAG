"""Phase 5 demo: build the six edge signals for the sample documents.

Run:  python examples/edges_example.py                      (offline: hash embedder, no questions)
      python examples/edges_example.py ollama gemma3:4b     (nomic-embed-text + LLM questions)
"""

import sys
from pathlib import Path

from spectrag import make_settings
from spectrag.utils.chunker import chunk_documents
from spectrag.utils.edges import SIGNALS, build_edge_signals, count_edges
from spectrag.utils.embedder import make_embedder
from spectrag.utils.entities import add_entities
from spectrag.utils.loader import load_docs
from spectrag.utils.questions import add_utility_questions, make_question_fn, ollama_llm
from spectrag.utils.references import add_references

DOCS_FOLDER = Path(__file__).parent / "docs"


def label(chunk):
    """Short name for a chunk, e.g. 'aurora_labs.md#2'."""
    return "%s#%d" % (chunk["doc_id"], chunk["position"])


def main():
    settings = make_settings({"embedder": sys.argv[1] if len(sys.argv) > 1 else "hash",
                              "semantic_k": 3})
    question_fn = make_question_fn(ollama_llm(sys.argv[2])) if len(sys.argv) > 2 else None
    embed = make_embedder(settings)
    chunks = chunk_documents(load_docs(DOCS_FOLDER), settings, embed)
    add_entities(chunks)
    add_references(chunks)
    add_utility_questions(chunks, question_fn)
    embeddings = embed([chunk["text"] for chunk in chunks])
    signals, _, _ = build_edge_signals(chunks, embeddings, embed, settings)

    for chunk in chunks:
        print("[%d] %-26s refs=%s" % (chunk["id"], label(chunk), [r["target"] for r in chunk["references"]]))
        print("     entities:", ", ".join(chunk["entities"][:8]))
        for question in chunk["utility_questions"]:
            print("     question:", question)
    print("\nEdges per signal:", count_edges(signals))
    for name in SIGNALS:
        print("\n" + name)
        for i, j, weight in signals[name]:
            print("  %.3f  %s -- %s" % (weight, label(chunks[i]), label(chunks[j])))


if __name__ == "__main__":
    main()
