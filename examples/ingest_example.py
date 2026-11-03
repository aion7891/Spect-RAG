"""Phase 6 demo: run the whole offline pipeline with SpectRAGIngestion.

Run:  python examples/ingest_example.py                     (offline hash embedder)
      python examples/ingest_example.py ollama              (nomic-embed-text)
      python examples/ingest_example.py ollama gemma3:4b    (+ LLM utility questions)
"""

import json
import sys
from pathlib import Path

from spectrag import SpectRAGIngestion
from spectrag.utils.questions import make_question_fn, ollama_llm
from spectrag.utils.store import load_index

HERE = Path(__file__).parent


def main():
    embedder = sys.argv[1] if len(sys.argv) > 1 else "hash"
    question_fn = make_question_fn(ollama_llm(sys.argv[2])) if len(sys.argv) > 2 else None
    ingest = SpectRAGIngestion(HERE / "demo_index", {"embedder": embedder, "n_clusters": 3},
                               question_fn=question_fn)
    info = ingest.ingest(HERE / "docs")
    print(json.dumps(info, indent=1))

    index = load_index(HERE / "demo_index")
    print("\nFiles:", ", ".join(sorted(p.name for p in (HERE / "demo_index").iterdir())))
    print("Smallest non-trivial eigenvalues:", [round(float(v), 3) for v in index["eigenvalues"][:4]])
    print("\nChunk  cluster  spectral u2   section / start")
    for chunk, coords in zip(index["chunks"], index["spectral"]):
        where = " > ".join(chunk["section"][-1:]) or chunk["text"][:30]
        print("[%d]    %d       %+.3f      %s: %s" % (chunk["id"], chunk["cluster"], coords[0],
                                                 chunk["doc_id"], where))
    print("\nStrongest graph edges:")
    for i, j, weight in sorted(index["edges"], key=lambda e: -e[2])[:6]:
        print("  %.3f  [%d] -- [%d]" % (weight, i, j))


if __name__ == "__main__":
    main()
