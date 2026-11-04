"""Phase 8 demo: graph expansion + proximity re-scoring (PageRank and spectral closeness).

Run:  python examples/expand_example.py            (offline hash embedder)
      python examples/expand_example.py ollama     (nomic-embed-text)
"""

import sys
from pathlib import Path

from spectrafuse import SpectRAGIngestion, make_settings
from spectrafuse.utils.bm25 import build_bm25
from spectrafuse.utils.diffusion import score_candidates
from spectrafuse.utils.embedder import make_embedder
from spectrafuse.utils.graph import neighbors
from spectrafuse.utils.seeding import find_seeds
from spectrafuse.utils.store import load_index

HERE = Path(__file__).parent
QUESTION = "Which company did the co-founder of Aurora Labs join after leaving?"


def main():
    embedder = sys.argv[1] if len(sys.argv) > 1 else "hash"
    SpectRAGIngestion(HERE / "demo_index", {"embedder": embedder}).ingest(HERE / "docs")
    index = load_index(HERE / "demo_index")
    settings = make_settings(dict(index["settings"], seed_count=2))  # paper: 20; 2 shows the effect
    embed = make_embedder(settings)
    bm25 = build_bm25([chunk["text"] for chunk in index["chunks"]])
    table = neighbors(index["edges"], len(index["chunks"]))

    seeds = find_seeds(QUESTION, embed([QUESTION], kind="query")[0], index, bm25, settings)
    result = score_candidates(seeds["seeds"], seeds["relevance"], index, table, settings)
    print("Q:", QUESTION)
    print("Seeds S0:", seeds["seeds"], "  candidates S1:", len(result["candidates"]))
    print("\nchunk  hop  relevance  pagerank  spectral  proximity  score   document")
    for k, chunk_id in enumerate(result["candidates"]):
        chunk = index["chunks"][chunk_id]
        print("[%d]    %d     %.2f       %.3f     %.2f      %.2f      %.2f    %s#%d" % (
            chunk_id, result["hop"][k], result["relevance"][k], result["ppr"][k],
            result["spectral"][k], result["proximity"][k], result["scores"][k],
            chunk["doc_id"], chunk["position"]))


if __name__ == "__main__":
    main()
