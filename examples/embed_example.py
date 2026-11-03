"""Phase 4 demo: chunk, embed, save, load, and do a simple dense search.

Run:  python examples/embed_example.py           (offline hash embedder)
      python examples/embed_example.py ollama    (real nomic-embed-text via Ollama)
"""

import sys
from pathlib import Path

import numpy as np

from spectrag import make_settings
from spectrag.utils.chunker import chunk_documents
from spectrag.utils.embedder import make_embedder
from spectrag.utils.loader import load_docs
from spectrag.utils.store import load_index, save_index

HERE = Path(__file__).parent
QUESTION = "Who is the chief technology officer of Helios Energy?"


def main():
    settings = make_settings({"embedder": sys.argv[1] if len(sys.argv) > 1 else "hash"})
    embed = make_embedder(settings)
    chunks = chunk_documents(load_docs(HERE / "docs"), settings, embed_fn=embed)
    vectors = embed([chunk["text"] for chunk in chunks], kind="document")
    folder = save_index(HERE / "demo_index", {"chunks": chunks, "embeddings": vectors,
                                              "settings": settings})
    print("Saved", len(chunks), "chunks, embeddings", vectors.shape, "to", folder.name)

    index = load_index(folder)  # read everything back from disk
    query = embed([QUESTION], kind="query")[0]
    scores = index["embeddings"] @ query  # cosine similarity (vectors are normalised)
    print("\nQuestion:", QUESTION)
    for rank, i in enumerate(np.argsort(-scores)[:3], start=1):
        chunk = index["chunks"][i]
        print("  %d. score %.3f  [%d] %s  %s" % (rank, scores[i], chunk["id"], chunk["doc_id"],
                                                 chunk["text"].splitlines()[0][:50]))


if __name__ == "__main__":
    main()
