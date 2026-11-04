"""Phase 3 demo: split the sample documents into chunks.

Run:  python examples/chunk_example.py            (paper default: 1200 tokens)
      python examples/chunk_example.py 60 20      (small chunks to see packing and overlap)
"""

import sys
from pathlib import Path

from spectrafuse import make_settings
from spectrafuse.utils.chunker import chunk_documents
from spectrafuse.utils.loader import load_docs

DOCS_FOLDER = Path(__file__).parent / "docs"


def main():
    sizes = [int(arg) for arg in sys.argv[1:3]]
    overrides = dict(zip(["chunk_tokens", "overlap_tokens"], sizes))
    settings = make_settings(overrides)
    chunks = chunk_documents(load_docs(DOCS_FOLDER), settings)
    print("chunk_tokens =", settings["chunk_tokens"], " overlap_tokens =", settings["overlap_tokens"])
    print(len(chunks), "chunks\n")
    for chunk in chunks:
        section = " > ".join(chunk["section"]) or "(no headings)"
        first_line = chunk["text"].splitlines()[0][:60]
        print("[%d] %s #%d  %s  %d tokens" % (chunk["id"], chunk["doc_id"], chunk["position"],
                                              chunk["mode"], chunk["tokens"]))
        print("    section:", section)
        print("    starts :", first_line)


if __name__ == "__main__":
    main()
