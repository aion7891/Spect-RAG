"""Phase 2 demo: load the sample documents and count their tokens and sentences.

Run:  python examples/load_example.py
"""

from pathlib import Path

from spectrag import make_settings
from spectrag.utils.loader import load_docs
from spectrag.utils.tokens import count_tokens, get_encoder, split_sentences

DOCS_FOLDER = Path(__file__).parent / "docs"


def main():
    settings = make_settings({"chunk_tokens": 256, "overlap_tokens": 64})
    print("Chunk size setting:", settings["chunk_tokens"], "tokens")
    print("Token counter:", "tiktoken cl100k_base" if get_encoder() else "word estimate")
    print()
    for doc in load_docs(DOCS_FOLDER):
        sentences = split_sentences(doc["text"])
        print(doc["doc_id"])
        print("  tokens:", count_tokens(doc["text"]), " sentences:", len(sentences))
        print("  first sentence:", sentences[0][:70])


if __name__ == "__main__":
    main()
