"""Phase 2: inspect chunks without embedding models or an index."""

import argparse
import json
from pathlib import Path

from spect_rag import SpectRAGIngestion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1_200)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--output", type=Path, help="Optional JSON file containing all chunks.")
    args = parser.parse_args()
    chunks = SpectRAGIngestion().prepare(
        args.documents, chunk_size=args.chunk_size, chunk_overlap=args.overlap,
        min_chunk_tokens=min(32, args.chunk_size),
    )
    for chunk in chunks:
        section = " > ".join(chunk["section_path"]) or "(no section)"
        print(f"{chunk['chunk_id']} | {chunk['token_count']} tokens | "
              f"pages={chunk['pages']} | {section}")
        print(chunk["text"][:200].replace("\n", " "))
        print()
    print(f"Prepared {len(chunks)} chunks. No index was built.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved inspectable chunks to {args.output}")


if __name__ == "__main__":
    main()
