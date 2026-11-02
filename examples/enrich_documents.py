"""Phase 3 offline smoke example; requires explicit opt-in to test callbacks."""

import argparse
import json
from pathlib import Path

from spect_rag import SpectRAGIngestion

from demo_callbacks import CALLS, demo_embed, demo_extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--demo", action="store_true", help="Use fake callbacks, not real models.")
    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/phase3/cache"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.demo:
        parser.error("Pass --demo for this wiring example; configure real callbacks in your application.")
    print("DEMO ONLY: deterministic test callbacks, not a semantic model or LLM.")
    ingestion = SpectRAGIngestion(
        embed_fn=demo_embed, embedding_model="demo-word-hash-v1",
        embedding_dimension=16, extract_fn=demo_extract,
        extraction_model="demo-keywords-v1", cache_dir=args.cache_dir,
    )
    chunks = ingestion.enrich(args.documents)
    for chunk in chunks:
        print(f"{chunk['chunk_id']} | {chunk['chunking_mode']} | "
              f"embedding={chunk['embedding'].shape} | concepts={chunk['concepts']}")
        print(f"  questions={chunk['utility_questions']}")
        print(f"  resolved citations={sum(c['status'] == 'resolved' for c in chunk['citations'])}")
    print(f"Callback calls this run: {CALLS}")
    print("No graph or retrieval index was built.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False, default=lambda value: value.tolist()),
            encoding="utf-8",
        )
        print(f"Saved inspectable outputs to {args.output}")


if __name__ == "__main__":
    main()
