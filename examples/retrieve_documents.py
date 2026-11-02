"""Phase 5: retrieve from a previously built index with the same demo model."""

import argparse
import json
from pathlib import Path

from spect_rag import SpectRAGRetrieval

from demo_callbacks import demo_embed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/demo"))
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--seed-k", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()
    if not args.demo:
        parser.error("Pass --demo for the test model; configure real callbacks in your application.")
    print("DEMO ONLY: word-hash vectors, not a real semantic model. No answer LLM is called.")
    retrieval = SpectRAGRetrieval(
        args.index_dir, embed_fn=demo_embed, embedding_model="demo-word-hash-v1",
        embedding_dimension=16, cache_dir="tmp/phase5/query-cache",
    )
    evidence = retrieval.retrieve(
        args.query, token_budget=args.token_budget,
        seed_k=args.seed_k, max_candidates=args.max_candidates,
    )
    for chunk in evidence:
        print(f"[{chunk['rank']}] {chunk['chunk_id']} | score={chunk['score']:.4f} | "
              f"{chunk['token_count']} tokens | {chunk['selection_reason']}")
        print(chunk["text"])
        print()
    trace = retrieval.last_trace
    print(f"Selected {len(evidence)} chunks, using {trace['used_tokens']}/{trace['token_budget']} text tokens.")
    print("Channels:", trace["active_channels"])
    print("Seeds:", len(trace["seed_chunk_ids"]), "Candidates:", len(trace["candidate_chunk_ids"]))
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(trace, indent=2, allow_nan=False), encoding="utf-8")
        print("Saved trace:", args.trace_output)


if __name__ == "__main__":
    main()
