"""Phase 4: build and reload an index with explicitly fake model callbacks."""

import argparse
from pathlib import Path

from spect_rag import SpectRAGIngestion, SpectRAGRetrieval

from demo_callbacks import CALLS, demo_embed, demo_extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/demo"))
    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/phase4/cache"))
    parser.add_argument("--spectral-dimension", type=int, default=32)
    args = parser.parse_args()
    if not args.demo:
        parser.error("Pass --demo for test callbacks; configure real callbacks in your application.")
    print("DEMO ONLY: word-hash vectors and keyword rules, not real models.")
    ingestion = SpectRAGIngestion(
        args.index_dir, embed_fn=demo_embed, embedding_model="demo-word-hash-v1",
        embedding_dimension=16, extract_fn=demo_extract,
        extraction_model="demo-keywords-v1", cache_dir=args.cache_dir,
    )
    ingestion.ingest(args.documents, spectral_dimension=args.spectral_dimension)
    index = SpectRAGRetrieval(args.index_dir).load()
    print("Saved snapshot:", index["snapshot_dir"])
    print("Counts:", index["manifest"]["counts"])
    print("Graph:", index["graph"].graph["stats"])
    print("Positive eigenvalues:", index["eigenvalues"].tolist())
    print("Dense / question FAISS rows:", index["dense_index"].ntotal, index["question_index"].ntotal)
    print("Callback calls:", CALLS)
    print("Full ingestion and reload passed. Query retrieval arrives in Phase 5.")


if __name__ == "__main__":
    main()
