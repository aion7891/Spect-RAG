"""Exact vs FAISS vs HNSW for SpectraFuse's two vector-search steps.

  1. Ingestion: the semantic k-nearest-neighbour graph (paper Eq. 3), k = 10.
  2. Query time: the dense top-1000 search used for seeding.

Run:  python benchmarks/ann_benchmark.py          (writes benchmarks/results/ann.json and ann.md)

Vectors are synthetic but clustered (200 topics, 768 dimensions like nomic-embed-text),
generated with a fixed seed, so every run uses the same data.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np

from spectrafuse import make_settings
from spectrafuse.utils.edges_content import ann_knn_edges, knn_edges
from spectrafuse.utils.vector_index import build_vector_index, installed, search_vectors

SIZES = [10_000, 50_000, 100_000]
DIMS, TOPICS, K, QUERIES, TOP_N = 768, 200, 10, 200, 1000
OUT = Path(__file__).parent / "results"


def clustered_vectors(count, seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(TOPICS, DIMS)).astype(np.float32)
    vectors = centres[rng.integers(0, TOPICS, count)] + 0.6 * rng.normal(size=(count, DIMS)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def edge_set(edges):
    return {(i, j) for i, j, _ in edges}


def run_size(count):
    vectors, queries = clustered_vectors(count, 0), clustered_vectors(QUERIES, 1)
    rows, exact_edges, exact_top = [], None, None
    for backend in ["exact", "faiss", "hnsw"]:
        if backend != "exact" and not installed(backend):
            continue
        start = time.perf_counter()
        index = build_vector_index(vectors, make_settings({"ann_backend": backend}))
        edges = knn_edges(vectors, K) if backend == "exact" else ann_knn_edges(index, K)
        build = time.perf_counter() - start
        start = time.perf_counter()
        _, top = search_vectors(index, queries, TOP_N)
        query_ms = 1000 * (time.perf_counter() - start) / QUERIES
        if backend == "exact":
            exact_edges, exact_top = edge_set(edges), top
        edge_recall = len(edge_set(edges) & exact_edges) / len(exact_edges)
        top_recall = np.mean([len(set(a) & set(b)) / TOP_N for a, b in zip(top, exact_top)])
        rows.append({"chunks": count, "backend": backend, "knn_graph_seconds": round(build, 2),
                     "edge_recall": round(float(edge_recall), 4), "query_ms": round(query_ms, 3),
                     "top1000_recall": round(float(top_recall), 4)})
        print(rows[-1], flush=True)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    rows = [row for count in SIZES for row in run_size(count)]
    machine = "%s, %s, Python %s" % (platform.platform(), platform.processor() or "cpu",
                                     platform.python_version())
    (OUT / "ann.json").write_text(json.dumps({"machine": machine, "rows": rows}, indent=1))
    lines = ["| Chunks | Backend | kNN graph build (s) | Edge recall vs exact | Query top-1000 (ms) | Top-1000 recall |",
             "|---|---|---|---|---|---|"]
    lines += ["| {chunks:,} | {backend} | {knn_graph_seconds} | {edge_recall:.3f} | {query_ms} | "
              "{top1000_recall:.3f} |".format(**row) for row in rows]
    (OUT / "ann.md").write_text("Machine: %s\n\n%s\n" % (machine, "\n".join(lines)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
