"""Nearest-neighbour search over L2-normalised vectors (inner product = cosine).

Backends (settings["ann_backend"]):
  "exact" - numpy, exact (the paper uses exact FAISS IndexFlatIP, Section 6.2)
  "faiss" - FAISS HNSW graph index (pip install faiss-cpu)       approximate, fast
  "hnsw"  - hnswlib HNSW index (pip install hnswlib)              approximate, fast
  "auto"  - exact up to ANN_AUTO_LIMIT vectors, then faiss, then hnsw if installed
Approximate backends trade a little recall for much faster search on large corpora
(Section 6.2: "can be replaced by approximate indexing methods").
"""

from pathlib import Path

import numpy as np

from spectrafuse.errors import MissingPackageError

ANN_AUTO_LIMIT = 20000
FILES = {"faiss": "vectors.faiss", "hnsw": "vectors.hnsw"}


def installed(backend):
    """True if the package for an approximate backend can be imported."""
    try:
        __import__({"faiss": "faiss", "hnsw": "hnswlib"}[backend])
        return True
    except ImportError:
        return False


def resolve_backend(backend, count):
    """Turn "auto" into a real backend; fail clearly if a backend is not installed."""
    if backend == "auto":
        if count <= ANN_AUTO_LIMIT:
            return "exact"
        return next((name for name in ("faiss", "hnsw") if installed(name)), "exact")
    if backend != "exact" and not installed(backend):
        package = {"faiss": "faiss-cpu", "hnsw": "hnswlib"}[backend]
        raise MissingPackageError("ann_backend '%s' needs: pip install %s" % (backend, package))
    return backend


def build_faiss(vectors, settings):
    import faiss
    index = faiss.IndexHNSWFlat(vectors.shape[1], settings["hnsw_m"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = settings["hnsw_ef_construction"]
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def build_hnsw(vectors, settings):
    import hnswlib
    index = hnswlib.Index(space="ip", dim=vectors.shape[1])
    index.init_index(max_elements=len(vectors), ef_construction=settings["hnsw_ef_construction"],
                     M=settings["hnsw_m"], random_seed=0)
    index.add_items(np.asarray(vectors, dtype=np.float32), np.arange(len(vectors)))
    return index


def build_vector_index(vectors, settings):
    """Build the index for the backend in settings["ann_backend"]."""
    backend = resolve_backend(settings["ann_backend"], len(vectors))
    builders = {"faiss": build_faiss, "hnsw": build_hnsw}
    index = builders[backend](vectors, settings) if backend in builders and len(vectors) else None
    return {"backend": backend if index is not None else "exact", "index": index,
            "vectors": vectors, "ef_search": settings["hnsw_ef_search"]}


def exact_search(vectors, queries, k, block_size=1024):
    """Exact top-k by inner product, in blocks of queries (memory: block_size x n)."""
    k = min(k, len(vectors))
    scores, ids = [], []
    for start in range(0, len(queries), block_size):
        sims = queries[start:start + block_size] @ vectors.T
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]  # the k best, not yet sorted
        top_sims = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_sims, axis=1, kind="stable")
        ids.append(np.take_along_axis(top, order, axis=1))
        scores.append(np.take_along_axis(top_sims, order, axis=1))
    return np.vstack(scores), np.vstack(ids)


def search_vectors(vector_index, queries, k):
    """Top-k (scores, ids) for every query row, best first. Missing results have id -1."""
    queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    k = min(k, len(vector_index["vectors"]))
    if k <= 0:
        return np.zeros((len(queries), 0)), np.zeros((len(queries), 0), dtype=np.int64)
    if vector_index["backend"] == "faiss":
        vector_index["index"].hnsw.efSearch = max(vector_index["ef_search"], k)
        scores, ids = vector_index["index"].search(queries, k)
        return scores, ids.astype(np.int64)
    if vector_index["backend"] == "hnsw":
        vector_index["index"].set_ef(max(vector_index["ef_search"], k))
        ids, distances = vector_index["index"].knn_query(queries, k=k)
        return 1.0 - distances, ids.astype(np.int64)  # hnswlib "ip" distance = 1 - dot
    return exact_search(vector_index["vectors"], queries, k)


def save_vector_index(vector_index, folder):
    """Save an approximate index next to the other index files (exact needs no file)."""
    backend, index = vector_index["backend"], vector_index["index"]
    if backend == "faiss":
        import faiss
        faiss.write_index(index, str(Path(folder) / FILES["faiss"]))
    elif backend == "hnsw":
        index.save_index(str(Path(folder) / FILES["hnsw"]))


def load_vector_index(folder, vectors, settings):
    """Load a saved approximate index if it matches the backend, otherwise build one."""
    backend = resolve_backend(settings["ann_backend"], len(vectors))
    path = Path(folder) / FILES.get(backend, "")
    if backend == "faiss" and path.is_file():
        import faiss
        index = faiss.read_index(str(path))
    elif backend == "hnsw" and path.is_file():
        import hnswlib
        index = hnswlib.Index(space="ip", dim=vectors.shape[1])
        index.load_index(str(path), max_elements=len(vectors))
    else:
        return build_vector_index(vectors, settings)
    return {"backend": backend, "index": index, "vectors": vectors,
            "ef_search": settings["hnsw_ef_search"]}
