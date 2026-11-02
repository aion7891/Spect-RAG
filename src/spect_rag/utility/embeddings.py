"""Validated, normalized, cached embeddings (paper Table 1).

The callback receives a batch of already-prefixed strings and returns one vector
per string in the same order. It may use any provider; this module has no SDK.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .cache import cache_key, read_cache, write_cache


def normalize_embeddings(vectors, *, rows: int, dimension: int) -> np.ndarray:
    """Reject malformed/non-finite/zero vectors and return L2-normalized float32."""
    values = np.asarray(vectors)
    if (
        values.shape != (rows, dimension)
        or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
    ):
        raise ValueError(f"Expected a real numeric embedding matrix of shape {(rows, dimension)}.")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Embeddings must contain only finite numbers.")
    if rows == 0:
        return np.empty((0, dimension), dtype=np.float32)
    scales = np.max(np.abs(values), axis=1)
    if np.any(scales == 0):
        raise ValueError("Embedding vectors must not be zero.")
    # Scaling first avoids overflow when a valid provider returns large values.
    values = values / scales[:, None]
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(values, dtype=np.float32)


def embed_texts(
    texts: list[str], embed_fn: Callable[[list[str]], Sequence[Sequence[float]]], *,
    model_id: str, dimension: int = 768, prefix: str = "search_document: ",
    cache_dir: str | Path | None = None, batch_size: int = 64,
) -> np.ndarray:
    """Embed unique missing texts in batches, preserving input order and duplicates."""
    if not callable(embed_fn):
        raise TypeError("embed_fn must be a callable accepting a list of strings.")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Provide an explicit embedding model identity.")
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("Embedding dimension must be a positive integer.")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Embedding batch_size must be a positive integer.")
    if not isinstance(prefix, str):
        raise TypeError("Embedding prefix must be a string.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding inputs must be non-empty strings.")
    if not texts:
        return np.empty((0, dimension), dtype=np.float32)

    settings = {"model": model_id, "dimension": dimension, "prefix": prefix}
    keys = [cache_key("embeddings", settings, text) for text in texts]
    vectors = {}
    missing = {}
    for key, text in zip(keys, texts):
        if key in vectors or key in missing:
            continue
        cached = read_cache(cache_dir, "embeddings", key)
        if cached is None:
            missing[key] = text
        else:
            vectors[key] = normalize_embeddings([cached], rows=1, dimension=dimension)[0]
    pending = list(missing.items())
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        result = embed_fn([prefix + text for _, text in batch])
        normalized = normalize_embeddings(result, rows=len(batch), dimension=dimension)
        for (key, _), vector in zip(batch, normalized):
            vectors[key] = vector
            write_cache(cache_dir, "embeddings", key, vector.tolist())
    return np.stack([vectors[key] for key in keys])
