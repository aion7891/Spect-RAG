"""Sentence-distance breakpoints for unstructured documents (Section 3.3)."""

from collections.abc import Callable

import numpy as np

from .embeddings import normalize_embeddings


def semantic_segments(units: list[dict], embed_fn: Callable, dimension: int) -> list[int]:
    """Assign segment IDs using distances strictly above the 90th percentile.

    Table boundaries are hard boundaries. Table rows are not sentence-embedded,
    and distances never jump over a table. Tied distances create no forced break.
    """
    pairs = [
        (i - 1, i) for i in range(1, len(units))
        if units[i - 1]["kind"] == units[i]["kind"] == "text"
    ]
    breaks = set()
    if pairs:
        indices = sorted({i for pair in pairs for i in pair})
        vectors = normalize_embeddings(
            embed_fn([units[i]["text"] for i in indices]),
            rows=len(indices), dimension=dimension,
        )
        positions = {index: row for row, index in enumerate(indices)}
        distances = np.asarray([
            1.0 - np.clip(np.dot(vectors[positions[left]], vectors[positions[right]]), -1.0, 1.0)
            for left, right in pairs
        ])
        threshold = float(np.percentile(distances, 90))
        breaks = {right for (_, right), distance in zip(pairs, distances) if distance > threshold}
    segments = []
    segment = 0
    for i, unit in enumerate(units):
        if i and (
            i in breaks or unit["kind"] != units[i - 1]["kind"]
            or (unit["kind"] == "table" and unit["block_index"] != units[i - 1]["block_index"])
        ):
            segment += 1
        segments.append(segment)
    return segments
