"""Semantic chunking for text without headings (paper Section 3.3).

A new segment starts wherever the distance between two neighbouring sentences
is above the given percentile (the paper uses the 90th percentile).
"""

import re
from collections import Counter

import numpy as np


def word_cosine(text_a, text_b):
    """Cosine similarity of two texts using word counts (no model needed)."""
    a = Counter(re.findall(r"\w+", text_a.lower()))
    b = Counter(re.findall(r"\w+", text_b.lower()))
    dot = sum(a[word] * b[word] for word in a)
    norm = (sum(v * v for v in a.values()) ** 0.5) * (sum(v * v for v in b.values()) ** 0.5)
    return dot / norm if norm else 0.0


def sentence_distances(sentences, embed_fn=None):
    """Distance (1 - cosine) between each sentence and the next one."""
    if embed_fn is None:
        return [1 - word_cosine(a, b) for a, b in zip(sentences, sentences[1:])]
    vectors = np.array(embed_fn(sentences), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
    return list(1 - similarities)


def find_breakpoints(distances, percentile):
    """Return the sentence indexes where a new segment should start.

    Ties at the threshold count as breaks (word overlap often gives many 1.0 distances),
    but if every distance is the same there is no topic change, so no breaks.
    """
    if not distances:
        return []
    threshold = np.percentile(distances, percentile)
    lowest = min(distances)
    return [i + 1 for i, distance in enumerate(distances)
            if distance >= threshold and distance > lowest]


def split_by_meaning(sentences, percentile=90, embed_fn=None):
    """Group sentences into topically coherent segments (lists of sentences)."""
    if len(sentences) < 3:
        return [sentences] if sentences else []
    breaks = find_breakpoints(sentence_distances(sentences, embed_fn), percentile)
    starts = [0] + breaks
    ends = breaks + [len(sentences)]
    return [sentences[start:end] for start, end in zip(starts, ends)]
