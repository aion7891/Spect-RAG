"""BM25 keyword scoring (paper Table 2, "BM25 lexical retrieval"; Robertson and Zaragoza [11]).

score(chunk, query) = sum over query words of
    idf(word) * count * (k1 + 1) / (count + k1 * (1 - b + b * length / average_length))
"""

import math
import re
from collections import Counter

import numpy as np


def tokenize(text):
    """Lowercase words: 'Helios Energy!' -> ['helios', 'energy']."""
    return re.findall(r"\w+", text.lower())


def build_bm25(texts):
    """Build the keyword index once: word -> (chunk numbers, counts) as numpy arrays."""
    lists = {}
    lengths = []
    for chunk_id, text in enumerate(texts):
        counts = Counter(tokenize(text))
        lengths.append(sum(counts.values()))
        for word, count in counts.items():
            lists.setdefault(word, ([], []))
            lists[word][0].append(chunk_id)
            lists[word][1].append(count)
    postings = {word: (np.array(ids), np.array(counts, dtype=float))
                for word, (ids, counts) in lists.items()}
    average = sum(lengths) / len(lengths) if lengths else 0.0
    return {"postings": postings, "lengths": np.array(lengths, dtype=float),
            "average_length": average or 1.0}


def idf(doc_freq, doc_count):
    """Rare words count more. This form (as in Lucene) is never negative."""
    return math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


def bm25_scores(bm25, query, k1=1.5, b=0.75):
    """BM25 score of every chunk for the query (0 for chunks that share no word)."""
    lengths = bm25["lengths"]
    scores = np.zeros(len(lengths))
    length_norm = k1 * (1 - b + b * lengths / bm25["average_length"])  # one value per chunk
    for word in set(tokenize(query)):
        if word not in bm25["postings"]:
            continue
        ids, counts = bm25["postings"][word]  # only chunks that contain the word
        weight = idf(len(ids), len(lengths))
        scores[ids] += weight * counts * (k1 + 1) / (counts + length_norm[ids])
    return scores
