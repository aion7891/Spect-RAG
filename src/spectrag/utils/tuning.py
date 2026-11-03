"""Hyperparameter tuning on validation questions: grid search or random search.

    grid = {"alpha": [0.1, 0.3, 0.5], "selector": ["spectrag", "mmr"], "edge_threshold": [0.05, 0.1]}
    results = tune(search, validation_questions, grid, objective="recall", budget=2560)
    best = results[0]["settings"]

Retrieval settings are applied per call. Graph settings (edge_threshold, graph_k,
edge_weights, ...) are applied with search.rebuild_graph() from the saved signals: no
document is embedded again. Trials are ordered so the graph is rebuilt only when needed.
"""

import itertools
import json
import random

import numpy as np

from spectrag.retrieval import GRAPH_SETTINGS
from spectrag.utils.evaluation import gold_map, score_selection

METRICS = ["recall", "complete", "precision", "f1", "mrr"]


def expand_grid(grid):
    """{"a": [1, 2], "b": [3]} -> [{"a": 1, "b": 3}, {"a": 2, "b": 3}]."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def sample_grid(grid, trials, seed=0):
    """'trials' different random combinations (all of them if the grid is small)."""
    combos = expand_grid(grid)
    if trials is None or trials >= len(combos):
        return combos
    return random.Random(seed).sample(combos, trials)


def graph_part(combo):
    """The graph settings of a trial, as a comparable key."""
    return json.dumps({k: combo[k] for k in sorted(combo) if k in GRAPH_SETTINGS}, sort_keys=True)


def prepare(search, questions):
    """Gold evidence and query vectors, computed once for all trials."""
    golds = [q["gold"] if "gold" in q else gold_map(search.index["chunks"], q["evidence"])
             for q in questions]
    vectors = search.embed([q["question"] for q in questions], kind="query")
    return golds, vectors


def score_trial(search, questions, golds, vectors, overrides):
    """Average metrics of SpectRAG with these settings over the questions."""
    rows = []
    for item, gold, vector in zip(questions, golds, vectors):
        _, scored, picks = search.retrieve(item["question"], overrides, query_vector=vector)
        rows.append(score_selection([scored["candidates"][k] for k, _ in picks], gold))
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def tune(search, questions, grid, objective="recall", budget=None, trials=None, seed=0):
    """Try settings combinations; return [{"settings": ..., metric: value, ...}] best first."""
    golds, vectors = prepare(search, questions)
    combos = sorted(sample_grid(grid, trials, seed), key=graph_part)  # group graph rebuilds
    original = {key: search.settings[key] for key in GRAPH_SETTINGS}
    current, results = graph_part(original), []
    for combo in combos:
        graph = {k: v for k, v in combo.items() if k in GRAPH_SETTINGS}
        if graph and graph_part(dict(original, **graph)) != current:
            search.rebuild_graph(dict(original, **graph))
            current = graph_part(dict(original, **graph))
        overrides = {k: v for k, v in combo.items() if k not in GRAPH_SETTINGS}
        if budget is not None:
            overrides["token_budget"] = budget
        results.append(dict({"settings": combo}, **score_trial(search, questions, golds,
                                                               vectors, overrides)))
    if current != graph_part(original):
        search.rebuild_graph(original)  # leave the search object as it was
    return sorted(results, key=lambda row: -row[objective])
