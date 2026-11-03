"""Slow, simple reference versions of the vectorised code (the original loop versions).

tests/test_equivalence.py checks that the fast code gives exactly the same results.
"""

import math
from collections import Counter

import numpy as np


def ref_start_state(candidate_count):
    """Everything selection needs to remember while it adds chunks."""
    return {"picked": [], "reasons": [], "used": 0, "clusters": {}, "docs": {},
            "max_similarity": np.zeros(candidate_count)}


def ref_add_pick(state, k, reason, scored, index, vectors):
    """Add candidate number k and update tokens, cluster/document counts and redundancy."""
    chunk = index["chunks"][scored["candidates"][k]]
    state["picked"].append(k)
    state["reasons"].append(reason)
    state["used"] += chunk["tokens"]
    state["clusters"][chunk["cluster"]] = state["clusters"].get(chunk["cluster"], 0) + 1
    state["docs"][chunk["doc_id"]] = state["docs"].get(chunk["doc_id"], 0) + 1
    state["max_similarity"] = np.maximum(state["max_similarity"], vectors @ vectors[k])


def ref_fits(state, k, scored, index, budget):
    """True if candidate k is not picked yet and fits in the remaining budget."""
    chunk = index["chunks"][scored["candidates"][k]]
    return k not in state["picked"] and state["used"] + chunk["tokens"] <= budget


def ref_utility(state, k, scored, index, weights):
    """Eq. 21: U(c | C). Gains are 1 for a new cluster/document, 1/2 for the second, ..."""
    chunk = index["chunks"][scored["candidates"][k]]
    cluster_gain = 1.0 / (1 + state["clusters"].get(chunk["cluster"], 0))
    doc_gain = 1.0 / (1 + state["docs"].get(chunk["doc_id"], 0))
    redundancy = max(0.0, float(state["max_similarity"][k])) if state["picked"] else 0.0
    return (weights["a"] * scored["scores"][k] + weights["b"] * scored["proximity"][k]
            + weights["c"] * cluster_gain + weights["d"] * doc_gain
            - weights["lambda"] * redundancy)


def ref_pick_tail(state, order, scored, index, settings, vectors):
    """Greedy step (Eq. 22): best utility per token^rho, above the precision floor."""
    floor = settings["precision_floor"] * float(np.max(scored["scores"]))
    while True:
        best, best_value = None, 0.0
        for k in order:
            if scored["scores"][k] < floor or not ref_fits(state, k, scored, index, settings["token_budget"]):
                continue
            tokens = max(1, index["chunks"][scored["candidates"][k]]["tokens"])
            value = ref_utility(state, k, scored, index, settings["select_weights"])
            value = value / tokens ** settings["length_power"]
            if value > best_value:  # only chunks that add something (U > 0)
                best, best_value = k, value
        if best is None:
            return
        ref_add_pick(state, best, "tail", scored, index, vectors)


def ref_select_context(scored, index, settings):
    """Return [(candidate number, reason), ...] in the order chunks were picked."""
    if len(scored["candidates"]) == 0:
        return []
    vectors = index["embeddings"][scored["candidates"]]
    order = [int(k) for k in np.argsort(-scored["scores"], kind="stable")]  # best score first
    state = ref_start_state(len(order))
    for k in order:  # 1. protected core
        if len(state["picked"]) == settings["core_size"]:
            break
        if ref_fits(state, k, scored, index, settings["token_budget"]):
            ref_add_pick(state, k, "core", scored, index, vectors)
    ref_pick_tail(state, order, scored, index, settings, vectors)  # 2. greedy tail
    for k in order:  # 3. backfill, so the budget is not left unused
        if ref_fits(state, k, scored, index, settings["token_budget"]):
            ref_add_pick(state, k, "backfill", scored, index, vectors)
    return list(zip(state["picked"], state["reasons"]))


def ref_expand_graph(seeds, edges, hops, max_candidates):
    table = {}
    for i, j, w in edges:
        table.setdefault(i, {})[j] = w
        table.setdefault(j, {})[i] = w
    hop = {seed: 0 for seed in seeds[:max_candidates]}
    frontier = list(hop)
    for distance in range(1, hops + 1):
        strength = {}
        for node in frontier:
            for other, weight in table.get(node, {}).items():
                if other not in hop:
                    strength[other] = max(strength.get(other, 0.0), weight)
        layer = sorted(strength, key=lambda node: (-strength[node], node))
        layer = layer[:max_candidates - len(hop)]
        for node in layer:
            hop[node] = distance
        if not layer:
            break
        frontier = layer
    return list(hop)


def ref_rrf_top(rankings, k, count):
    fused = {}
    for ranked in rankings:
        for rank, chunk_id in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:count], fused


def ref_bm25(texts, query, k1=1.5, b=0.75):
    docs = [Counter(t.lower().split()) for t in texts]
    lengths = [sum(d.values()) for d in docs]
    average = sum(lengths) / len(lengths)
    scores = []
    for doc, length in zip(docs, lengths):
        score = 0.0
        for word in set(query.lower().split()):
            df = sum(1 for d in docs if word in d)
            if word in doc:
                idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
                tf = doc[word]
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
        scores.append(score)
    return np.array(scores)
