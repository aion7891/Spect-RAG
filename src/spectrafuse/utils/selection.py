"""Token-budgeted, cluster-aware context selection (paper Section 3.8, Eq. 19-22,
Table 2 stage 4, Algorithm 1 lines 8-12).

1. Protected core: the 'core_size' best-scored candidates are always kept.
2. Greedy tail:    repeatedly add the candidate with the best U(c | C) / tokens(c)^rho,
                   among candidates above the precision floor, where
                   U = a*score + b*proximity + c*ClusterGain + d*DocGain - lambda*Redundancy.
3. Backfill:       if budget is left, add the best remaining candidates that still fit.
The total token count never goes over the budget B (Eq. 19).

All per-candidate values are numpy arrays, so one greedy step costs O(candidates).
"""

import numpy as np


def group_numbers(values):
    """Turn labels (cluster ids, doc ids) into numbers 0..g-1, one per candidate."""
    numbers = {}
    return np.array([numbers.setdefault(value, len(numbers)) for value in values]), len(numbers)


def start_state(scored, index):
    """Everything selection needs to remember, as arrays aligned with the candidates."""
    chunks = [index["chunks"][c] for c in scored["candidates"]]
    clusters, cluster_count = group_numbers([chunk["cluster"] for chunk in chunks])
    docs, doc_count = group_numbers([chunk["doc_id"] for chunk in chunks])
    size = len(chunks)
    return {"picked": [], "reasons": [], "used": 0,
            "is_picked": np.zeros(size, dtype=bool),
            "tokens": np.array([chunk["tokens"] for chunk in chunks], dtype=float),
            "clusters": clusters, "cluster_picks": np.zeros(cluster_count),
            "docs": docs, "doc_picks": np.zeros(doc_count),
            "max_similarity": np.zeros(size)}


def add_pick(state, k, reason, vectors):
    """Add candidate number k and update tokens, cluster/document counts and redundancy."""
    state["picked"].append(int(k))
    state["reasons"].append(reason)
    state["is_picked"][k] = True
    state["used"] += state["tokens"][k]
    state["cluster_picks"][state["clusters"][k]] += 1
    state["doc_picks"][state["docs"][k]] += 1
    state["max_similarity"] = np.maximum(state["max_similarity"], vectors @ vectors[k])


def fits(state, budget):
    """True for every candidate that is not picked yet and fits in the remaining budget."""
    return ~state["is_picked"] & (state["used"] + state["tokens"] <= budget)


def utilities(state, scored, weights):
    """Eq. 21 for every candidate. Gains are 1 for a new cluster/document, 1/2 for the second."""
    cluster_gain = 1.0 / (1 + state["cluster_picks"][state["clusters"]])
    doc_gain = 1.0 / (1 + state["doc_picks"][state["docs"]])
    redundancy = np.maximum(0.0, state["max_similarity"]) if state["picked"] else 0.0
    return (weights["a"] * scored["scores"] + weights["b"] * scored["proximity"]
            + weights["c"] * cluster_gain + weights["d"] * doc_gain
            - weights["lambda"] * redundancy)


def pick_tail(state, order, scored, settings, vectors):
    """Greedy step (Eq. 22): best utility per token^rho, above the precision floor."""
    above_floor = scored["scores"] >= settings["precision_floor"] * float(np.max(scored["scores"]))
    length = np.maximum(1.0, state["tokens"]) ** settings["length_power"]
    while True:
        utility = utilities(state, scored, settings["select_weights"])
        allowed = fits(state, settings["token_budget"]) & above_floor & (utility > 0)
        if not allowed.any():  # only chunks that add something (U > 0) are picked
            return
        value = np.where(allowed, utility / length, -np.inf)[order]
        add_pick(state, order[int(np.argmax(value))], "tail", vectors)  # ties: higher score


def select_context(scored, index, settings):
    """Return [(candidate number, reason), ...] in the order chunks were picked."""
    if len(scored["candidates"]) == 0:
        return []
    vectors = index["embeddings"][scored["candidates"]]
    order = np.argsort(-scored["scores"], kind="stable")  # best score first
    state = start_state(scored, index)
    budget = settings["token_budget"]
    for k in order:  # 1. protected core
        if len(state["picked"]) == settings["core_size"]:
            break
        if fits(state, budget)[k]:
            add_pick(state, k, "core", vectors)
    pick_tail(state, order, scored, settings, vectors)  # 2. greedy tail
    for k in order:  # 3. backfill, so the budget is not left unused
        if fits(state, budget)[k]:
            add_pick(state, k, "backfill", vectors)
    return list(zip(state["picked"], state["reasons"]))
