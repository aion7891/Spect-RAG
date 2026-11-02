"""Token-budgeted, cluster-aware context selection (paper Section 3.8, Eq. 19-22,
Table 2 stage 4, Algorithm 1 lines 8-12).

1. Protected core: the 'core_size' best-scored candidates are always kept.
2. Greedy tail:    repeatedly add the candidate with the best U(c | C) / tokens(c)^rho,
                   among candidates above the precision floor, where
                   U = a*score + b*proximity + c*ClusterGain + d*DocGain - lambda*Redundancy.
3. Backfill:       if budget is left, add the best remaining candidates that still fit.
The total token count never goes over the budget B (Eq. 19).
"""

import numpy as np


def start_state(candidate_count):
    """Everything selection needs to remember while it adds chunks."""
    return {"picked": [], "reasons": [], "used": 0, "clusters": {}, "docs": {},
            "max_similarity": np.zeros(candidate_count)}


def add_pick(state, k, reason, scored, index, vectors):
    """Add candidate number k and update tokens, cluster/document counts and redundancy."""
    chunk = index["chunks"][scored["candidates"][k]]
    state["picked"].append(k)
    state["reasons"].append(reason)
    state["used"] += chunk["tokens"]
    state["clusters"][chunk["cluster"]] = state["clusters"].get(chunk["cluster"], 0) + 1
    state["docs"][chunk["doc_id"]] = state["docs"].get(chunk["doc_id"], 0) + 1
    state["max_similarity"] = np.maximum(state["max_similarity"], vectors @ vectors[k])


def fits(state, k, scored, index, budget):
    """True if candidate k is not picked yet and fits in the remaining budget."""
    chunk = index["chunks"][scored["candidates"][k]]
    return k not in state["picked"] and state["used"] + chunk["tokens"] <= budget


def utility(state, k, scored, index, weights):
    """Eq. 21: U(c | C). Gains are 1 for a new cluster/document, 1/2 for the second, ..."""
    chunk = index["chunks"][scored["candidates"][k]]
    cluster_gain = 1.0 / (1 + state["clusters"].get(chunk["cluster"], 0))
    doc_gain = 1.0 / (1 + state["docs"].get(chunk["doc_id"], 0))
    redundancy = max(0.0, float(state["max_similarity"][k])) if state["picked"] else 0.0
    return (weights["a"] * scored["scores"][k] + weights["b"] * scored["proximity"][k]
            + weights["c"] * cluster_gain + weights["d"] * doc_gain
            - weights["lambda"] * redundancy)


def pick_tail(state, order, scored, index, settings, vectors):
    """Greedy step (Eq. 22): best utility per token^rho, above the precision floor."""
    floor = settings["precision_floor"] * float(np.max(scored["scores"]))
    while True:
        best, best_value = None, 0.0
        for k in order:
            if scored["scores"][k] < floor or not fits(state, k, scored, index, settings["token_budget"]):
                continue
            tokens = max(1, index["chunks"][scored["candidates"][k]]["tokens"])
            value = utility(state, k, scored, index, settings["select_weights"])
            value = value / tokens ** settings["length_power"]
            if value > best_value:  # only chunks that add something (U > 0)
                best, best_value = k, value
        if best is None:
            return
        add_pick(state, best, "tail", scored, index, vectors)


def select_context(scored, index, settings):
    """Return [(candidate number, reason), ...] in the order chunks were picked."""
    if len(scored["candidates"]) == 0:
        return []
    vectors = index["embeddings"][scored["candidates"]]
    order = [int(k) for k in np.argsort(-scored["scores"], kind="stable")]  # best score first
    state = start_state(len(order))
    for k in order:  # 1. protected core
        if len(state["picked"]) == settings["core_size"]:
            break
        if fits(state, k, scored, index, settings["token_budget"]):
            add_pick(state, k, "core", scored, index, vectors)
    pick_tail(state, order, scored, index, settings, vectors)  # 2. greedy tail
    for k in order:  # 3. backfill, so the budget is not left unused
        if fits(state, k, scored, index, settings["token_budget"]):
            add_pick(state, k, "backfill", scored, index, vectors)
    return list(zip(state["picked"], state["reasons"]))
