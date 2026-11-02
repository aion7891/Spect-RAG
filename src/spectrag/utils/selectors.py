"""All context selectors (paper Section 3.8 and Table 3, "Context selection").

settings["selector"] picks one:
  "spectrag"         - protected core + greedy tail (Eq. 21-22) + backfill (the default)
  "mmr"              - Maximal Marginal Relevance (Carbonell and Goldstein [4]) under the budget
  "protected_mmr"    - the protected core first, then MMR
  "anchor_expand_ce" - anchors (best chunks), then their graph neighbours ranked by a
                       cross-encoder if a rerank_fn is given (else by score), then backfill
  "topk"             - best scores first until the budget is full (an ablation / baseline)
The paper names the three alternatives but does not define them; see docs/DESIGN.md (D56-D58).
Every selector returns [(candidate number, reason), ...] and never exceeds the token budget.
"""

import numpy as np

from spectrag.utils.selection import add_pick, fits, select_context, start_state

SELECTORS = ("spectrag", "mmr", "protected_mmr", "anchor_expand_ce", "topk")


def setup(scored, index):
    """Vectors, best-first order and an empty selection state for the candidates."""
    vectors = index["embeddings"][scored["candidates"]]
    order = np.argsort(-scored["scores"], kind="stable")
    return vectors, order, start_state(scored, index)


def fill_in_order(state, order, budget, vectors, reason):
    """Add every candidate in 'order' that still fits."""
    for k in order:
        if fits(state, budget)[k]:
            add_pick(state, k, reason, vectors)


def pick_core(state, order, settings, vectors, reason="core"):
    """The core_size best-scored candidates that fit."""
    for k in order:
        if len(state["picked"]) >= settings["core_size"]:
            return
        if fits(state, settings["token_budget"])[k]:
            add_pick(state, k, reason, vectors)


def mmr_loop(state, order, scored, settings, vectors):
    """MMR: repeatedly pick argmax  l * score - (1 - l) * max similarity to the picks."""
    weight = settings["mmr_lambda"]
    while True:
        allowed = fits(state, settings["token_budget"])
        if not allowed.any():
            return
        redundancy = state["max_similarity"] if state["picked"] else 0.0
        value = weight * scored["scores"] - (1 - weight) * redundancy
        value = np.where(allowed, value, -np.inf)[order]
        add_pick(state, order[int(np.argmax(value))], "mmr", vectors)  # ties: higher score


def select_mmr(scored, index, settings, protected=False):
    vectors, order, state = setup(scored, index)
    if protected:
        pick_core(state, order, settings, vectors)
    mmr_loop(state, order, scored, settings, vectors)
    return list(zip(state["picked"], state["reasons"]))


def select_anchor_expand(scored, index, settings, graph, rerank_scores=None):
    """Anchors, then the anchors' graph neighbours (best rerank score first), then backfill."""
    vectors, order, state = setup(scored, index)
    rank_by = scored["scores"] if rerank_scores is None else np.asarray(rerank_scores)
    rerank_order = np.argsort(-rank_by, kind="stable")
    pick_core(state, order, settings, vectors, reason="anchor")
    position = {chunk_id: k for k, chunk_id in enumerate(scored["candidates"])}
    linked = np.zeros(len(order), dtype=bool)
    for k in list(state["picked"]):
        node = scored["candidates"][k]
        for other in graph["indices"][graph["indptr"][node]:graph["indptr"][node + 1]]:
            if int(other) in position:
                linked[position[int(other)]] = True
    fill_in_order(state, [k for k in rerank_order if linked[k]], settings["token_budget"],
                  vectors, "expand")
    fill_in_order(state, rerank_order, settings["token_budget"], vectors, "backfill")
    return list(zip(state["picked"], state["reasons"]))


def select_topk(scored, index, settings):
    vectors, order, state = setup(scored, index)
    fill_in_order(state, order, settings["token_budget"], vectors, "topk")
    return list(zip(state["picked"], state["reasons"]))


def select(scored, index, settings, graph=None, rerank_scores=None):
    """Run the selector named in settings["selector"]."""
    if len(scored["candidates"]) == 0:
        return []
    name = settings["selector"]
    if name == "spectrag":
        return select_context(scored, index, settings)
    if name in ("mmr", "protected_mmr"):
        return select_mmr(scored, index, settings, protected=name == "protected_mmr")
    if name == "anchor_expand_ce":
        return select_anchor_expand(scored, index, settings, graph, rerank_scores)
    if name == "topk":
        return select_topk(scored, index, settings)
    raise ValueError("Unknown selector: %r (choose from %s)" % (name, ", ".join(SELECTORS)))
