"""Algorithm 1, steps 8-12: protected core, precision tail, and optional backfill."""


def select_context(chunks: list[dict], scores: dict[str, dict], settings: dict) -> dict:
    """Use sum(chunk.token_count) as the evidence-text budget (Equation 19).

    Coverage only breaks score ties within 1e-9; it never adds a relevance bonus.
    Zero-score chunks are excluded at every stage, including protected core.
    """
    records = {chunk["chunk_id"]: chunk for chunk in chunks}
    order = {chunk["chunk_id"]: row for row, chunk in enumerate(chunks)}
    ranked = sorted(scores, key=lambda node: (-scores[node]["score"], order[node]))
    selected, reasons, covered = [], {}, set()
    used = 0

    def fits(node):
        return used + records[node]["token_count"] <= settings["token_budget"]

    def add(node, reason):
        nonlocal used
        selected.append(node)
        reasons[node] = reason
        covered.add(records[node]["doc_id"])
        used += records[node]["token_count"]

    for node in ranked:
        if len(selected) >= settings["core_size"]:
            break
        if scores[node]["score"] > 0 and fits(node):
            add(node, "core")
    while True:
        eligible = [
            node for node in ranked if node not in reasons and scores[node]["score"] > 0
            and scores[node]["score"] >= settings["precision_floor"] and fits(node)
        ]
        if not eligible:
            break
        best = scores[eligible[0]]["score"]
        tied = [node for node in eligible if best - scores[node]["score"] <= settings["coverage_tolerance"]]
        node = min(tied, key=lambda node: (records[node]["doc_id"] in covered, order[node]))
        add(node, "tail")
    if settings["backfill"]:
        for node in ranked:
            if node not in reasons and scores[node]["score"] > 0 and fits(node):
                add(node, "backfill")
    excluded = {}
    for node in ranked:
        if node in reasons:
            continue
        if scores[node]["score"] <= 0:
            excluded[node] = "zero_score"
        elif not fits(node):
            excluded[node] = "over_budget"
        else:
            excluded[node] = "below_precision_floor"
    return {
        "selection_order": selected, "selected": [node for node in ranked if node in reasons],
        "reasons": reasons, "excluded": excluded, "used_tokens": used,
    }
