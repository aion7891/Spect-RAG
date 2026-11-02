"""Serializable evidence records and a bounded trace for inspecting retrieval."""


def evidence_records(chunks: list[dict], scores: dict, selection: dict) -> list[dict]:
    records = {chunk["chunk_id"]: chunk for chunk in chunks}
    selection_order = {node: rank for rank, node in enumerate(selection["selection_order"], start=1)}
    result = []
    for rank, node in enumerate(selection["selected"], start=1):
        metadata = {
            key: value for key, value in records[node].items()
            if key not in {"embedding", "utility_question_embeddings", "spectral_coords"}
        }
        result.append({
            **metadata, **scores[node], "rank": rank,
            "selection_order": selection_order[node], "selection_reason": selection["reasons"][node],
        })
    return result


def retrieval_trace(
    query: str, index: dict, initial: dict, candidates: list[str], hops: dict,
    scores: dict, seed_weights: dict, selection: dict, settings: dict,
) -> dict:
    ids = [chunk["chunk_id"] for chunk in index["chunks"]]
    rows = {node: row for row, node in enumerate(ids)}
    seeds = [ids[row] for row in initial["seed_rows"]]
    return {
        "query": query, "snapshot_generation": index["manifest"]["generation"],
        "settings": settings, "token_budget": settings["token_budget"],
        "used_tokens": selection["used_tokens"],
        "unused_tokens": settings["token_budget"] - selection["used_tokens"],
        "status": "ok" if selection["selected"] else ("no_positive_seed" if not seeds else "no_context_selected"),
        "seed_chunk_ids": seeds, "candidate_chunk_ids": candidates,
        "candidate_cap_reached": len(candidates) == settings["max_candidates"],
        "active_channels": [name for name, ranking in initial["rankings"].items() if ranking],
        "channel_rankings": {
            name: [
                {"chunk_id": ids[row], "rank": rank, "raw_score": float(initial["raw"][name][row])}
                for rank, row in enumerate(ranking[:settings["seed_k"]], start=1)
            ] for name, ranking in initial["rankings"].items()
        },
        "candidate_scores": [
            {
                "chunk_id": node, **scores[node], "hop": hops[node], "is_seed": node in seed_weights,
                "seed_weight": seed_weights.get(node, 0.0), "rrf_score": float(initial["rrf"][rows[node]]),
                "dense_raw": float(initial["raw"]["dense"][rows[node]]),
                "bm25_raw": float(initial["raw"]["bm25"][rows[node]]),
                "utility_raw": float(initial["raw"]["utility"][rows[node]]),
                "selection_reason": selection["reasons"].get(node),
                "exclusion_reason": selection["excluded"].get(node),
            }
            for node in candidates
        ],
        "selection_order": selection["selection_order"], "selected_chunk_ids": selection["selected"],
    }
