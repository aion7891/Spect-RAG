"""Algorithm 1, steps 4-5: deterministic, memory-bounded two-hop BFS."""

import heapq

import networkx as nx


def expand_candidates(
    graph: nx.Graph, seeds: list[str], relevance: dict[str, float], *, max_candidates: int = 200,
) -> tuple[list[str], nx.Graph, dict[str, int]]:
    """Preserve seeds, then prioritize relevance, edge strength, and corpus order.

    At each hop a bounded heap retains only the best remaining-capacity neighbors.
    High-degree adjacency still needs scanning, but its full union is not stored.
    """
    if type(max_candidates) is not int or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer.")
    if len(set(seeds)) != len(seeds) or len(seeds) > max_candidates:
        raise ValueError("Seeds must be unique and fit within the candidate cap.")
    if any(seed not in graph for seed in seeds):
        raise ValueError("Every seed must exist in the indexed graph.")
    order = {node: row for row, node in enumerate(graph)}
    candidates, visited, frontier = list(seeds), set(seeds), list(seeds)
    hops = {seed: 0 for seed in seeds}
    for hop in (1, 2):
        capacity = max_candidates - len(candidates)
        if not capacity or not frontier:
            break
        best, heap = {}, []
        for source in frontier:
            for target, attributes in graph[source].items():
                if target in visited:
                    continue
                quality = (relevance[target], attributes["weight"], -order[target])
                while heap and best.get(heap[0][1]) != heap[0][0]:
                    heapq.heappop(heap)
                if target in best:
                    if quality <= best[target]:
                        continue
                elif len(best) == capacity:
                    if quality <= heap[0][0]:
                        continue
                    _, removed = heapq.heappop(heap)
                    del best[removed]
                best[target] = quality
                heapq.heappush(heap, (quality, target))
                # Updated neighbor scores leave stale heap entries; compact them.
                if len(heap) > 2 * capacity:
                    heap = [(quality, node) for node, quality in best.items()]
                    heapq.heapify(heap)
        frontier = sorted(best, key=best.get, reverse=True)
        candidates.extend(frontier)
        visited.update(frontier)
        hops.update({node: hop for node in frontier})
    return candidates, graph.subgraph(candidates).copy(), hops
