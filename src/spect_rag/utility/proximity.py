"""Algorithm 1, steps 6-7: spectral/PPR proximity and candidate re-scoring."""

import networkx as nx
import numpy as np

from .ranking import normalize_positive


def spectral_support(graph: nx.Graph, coordinates: np.ndarray) -> dict[str, int | None]:
    """Identify components represented by the retained positive spectral basis.

    A zero coordinate row inside a supported component can be valid. Isolates
    and entire components missing from the retained basis have no spectral signal.
    """
    rows = {node: row for row, node in enumerate(graph)}
    result = {}
    for number, component in enumerate(nx.connected_components(graph)):
        indices = [rows[node] for node in component]
        supported = len(component) > 1 and bool(np.any(np.abs(coordinates[indices]) > 1e-12))
        result.update({node: number if supported else None for node in component})
    return result


def score_candidates(
    subgraph: nx.Graph, seeds: list[str], candidates: list[str],
    relevance: dict[str, float], coordinates: dict[str, np.ndarray],
    support: dict[str, int | None], *, alpha: float = 0.4,
) -> tuple[dict[str, dict], dict[str, float]]:
    """Mean spectral/PPR proximity; PPR alone where spectral evidence is unavailable."""
    if not candidates:
        return {}, {}
    weights = np.asarray([max(0.0, relevance[seed]) for seed in seeds])
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(seeds), 1 / len(seeds))
    seed_weights = dict(zip(seeds, (float(value) for value in weights)))
    personalization = {node: seed_weights.get(node, 0.0) for node in subgraph}
    if subgraph.number_of_edges():
        try:
            # Zero-mass disconnected components stay exactly zero from the start.
            ppr = nx.pagerank(
                subgraph, alpha=0.85, personalization=personalization,
                dangling=personalization, nstart=personalization,
                weight="weight", tol=1e-10, max_iter=1000,
            )
        except nx.PowerIterationFailedConvergence as exc:
            raise RuntimeError("Personalized PageRank did not converge.") from exc
    else:
        ppr = personalization
    ppr_scores = normalize_positive([ppr[node] for node in candidates])
    spectral_raw, available = [], []
    for node in candidates:
        compatible = [
            seed for seed in seeds
            if support[node] is not None and support[node] == support[seed] and seed_weights[seed] > 0
        ]
        available.append(bool(compatible))
        spectral_raw.append(sum(
            seed_weights[seed] * np.exp(-float(np.sum((coordinates[node] - coordinates[seed]) ** 2)))
            for seed in compatible
        ))
    spectral_scores = normalize_positive(spectral_raw)
    result = {}
    for row, node in enumerate(candidates):
        proximity = (
            0.5 * (spectral_scores[row] + ppr_scores[row])
            if available[row] else ppr_scores[row]
        )
        result[node] = {
            "relevance_score": float(relevance[node]),
            "ppr_raw": float(ppr[node]), "ppr_score": float(ppr_scores[row]),
            "spectral_raw": float(spectral_raw[row]), "spectral_score": float(spectral_scores[row]),
            "spectral_available": available[row], "proximity_score": float(proximity),
            "score": float((1 - alpha) * relevance[node] + alpha * proximity),
        }
    return result, seed_weights
