"""Relevance diffusion and proximity re-scoring (paper Section 3.7, Table 2 stage 3,
Algorithm 1 lines 6-7).

proximity(c) = mean of
  - Personalized PageRank from the seeds over the candidate subgraph (Eq. 18), and
  - spectral closeness of c to the seeds in the spectral coordinates.
score(c) = (1 - alpha) * relevance(c) + alpha * proximity(c)
"""

import numpy as np

from spectrag.utils.expansion import expand_graph, subgraph_matrix
from spectrag.utils.seeding import min_max
from spectrag.utils.spectral import inverse_sqrt


def restart_vector(candidates, seeds, relevance):
    """Where the random walk restarts: the seeds, in proportion to their relevance."""
    seed_set = set(seeds)
    restart = np.array([relevance[c] if c in seed_set else 0.0 for c in candidates])
    if restart.sum() <= 0:  # all seeds have relevance 0: treat them equally
        restart = np.array([1.0 if c in seed_set else 0.0 for c in candidates])
    return restart / restart.sum()


def personalized_pagerank(matrix, restart, beta=0.15, steps=30):
    """Eq. 18: s <- beta * restart + (1 - beta) * A_tilde @ s, with A_tilde = D^-1/2 A D^-1/2."""
    scale = inverse_sqrt(matrix.sum(axis=1))
    a_tilde = scale[:, None] * matrix * scale[None, :]
    scores = restart.copy()
    for _ in range(steps):
        scores = beta * restart + (1 - beta) * a_tilde @ scores
    return scores


def spectral_closeness(coords, eigenvalues, candidates, restart, t=1.0):
    """Cosine (mapped to 0..1) between each candidate and the seeds' weighted centre.

    Each spectral coordinate is scaled by exp(-t * eigenvalue), as in the heat kernel
    exp(-t L) of Eq. 17, so smooth community-level directions count most.
    """
    if coords.shape[1] == 0:
        return np.full(len(candidates), 0.5)
    points = coords[candidates] * np.exp(-t * np.asarray(eigenvalues))[None, :]
    centre = restart @ points  # seeds weighted by relevance (restart is 0 for non-seeds)
    norms = np.linalg.norm(points, axis=1) * np.linalg.norm(centre)
    cosine = np.divide(points @ centre, norms, out=np.zeros(len(candidates)), where=norms > 0)
    return (cosine + 1) / 2


def score_candidates(seeds, relevance, index, table, settings):
    """Run stages 2 and 3. Returns candidates and every score, aligned with 'candidates'."""
    candidates, hop = expand_graph(seeds, table, settings["hops"], settings["max_candidates"])
    restart = restart_vector(candidates, seeds, relevance)
    ppr = personalized_pagerank(subgraph_matrix(table, candidates), restart,
                                settings["ppr_beta"], settings["ppr_steps"])
    spectral = spectral_closeness(index["spectral"], index["eigenvalues"], candidates,
                                  restart, settings["diffusion_t"])
    proximity = (min_max(ppr) + spectral) / 2  # paper: "the mean of" both signals
    rel = np.array([relevance[c] for c in candidates])
    scores = (1 - settings["alpha"]) * rel + settings["alpha"] * proximity
    return {"candidates": candidates, "hop": [hop[c] for c in candidates],
            "relevance": rel, "ppr": ppr, "spectral": spectral,
            "proximity": proximity, "scores": scores}
