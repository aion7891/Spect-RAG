"""Relevance diffusion and proximity re-scoring (paper Section 3.7, Table 2 stage 3,
Algorithm 1 lines 6-7).

settings["proximity_mode"] picks the graph signal:
  "hybrid"   - mean of Personalized PageRank (Eq. 18) and spectral closeness (the paper's
               "hybrid mode", Table 2; the default)
  "ppr"      - Personalized PageRank from the seeds over the candidate subgraph only
  "spectral" - spectral closeness to the seeds only
  "heat"     - heat-kernel diffusion exp(-t L) s (Eq. 17) from the stored eigenpairs
  "none"     - no graph signal (an ablation; same as alpha = 0)
score(c) = (1 - alpha) * relevance(c) + alpha * proximity(c)
"""

PROXIMITY_MODES = ("hybrid", "ppr", "spectral", "heat", "none")

import numpy as np

from spectrafuse.errors import SettingsError
from spectrafuse.utils.clusters import trivial_eigenvector
from spectrafuse.utils.expansion import expand_graph, subgraph_matrix
from spectrafuse.utils.graph import degrees
from spectrafuse.utils.seeding import min_max
from spectrafuse.utils.spectral import inverse_sqrt


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


def heat_diffusion(index, candidates, restart, t=1.0):
    """Eq. 17, s' = exp(-t L_sym) s, truncated to the stored eigenpairs:
    s' ~ u1 (u1 . s) + U exp(-t Lambda) U^T s, where u1 = sqrt(degree) has eigenvalue 0."""
    coords, eigenvalues = index["spectral"], np.asarray(index["eigenvalues"])
    if "degree" not in index:
        index["degree"] = degrees(index["edges"], len(coords))
    start = np.zeros(len(coords))
    start[candidates] = restart  # relevance mass on the seeds
    trivial = trivial_eigenvector(index["degree"])
    diffused = trivial * (trivial @ start) + coords @ (np.exp(-t * eigenvalues) * (coords.T @ start))
    return diffused[candidates]


def proximity_scores(mode, index, table, candidates, restart, settings):
    """The graph signal for every candidate, scaled to 0..1. Returns (proximity, ppr, spectral)."""
    ppr = spectral = np.zeros(len(candidates))
    if mode in ("hybrid", "ppr"):
        ppr = personalized_pagerank(subgraph_matrix(table, candidates), restart,
                                    settings["ppr_beta"], settings["ppr_steps"])
    if mode in ("hybrid", "spectral"):
        spectral = spectral_closeness(index["spectral"], index["eigenvalues"], candidates,
                                      restart, settings["diffusion_t"])
    if mode == "hybrid":
        return (min_max(ppr) + spectral) / 2, ppr, spectral  # paper: "the mean of" both
    if mode == "ppr":
        return min_max(ppr), ppr, spectral
    if mode == "spectral":
        return spectral, ppr, spectral
    if mode == "heat":
        return min_max(heat_diffusion(index, candidates, restart, settings["diffusion_t"])), ppr, spectral
    if mode == "none":
        return np.zeros(len(candidates)), ppr, spectral
    raise SettingsError("Unknown proximity_mode: %r (choose from %s)" % (mode, ", ".join(PROXIMITY_MODES)))


def score_candidates(seeds, relevance, index, table, settings):
    """Run stages 2 and 3. Returns candidates and every score, aligned with 'candidates'."""
    candidates, hop = expand_graph(seeds, table, settings["hops"], settings["max_candidates"])
    restart = restart_vector(candidates, seeds, relevance)
    proximity, ppr, spectral = proximity_scores(settings["proximity_mode"], index, table,
                                                candidates, restart, settings)
    rel = np.array([relevance[c] for c in candidates])
    scores = (1 - settings["alpha"]) * rel + settings["alpha"] * proximity
    return {"candidates": candidates, "hop": [hop[c] for c in candidates],
            "relevance": rel, "ppr": ppr, "spectral": spectral,
            "proximity": proximity, "scores": scores}
