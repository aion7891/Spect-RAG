from pathlib import Path

import numpy as np
import pytest

from spectrag import SpectRAGIngestion, make_settings
from spectrag.utils.bm25 import build_bm25
from spectrag.utils.diffusion import (personalized_pagerank, restart_vector, score_candidates,
                                      spectral_closeness)
from spectrag.utils.embedder import make_embedder
from spectrag.utils.graph import build_matrix, neighbors
from spectrag.utils.seeding import find_seeds
from spectrag.utils.spectral import normalized_laplacian, spectral_coords
from spectrag.utils.store import load_index

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"
# Two triangles (0,1,2) and (3,4,5) joined by one weak bridge edge 2-3.
TWO_COMMUNITIES = [[0, 1, 1.0], [0, 2, 1.0], [1, 2, 1.0],
                   [3, 4, 1.0], [3, 5, 1.0], [4, 5, 1.0], [2, 3, 0.1]]


def test_restart_vector():
    assert np.allclose(restart_vector([5, 6, 7], seeds=[5, 7], relevance={5: 0.3, 6: 0.9, 7: 0.1}),
                       [0.75, 0, 0.25])
    assert np.allclose(restart_vector([5, 6], seeds=[5, 6], relevance={5: 0.0, 6: 0.0}), [0.5, 0.5])


def test_pagerank_matches_closed_form():
    matrix = build_matrix(TWO_COMMUNITIES, 6)
    restart = np.array([1.0, 0, 0, 0, 0, 0])
    scale = 1 / np.sqrt(matrix.sum(axis=1))
    a_tilde = scale[:, None] * matrix * scale[None, :]
    exact = 0.15 * np.linalg.solve(np.eye(6) - 0.85 * a_tilde, restart)   # fixed point of Eq. 18
    assert np.allclose(personalized_pagerank(matrix, restart, 0.15, steps=300), exact)


def test_pagerank_reaches_bridge_before_far_nodes():
    scores = personalized_pagerank(build_matrix(TWO_COMMUNITIES, 6), np.eye(6)[0])
    assert scores[0] > scores[1] > scores[3] > scores[4]    # seed > own community > bridge > far
    assert scores[3] > scores[5]


def test_spectral_closeness_prefers_seed_community():
    laplacian = normalized_laplacian(build_matrix(TWO_COMMUNITIES, 6))
    coords, values = spectral_coords(laplacian, 32)
    closeness = spectral_closeness(coords, values, list(range(6)), np.eye(6)[0])
    assert closeness.min() >= 0 and closeness.max() <= 1
    assert min(closeness[:3]) > max(closeness[3:])
    assert np.allclose(spectral_closeness(np.zeros((6, 0)), [], [0, 1], np.array([1, 0])), 0.5)


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    folder = tmp_path_factory.mktemp("idx")
    SpectRAGIngestion(folder).ingest(EXAMPLES)
    index = load_index(folder)
    settings = make_settings(dict(index["settings"], seed_count=2))
    embed = make_embedder(settings)
    question = "Which company did the co-founder of Aurora Labs join after leaving?"
    bm25 = build_bm25([c["text"] for c in index["chunks"]])
    seeds = find_seeds(question, embed([question], kind="query")[0], index, bm25, settings)
    return index, settings, seeds, neighbors(index["edges"], len(index["chunks"]))


def test_expansion_brings_in_bridge_chunks(sample):
    index, settings, seeds, table = sample
    result = score_candidates(seeds["seeds"], seeds["relevance"], index, table, settings)
    helios = [c["id"] for c in index["chunks"] if c["doc_id"] == "helios_energy.txt"]
    assert not set(helios) & set(seeds["seeds"])           # not found by seeding ...
    assert set(helios) <= set(result["candidates"])        # ... but reached through the graph
    assert result["candidates"][:2] == seeds["seeds"] and result["hop"][:2] == [0, 0]


def test_scores_follow_the_formula(sample):
    index, settings, seeds, table = sample
    result = score_candidates(seeds["seeds"], seeds["relevance"], index, table, settings)
    alpha = settings["alpha"]
    assert np.allclose(result["scores"], (1 - alpha) * result["relevance"] + alpha * result["proximity"])
    assert result["proximity"].min() >= 0 and result["proximity"].max() <= 1
    settings = dict(settings, alpha=0.0)
    result = score_candidates(seeds["seeds"], seeds["relevance"], index, table, settings)
    assert np.allclose(result["scores"], result["relevance"])


def test_max_candidates_is_respected(sample):
    index, settings, seeds, table = sample
    result = score_candidates(seeds["seeds"], seeds["relevance"], index, table,
                              dict(settings, max_candidates=3))
    assert len(result["candidates"]) == 3
    assert all(len(result[key]) == 3 for key in ["hop", "relevance", "ppr", "spectral", "scores"])
