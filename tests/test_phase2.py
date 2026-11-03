"""Proximity modes, vector indexes (exact / FAISS / HNSW), rebuild_graph and tuning."""

from pathlib import Path

import numpy as np
import pytest

from spectrag import SpectRAGIngestion, SpectRAGRetrieval, make_settings
from spectrag.utils.diffusion import PROXIMITY_MODES, heat_diffusion
from spectrag.utils.edges_content import ann_knn_edges, knn_edges
from spectrag.utils.graph import build_matrix, degrees
from spectrag.utils.spectral import normalized_laplacian, spectral_coords
from spectrag.utils.store import load_index
from spectrag.utils.tuning import expand_grid, sample_grid, tune
from spectrag.utils.vector_index import (build_vector_index, exact_search, load_vector_index,
                                         resolve_backend, save_vector_index, search_vectors)

EXAMPLES = Path(__file__).parent.parent / "examples"
TWO_COMMUNITIES = [[0, 1, 1.0], [0, 2, 1.0], [1, 2, 1.0],
                   [3, 4, 1.0], [3, 5, 1.0], [4, 5, 1.0], [2, 3, 0.1]]


def unit_vectors(n, d, seed):
    vectors = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_heat_diffusion_equals_matrix_exponential():
    expm = pytest.importorskip("scipy.linalg").expm
    laplacian = normalized_laplacian(build_matrix(TWO_COMMUNITIES, 6))
    coords, values = spectral_coords(laplacian, dims=5)               # the full basis
    index = {"spectral": coords, "eigenvalues": values, "edges": TWO_COMMUNITIES,
             "degree": degrees(TWO_COMMUNITIES, 6)}
    restart = np.array([0.7, 0.3, 0, 0, 0, 0])
    exact = expm(-1.5 * laplacian) @ restart                         # Eq. 17
    assert np.allclose(heat_diffusion(index, list(range(6)), restart, t=1.5), exact)


def test_exact_search_matches_brute_force():
    vectors, queries = unit_vectors(300, 16, 0), unit_vectors(7, 16, 1)
    scores, ids = exact_search(vectors, queries, 10, block_size=3)
    brute = np.argsort(-(queries @ vectors.T), axis=1)[:, :10]
    assert np.array_equal(ids, brute)
    assert np.allclose(scores, np.take_along_axis(queries @ vectors.T, brute, axis=1))


@pytest.mark.parametrize("backend", ["faiss", "hnsw"])
def test_approximate_backends_find_the_true_neighbours(backend, tmp_path):
    pytest.importorskip({"faiss": "faiss", "hnsw": "hnswlib"}[backend])
    vectors, queries = unit_vectors(2000, 32, 2), unit_vectors(50, 32, 3)
    settings = make_settings({"ann_backend": backend})
    index = build_vector_index(vectors, settings)
    assert index["backend"] == backend
    _, true_ids = exact_search(vectors, queries, 10)
    scores, ids = search_vectors(index, queries, 10)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, true_ids)])
    assert recall >= 0.95
    assert np.allclose(scores[:, 0], np.sum(queries * vectors[ids[:, 0]], axis=1), atol=1e-4)
    save_vector_index(index, tmp_path)
    loaded = load_vector_index(tmp_path, vectors, settings)
    assert np.array_equal(search_vectors(loaded, queries, 10)[1], ids)


def test_ann_semantic_edges_match_exact_edges():
    pytest.importorskip("faiss")
    vectors = unit_vectors(500, 24, 4)
    exact = {(i, j) for i, j, _ in knn_edges(vectors, 5)}
    index = build_vector_index(vectors, make_settings({"ann_backend": "faiss"}))
    approx = {(i, j) for i, j, _ in ann_knn_edges(index, 5)}
    assert len(exact & approx) / len(exact) >= 0.95


def test_auto_backend_is_exact_for_small_corpora():
    assert resolve_backend("auto", 1000) == "exact"
    assert resolve_backend("exact", 10 ** 7) == "exact"


@pytest.fixture(scope="module")
def eval_index(tmp_path_factory):
    folder = tmp_path_factory.mktemp("eval")
    SpectRAGIngestion(folder, {"chunk_tokens": 128, "overlap_tokens": 32}).ingest(EXAMPLES / "eval" / "docs")
    return folder


@pytest.mark.parametrize("mode", PROXIMITY_MODES)
def test_every_proximity_mode_runs_and_is_scaled(mode, eval_index):
    search = SpectRAGRetrieval(eval_index, {"proximity_mode": mode})
    result = search.search("Who leads the company that makes the Sunforge X7 panel?")
    assert result["chunks"] and result["tokens"] <= search.settings["token_budget"]
    _, scored, _ = search.retrieve("Who leads the company that makes the Sunforge X7 panel?")
    assert scored["proximity"].min() >= 0 and scored["proximity"].max() <= 1


@pytest.mark.parametrize("backend", ["faiss", "hnsw"])
def test_retrieval_with_an_approximate_backend(backend, eval_index):
    pytest.importorskip({"faiss": "faiss", "hnsw": "hnswlib"}[backend])
    question = "Who manages the solar park that uses Sunforge X7 panels?"
    exact = SpectRAGRetrieval(eval_index).search(question)
    approx = SpectRAGRetrieval(eval_index, {"ann_backend": backend, "dense_top_n": 20}).search(question)
    assert approx["chunks"][0]["id"] == exact["chunks"][0]["id"]


def test_rebuild_graph_equals_ingesting_with_the_same_threshold(tmp_path):
    settings = {"chunk_tokens": 128, "overlap_tokens": 32}
    SpectRAGIngestion(tmp_path / "a", dict(settings, edge_threshold=0.12)).ingest(EXAMPLES / "eval" / "docs")
    SpectRAGIngestion(tmp_path / "b", settings).ingest(EXAMPLES / "eval" / "docs")
    direct = load_index(tmp_path / "a")
    rebuilt = SpectRAGRetrieval(tmp_path / "b", {"edge_threshold": 0.12})
    assert [[i, j, w] for i, j, w in direct["edges"].tolist()] == rebuilt.index["edges"]
    assert np.allclose(np.abs(direct["spectral"]), np.abs(rebuilt.index["spectral"]))
    with pytest.raises(ValueError, match="rebuild_graph"):
        rebuilt.search("anything", settings={"graph_k": 3})


def test_tuning_grid_and_restores_the_graph(eval_index):
    from spectrag.utils.evaluation import load_questions
    search = SpectRAGRetrieval(eval_index, {"seed_count": 5})
    edges_before = list(search.index["edges"])
    questions = load_questions(EXAMPLES / "eval" / "questions.json")[:6]
    grid = {"alpha": [0.0, 0.3], "selector": ["spectrag", "mmr"], "edge_threshold": [0.05, 0.2]}
    results = tune(search, questions, grid, objective="recall", budget=400)
    assert len(results) == 8 == len(expand_grid(grid))
    assert results[0]["recall"] == max(row["recall"] for row in results)
    assert search.index["edges"] == edges_before                     # graph restored
    assert len(sample_grid(grid, 3, seed=1)) == 3
