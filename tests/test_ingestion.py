"""Integration tests: the whole offline pipeline on the sample documents."""

from pathlib import Path

import numpy as np
import pytest

from spectrafuse import SpectRAGIngestion
from spectrafuse.utils.store import load_index

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"
EXPECTED_FILES = ["chunks.json", "edges.npy", "eigenvalues.npy", "embedder_probe.npy",
                  "embeddings.npy", "info.json",
                  "question_owners.npy", "question_vectors.npy", "settings.json",
                  "signals.npz", "spectral.npy"]


def test_ingest_writes_a_complete_index(tmp_path):
    info = SpectRAGIngestion(tmp_path / "idx").ingest(EXAMPLES)
    assert sorted(p.name for p in (tmp_path / "idx").iterdir()) == EXPECTED_FILES
    index = load_index(tmp_path / "idx")
    n = len(index["chunks"])
    assert info["documents"] == 3 and info["chunks"] == n
    assert index["embeddings"].shape == (n, 768)
    assert index["spectral"].shape == (n, min(32, n - 1))
    assert all(isinstance(c["cluster"], int) for c in index["chunks"])
    assert all("entities" in c and "references" in c for c in index["chunks"])


def test_graph_edges_are_valid(tmp_path):
    info = SpectRAGIngestion(tmp_path).ingest(EXAMPLES)
    index = load_index(tmp_path)
    threshold = index["settings"]["edge_threshold"]
    assert len(index["edges"]) == info["graph_edges"] > 0
    for i, j, weight in index["edges"]:
        assert i < j and threshold <= weight <= 1.0


def test_utility_weight_is_shared_when_no_questions(tmp_path):
    info = SpectRAGIngestion(tmp_path).ingest(EXAMPLES)
    assert info["weights_used"]["utility"] == 0.0
    assert sum(info["weights_used"].values()) == pytest.approx(1.0)


def test_question_fn_and_entity_fn_are_used(tmp_path):
    ingest = SpectRAGIngestion(tmp_path, question_fn=lambda text: ["What is in it?"],
                               entity_fn=lambda text: ["shared"])
    info = ingest.ingest(EXAMPLES)
    index = load_index(tmp_path)
    assert info["weights_used"]["utility"] > 0
    assert index["question_vectors"].shape == (info["chunks"], 768)
    assert index["question_owners"].tolist() == list(range(info["chunks"]))
    assert all(c["entities"] == ["shared"] for c in index["chunks"])


def test_settings_are_applied(tmp_path):
    info = SpectRAGIngestion(tmp_path, {"chunk_tokens": 60, "overlap_tokens": 20,
                                        "spectral_dims": 4, "n_clusters": 3}).ingest(EXAMPLES)
    index = load_index(tmp_path)
    assert info["chunks"] > 8
    assert index["spectral"].shape[1] == 4
    assert info["clusters"] <= 3


def test_ingest_is_deterministic(tmp_path):
    SpectRAGIngestion(tmp_path / "a").ingest(EXAMPLES)
    SpectRAGIngestion(tmp_path / "b").ingest(EXAMPLES)
    a, b = load_index(tmp_path / "a"), load_index(tmp_path / "b")
    assert np.array_equal(a["edges"], b["edges"])
    assert np.allclose(np.abs(a["spectral"]), np.abs(b["spectral"]))


def test_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        SpectRAGIngestion(tmp_path / "idx").ingest(tmp_path)


def test_semantic_embedder_is_used_only_for_chunk_breaks(tmp_path):
    calls = []

    def counting_embedder(texts):
        calls.append(len(texts))
        from spectrafuse.utils.embedder import hash_embed
        return hash_embed(texts, 32)

    SpectRAGIngestion(tmp_path, {"embedder": counting_embedder, "semantic_embedder": "hash"}).ingest(EXAMPLES)
    index = load_index(tmp_path)
    assert index["embeddings"].shape[1] == 32                         # chunks: the main embedder
    assert sum(calls) == len(index["chunks"]) + 1   # sentences went to "hash"; +1: the fingerprint
