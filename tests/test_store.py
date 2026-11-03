import numpy as np
import pytest

from spectrag.config import make_settings
from spectrag.utils.store import load_index, save_index


def test_save_and_load_round_trip(tmp_path):
    chunks = [{"id": 0, "doc_id": "a.md", "section": ["A", "Évora"], "text": "Olá"}]
    vectors = np.random.default_rng(0).random((1, 8)).astype(np.float32)
    save_index(tmp_path / "idx", {"chunks": chunks, "embeddings": vectors})
    assert sorted(p.name for p in (tmp_path / "idx").iterdir()) == ["chunks.json", "embeddings.npy"]
    index = load_index(tmp_path / "idx")
    assert index["chunks"] == chunks
    assert np.array_equal(index["embeddings"], vectors)
    assert index["embeddings"].dtype == np.float32


def test_settings_with_function_can_be_saved(tmp_path):
    settings = make_settings({"embedder": lambda texts: texts})
    save_index(tmp_path, {"settings": settings})
    loaded = load_index(tmp_path)["settings"]
    assert loaded["chunk_tokens"] == 1200
    assert "function" in loaded["embedder"]        # stored as text, not code


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "nothing")


def test_edge_arrays_and_signal_npz_round_trip(tmp_path):
    from spectrag.utils.store import array_to_edges, edges_to_array
    edges = [[0, 1, 0.5], [2, 7, 0.25]]
    signals = {"semantic": edges_to_array(edges), "citation": edges_to_array([])}
    save_index(tmp_path, {"edges": edges_to_array(edges), "signals": signals})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["edges.npy", "signals.npz"]
    index = load_index(tmp_path)
    assert array_to_edges(index["edges"]) == edges
    assert all(isinstance(v, int) for v in array_to_edges(index["edges"])[1][:2])
    assert index["signals"]["citation"].shape == (0, 3)
    assert "signals" not in load_index(tmp_path, skip=("signals",))


def test_old_json_edge_lists_still_load(tmp_path):
    from spectrag.utils.store import array_to_edges, save_json
    save_json(tmp_path / "edges.json", [[0, 1, 0.5]])            # format of older indexes
    assert array_to_edges(load_index(tmp_path)["edges"]) == [[0, 1, 0.5]]
