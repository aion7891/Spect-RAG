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
