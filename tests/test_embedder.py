import json
import urllib.request

import numpy as np
import pytest

from spectrafuse.config import make_settings
from spectrafuse.utils.embedder import hash_embed, make_embedder, normalize, ollama_embed


def test_normalize_rows_have_length_one():
    vectors = normalize([[3, 4], [0, 0], [1, 0]])
    assert np.allclose(np.linalg.norm(vectors, axis=1), [1, 0, 1])
    assert np.allclose(vectors[0], [0.6, 0.8])


def test_hash_embed_shape_and_norm():
    vectors = hash_embed(["solar farm", "grape harvest", ""], dims=64)
    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors[:2], axis=1), 1.0)
    assert not vectors[2].any()                    # empty text -> zero vector


def test_hash_embed_is_deterministic():
    assert np.array_equal(hash_embed(["Maya Chen"]), hash_embed(["Maya Chen"]))


def test_hash_embed_similar_texts_score_higher():
    a, b, c = hash_embed(["solar farms in Portugal", "solar farms in Spain", "grape harvest"])
    assert a @ b > a @ c


def test_make_embedder_hash_uses_dims():
    embed = make_embedder(make_settings({"embed_dims": 32}))
    assert embed(["hello world"]).shape == (1, 32)


def test_make_embedder_custom_function_is_normalised():
    embed = make_embedder(make_settings({"embedder": lambda texts: [[2.0, 0.0] for _ in texts]}))
    assert np.allclose(embed(["a", "b"], kind="query"), [[1, 0], [1, 0]])


def test_make_embedder_unknown_raises():
    with pytest.raises(ValueError):
        make_embedder(make_settings({"embedder": "nope"}))


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_ollama_embed_sends_prefixes_and_batches(monkeypatch):
    sent = []

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        sent.append(body)
        return FakeResponse({"embeddings": [[1.0, 1.0] for _ in body["input"]]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    vectors = ollama_embed(["a", "b", "c"], kind="query", batch_size=2)
    assert [len(body["input"]) for body in sent] == [2, 1]
    assert sent[0]["input"][0] == "search_query: a"
    assert sent[0]["model"] == "nomic-embed-text"
    assert vectors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_ollama_connection_error_is_friendly(monkeypatch):
    def broken_urlopen(request, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", broken_urlopen)
    with pytest.raises(ConnectionError, match="ollama serve"):
        ollama_embed(["a"])


def ollama_is_running():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not ollama_is_running(), reason="Ollama is not running")
def test_real_nomic_embed_text():
    try:
        vectors = ollama_embed(["Solar farms in Portugal.", "Solar power in Spain.", "Grapes."])
    except ConnectionError as error:
        pytest.skip(str(error))  # server is up but the model is missing
    assert vectors.shape == (3, 768)               # paper: 768-dimensional
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
