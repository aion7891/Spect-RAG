"""Your own embed function is saved by name, so the index can always be opened again."""

import functools
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from spectrag import (EmbedderError, SettingsError, SpectRAGIngestion, SpectRAGRetrieval,
                      make_settings)
from spectrag.utils.embedder import hash_embed
from spectrag.utils.embedder_id import (embedder_name, function_name, read_embedder_name,
                                        saved_settings)
from spectrag.utils.store import load_index

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"


class SmallEmbedder:
    """A callable object, like a wrapper around a model."""

    def __call__(self, texts):
        return hash_embed(texts, 64)


def small_embed(texts):
    return hash_embed(texts, 64)


@pytest.mark.parametrize("fn, name", [
    (small_embed, "small_embed"), (lambda texts: texts, "<lambda>"),
    (functools.partial(hash_embed, dims=64), "hash_embed"), (SmallEmbedder(), "SmallEmbedder"),
])
def test_function_name(fn, name):
    assert function_name(fn) == name
    assert embedder_name(fn) == "custom:" + name


def test_saved_settings_are_plain_json():
    settings = make_settings({"embedder": small_embed, "semantic_embedder": "hash"})
    saved = saved_settings(settings)
    assert saved["embedder"] == "custom:small_embed" and saved["semantic_embedder"] == "hash"
    assert json.loads(json.dumps(saved)) == saved            # nothing needs default=str
    assert make_settings(saved)["embedder"] == "custom:small_embed"   # and it validates


@pytest.mark.parametrize("value, expected", [
    ("hash", "hash"), ("ollama", "ollama"), (None, None), ("custom:f", "custom:f"),
    ("<function f at 0x10>", "custom:<function f at 0x10>"),                 # indexes before 0.1.2
    ("functools.partial(<function f>)", "custom:functools.partial(<function f>)"),
])
def test_read_embedder_name(value, expected):
    assert read_embedder_name(value) == expected


def test_typos_are_still_rejected():
    with pytest.raises(SettingsError, match="embedder must be"):
        make_settings({"embedder": "olama"})


@pytest.mark.parametrize("fn", [functools.partial(hash_embed, dims=64), SmallEmbedder()])
def test_partial_and_callable_object_indexes_open_again(tmp_path, fn):
    SpectRAGIngestion(tmp_path, {"embedder": fn}).ingest(EXAMPLES)
    saved = json.loads((tmp_path / "settings.json").read_text())["embedder"]
    assert saved == embedder_name(fn)
    result = SpectRAGRetrieval(tmp_path, embed_fn=fn).search("Where is Helios Energy based?")
    assert result["chunks"]
    with pytest.raises(SettingsError, match=r"\(%s\)\. Pass it again" % function_name(fn)):
        SpectRAGRetrieval(tmp_path)


def test_index_saved_by_an_older_version_opens_with_embed_fn(tmp_path):
    SpectRAGIngestion(tmp_path, {"embedder": small_embed}).ingest(EXAMPLES)
    path = tmp_path / "settings.json"
    old = dict(json.loads(path.read_text()), embedder="functools.partial(<function x at 0x1>)")
    path.write_text(json.dumps(old))                                  # what 0.1.1 wrote
    assert SpectRAGRetrieval(tmp_path, embed_fn=small_embed).search("Helios")["chunks"]


def reversed_embed(texts):
    """A different 768-d embedder: the same size as "hash", but other vectors."""
    return hash_embed([text[::-1] for text in texts], 768)


def test_index_keeps_an_embedder_fingerprint(tmp_path):
    SpectRAGIngestion(tmp_path).ingest(EXAMPLES)
    probe = load_index(tmp_path, only=("embedder_probe",))["embedder_probe"]
    assert probe.shape == (768,) and abs(float(probe @ probe) - 1) < 1e-5
    assert SpectRAGRetrieval(tmp_path).search("Helios")["chunks"]           # same embedder: fine


def test_a_different_embedder_of_the_same_size_is_refused(tmp_path):
    SpectRAGIngestion(tmp_path).ingest(EXAMPLES)                           # built with "hash"
    with pytest.raises(EmbedderError, match=r"different embedder .*built with \(hash\)"):
        SpectRAGRetrieval(tmp_path, embed_fn=reversed_embed)


def test_tiny_differences_are_allowed(tmp_path):
    SpectRAGIngestion(tmp_path).ingest(EXAMPLES)
    noise = np.random.default_rng(0)
    noisy = lambda texts: hash_embed(texts, 768) + noise.normal(0, 1e-4, (len(texts), 768))  # noqa: E731
    assert SpectRAGRetrieval(tmp_path, embed_fn=noisy).search("Helios")["chunks"]


def test_index_without_a_fingerprint_still_opens(tmp_path, caplog):
    SpectRAGIngestion(tmp_path).ingest(EXAMPLES)
    (tmp_path / "embedder_probe.npy").unlink()                             # as made by 0.1.1
    with caplog.at_level(logging.INFO, logger="spectrag"):
        assert SpectRAGRetrieval(tmp_path, embed_fn=reversed_embed).search("Helios")["chunks"]
    assert "embedder not checked" in caplog.text
