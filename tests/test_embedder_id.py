"""Your own embed function is saved by name, so the index can always be opened again."""

import functools
import json
from pathlib import Path

import pytest

from spectrag import SettingsError, SpectRAGIngestion, SpectRAGRetrieval, make_settings
from spectrag.utils.embedder import hash_embed
from spectrag.utils.embedder_id import (embedder_name, function_name, read_embedder_name,
                                        saved_settings)

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
