"""Phase 4: settings validation, errors, index format, safe saving, logging and the CLI."""

import json
import logging
from pathlib import Path

import pytest

import spectrafuse
from spectrafuse import (DocumentError, EmbedderError, IndexError_, NoDocumentsError, SettingsError,
                      SpectRAGError, SpectRAGIngestion, SpectRAGRetrieval, UnknownSettingError,
                      make_settings)
from spectrafuse.cli import main, parse_sets
from spectrafuse.utils.embedder import hash_embed

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"


@pytest.mark.parametrize("overrides, message", [
    ({"chunk_tokens": 0}, "chunk_tokens must be a whole number >= 16"),
    ({"seed_count": 2.5}, "seed_count must be a whole number"),
    ({"hops": True}, "hops must be a whole number"),
    ({"alpha": 1.5}, "alpha must be a number between 0 and 1"),
    ({"selector": "best"}, "selector must be one of spectrag, mmr"),
    ({"proximity_mode": "fast"}, "proximity_mode must be one of"),
    ({"ann_backend": "annoy"}, "ann_backend must be one of"),
    ({"overlap_tokens": 1200}, "overlap_tokens (1200) must be smaller than chunk_tokens"),
    ({"edge_weights": {"semantic": -1}}, "edge_weights must be numbers >= 0"),
    ({"edge_weights": {"colour": 1}}, "edge_weights must be a dict with keys from"),
    ({"select_weights": {"a": 1}}, "select_weights needs keys a, b, c, d, lambda"),
    ({"n_clusters": 0}, "n_clusters must be None"),
    ({"embedder": "openai"}, "embedder must be 'hash', 'ollama' or a function"),
])
def test_bad_settings_give_clear_errors(overrides, message):
    with pytest.raises(SettingsError) as error:
        make_settings(overrides)
    assert message in str(error.value)
    assert isinstance(error.value, ValueError) and isinstance(error.value, SpectRAGError)


def test_unknown_setting_suggests_the_right_name():
    with pytest.raises(UnknownSettingError, match="did you mean 'token_budget'"):
        make_settings({"token_buget": 100})
    with pytest.raises(KeyError):                                  # still a KeyError
        make_settings({"nonsense": 1})


def test_every_default_is_valid():
    assert make_settings() == spectrafuse.DEFAULTS


def test_document_errors():
    ingest = SpectRAGIngestion("unused")
    with pytest.raises(NoDocumentsError):
        ingest.ingest_documents([])
    with pytest.raises(DocumentError, match="must be a dict"):
        ingest.ingest_documents([{"text": "no id"}])
    with pytest.raises(DocumentError, match="same doc_id"):
        ingest.ingest_documents([{"doc_id": "a", "text": "x"}, {"doc_id": "a", "text": "y"}])
    with pytest.raises(NoDocumentsError, match="empty"):
        ingest.ingest_documents([{"doc_id": "a", "text": "   "}])


def test_index_errors(tmp_path):
    with pytest.raises(IndexError_, match="No index at"):
        SpectRAGRetrieval(tmp_path / "missing")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "info.json").write_text("{}")
    with pytest.raises(IndexError_, match="incomplete .*chunks.json"):
        SpectRAGRetrieval(tmp_path / "broken")
    assert issubclass(IndexError_, FileNotFoundError)


def test_future_index_format_is_refused(tmp_path):
    SpectRAGIngestion(tmp_path / "idx").ingest(EXAMPLES)
    info_path = tmp_path / "idx" / "info.json"
    info = json.loads(info_path.read_text())
    assert info["format_version"] == spectrafuse.INDEX_FORMAT and info["spectrafuse_version"] == spectrafuse.__version__
    info_path.write_text(json.dumps(dict(info, format_version=99)))
    with pytest.raises(IndexError_, match="Upgrade spectrafuse"):
        SpectRAGRetrieval(tmp_path / "idx")


def test_embedder_with_the_wrong_size_is_caught(tmp_path):
    SpectRAGIngestion(tmp_path, {"embedder": lambda texts: hash_embed(texts, 32)}).ingest(EXAMPLES)
    with pytest.raises(EmbedderError, match="64 dimensions but the index has 32"):
        SpectRAGRetrieval(tmp_path, embed_fn=lambda texts: hash_embed(texts, 64))  # caught on open
    search = SpectRAGRetrieval(tmp_path, embed_fn=lambda texts: hash_embed(texts, 32))
    with pytest.raises(EmbedderError, match="64 dimensions but the index has 32"):
        search.search("anything", query_vector=hash_embed(["anything"], 64)[0])  # and per query


def test_saving_is_safe(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "my_file.txt").write_text("keep me")
    with pytest.raises(IndexError_, match="not a SpectraFuse index"):
        SpectRAGIngestion(tmp_path / "notes").ingest(EXAMPLES)
    assert (tmp_path / "notes" / "my_file.txt").read_text() == "keep me"   # nothing deleted
    first = SpectRAGIngestion(tmp_path / "idx").ingest(EXAMPLES)
    second = SpectRAGIngestion(tmp_path / "idx", {"chunk_tokens": 60, "overlap_tokens": 20}).ingest(EXAMPLES)
    assert second["chunks"] > first["chunks"]                     # an old index is replaced
    assert not (tmp_path / "idx.partial").exists()
    assert len(SpectRAGRetrieval(tmp_path / "idx").index["chunks"]) == second["chunks"]


def test_logging_is_silent_by_default_and_can_be_enabled(tmp_path, caplog):
    logger = spectrafuse.enable_logging("INFO")
    with caplog.at_level(logging.INFO, logger="spectrafuse"):
        SpectRAGIngestion(tmp_path / "idx").ingest(EXAMPLES)
    steps = [r.getMessage() for r in caplog.records]
    assert any(s.startswith("chunking:") for s in steps) and any(s.startswith("saved index") for s in steps)
    assert len([h for h in logger.handlers if getattr(h, "spectrafuse", False)]) == 1
    spectrafuse.enable_logging("INFO")                                # no duplicate handler
    assert len([h for h in logger.handlers if getattr(h, "spectrafuse", False)]) == 1
    logger.setLevel(logging.WARNING)


def test_cli_ingest_search_info(tmp_path, capsys):
    index = str(tmp_path / "idx")
    assert main(["ingest", str(EXAMPLES), "--index", index, "--set", "graph_k=5"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["chunks"] >= 8 and info["documents"] == 3            # 8 with tiktoken, 9 without
    assert main(["search", "Where is Helios Energy based?", "--index", index, "--json", "--budget", "200"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tokens"] <= 200 and result["chunks"]
    assert main(["info", "--index", index]) == 0
    assert json.loads(capsys.readouterr().out)["format_version"] == spectrafuse.INDEX_FORMAT
    assert main(["search", "q", "--index", str(tmp_path / "nope")]) == 1
    assert "No index at" in capsys.readouterr().err


def test_cli_settings_parsing():
    assert parse_sets(["alpha=0.5", "selector=mmr", "n_clusters=null", "edge_weights={\"semantic\": 1}"]) == {
        "alpha": 0.5, "selector": "mmr", "n_clusters": None, "edge_weights": {"semantic": 1}}
