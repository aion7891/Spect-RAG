"""Every error SpectraFuse raises on purpose is a SpectRAGError, and still the old built-in one."""

from pathlib import Path

import pytest

from spectrafuse import (DocumentError, DocumentNotFoundError, IndexError_, LLMError,
                      MissingPackageError, SettingsError, SpectRAGError, SpectRAGIngestion,
                      SpectRAGRetrieval)
from spectrafuse.cli import main
from spectrafuse.utils import vector_index
from spectrafuse.utils.diffusion import proximity_scores
from spectrafuse.utils.embedder import hash_embed
from spectrafuse.utils.evaluation import gold_map
from spectrafuse.utils.llm import post_json
from spectrafuse.utils.loader import load_docs
from spectrafuse.utils.selectors import select
from spectrafuse.utils.store import load_index

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"


@pytest.mark.parametrize("error, builtin", [
    (DocumentNotFoundError, FileNotFoundError), (DocumentNotFoundError, DocumentError),
    (MissingPackageError, ImportError), (LLMError, ConnectionError),
])
def test_new_errors_keep_their_builtin(error, builtin):
    assert issubclass(error, SpectRAGError) and issubclass(error, builtin)


def test_graph_setting_per_call_is_a_settings_error(tmp_path):
    SpectRAGIngestion(tmp_path).ingest(EXAMPLES)
    with pytest.raises(SettingsError, match="Use rebuild_graph"):
        SpectRAGRetrieval(tmp_path).search("Helios", settings={"graph_k": 3})


def test_custom_embedder_without_embed_fn_is_a_settings_error(tmp_path):
    SpectRAGIngestion(tmp_path, {"embedder": lambda texts: hash_embed(texts, 32)}).ingest(EXAMPLES)
    with pytest.raises(SettingsError, match="Pass it again"):
        SpectRAGRetrieval(tmp_path)


def test_missing_folders_and_files(tmp_path):
    with pytest.raises(IndexError_, match="No index folder"):
        load_index(tmp_path / "missing")
    with pytest.raises(DocumentNotFoundError, match="No such file or folder"):
        load_docs(tmp_path / "missing")


def test_missing_ann_package(monkeypatch):
    monkeypatch.setattr(vector_index, "installed", lambda backend: False)
    with pytest.raises(MissingPackageError, match="pip install faiss-cpu"):
        vector_index.resolve_backend("faiss", 10)


def test_unreachable_llm_is_an_llm_error():
    with pytest.raises(LLMError, match="failed after 1 tries"):
        post_json("http://127.0.0.1:9/api/generate", {}, timeout=2, retries=0)  # port 9: closed


def test_unknown_names_inside_the_pipeline():
    with pytest.raises(SettingsError, match="Unknown selector"):
        select({"candidates": [0]}, None, {"selector": "greedy"})
    with pytest.raises(SettingsError, match="Unknown proximity_mode"):
        proximity_scores("fast", None, None, [0], None, {})


def test_evidence_not_in_any_chunk():
    with pytest.raises(DocumentError, match="Evidence not found"):
        gold_map([{"text": "Helios Energy is in Porto."}], ["Lisbon"])


def test_cli_reports_errors_without_a_traceback(tmp_path, capsys):
    assert main(["ingest", str(tmp_path / "missing"), "--index", str(tmp_path / "idx")]) == 1
    assert "spectrafuse: error: No such file or folder" in capsys.readouterr().err
