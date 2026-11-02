from pathlib import Path

import pytest

from spectrag.utils.loader import list_files, load_docs, read_file

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"


def test_load_sample_folder():
    docs = load_docs(EXAMPLES)
    ids = [doc["doc_id"] for doc in docs]
    assert ids == ["aurora_labs.md", "helios_energy.txt", "river_valley_farming.md"]
    assert "Maya Chen" in docs[0]["text"]


def test_load_single_file():
    docs = load_docs(EXAMPLES / "helios_energy.txt")
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "helios_energy.txt"


def test_nested_folders_and_filters(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("Alpha text.")
    (tmp_path / "sub" / "b.md").write_text("# Beta")
    (tmp_path / "image.png").write_bytes(b"not text")   # unsupported, ignored
    (tmp_path / "empty.txt").write_text("   ")           # empty, skipped
    docs = load_docs(tmp_path)
    assert [doc["doc_id"] for doc in docs] == ["a.txt", "sub/b.md"]


def test_list_files_ignores_unsupported(tmp_path):
    (tmp_path / "notes.csv").write_text("x,y")
    assert list_files(tmp_path) == []


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_docs("does/not/exist")


def test_read_pdf_paper():
    pytest.importorskip("pypdf")  # optional dependency
    paper = Path(__file__).parent.parent / "docs" / "2610.36991v2.pdf"
    assert "SpectRAG" in read_file(paper)


def test_read_file_keeps_unicode(tmp_path):
    path = tmp_path / "u.txt"
    path.write_text("Évora produces power.", encoding="utf-8")
    assert read_file(path) == "Évora produces power."
