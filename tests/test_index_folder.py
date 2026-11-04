"""Replacing an index never leaves the folder without a complete index, even after a crash."""

import json
import logging
import shutil
from pathlib import Path

import pytest

from spectrafuse import SpectRAGIngestion, SpectRAGRetrieval
from spectrafuse.utils.index_folder import readable_folder, recover_folder, replace_folder

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"
SMALL_CHUNKS = {"chunk_tokens": 60, "overlap_tokens": 20}


def chunk_count(folder):
    return json.loads((folder / "info.json").read_text())["chunks"]


def two_indexes(tmp_path):
    """An old index in 'idx' and a finished new one in 'idx.partial' (with more chunks)."""
    SpectRAGIngestion(tmp_path / "idx").ingest(EXAMPLES)
    SpectRAGIngestion(tmp_path / "new", SMALL_CHUNKS).ingest(EXAMPLES)
    (tmp_path / "new").rename(tmp_path / "idx.partial")
    return tmp_path / "idx", tmp_path / "idx.partial", tmp_path / "idx.old"


def test_replace_swaps_in_the_new_index_and_cleans_up(tmp_path):
    target, partial, old = two_indexes(tmp_path)
    before = chunk_count(target)
    replace_folder(partial, target)
    assert chunk_count(target) > before
    assert not partial.exists() and not old.exists()


def test_a_failed_swap_keeps_the_old_index(tmp_path, monkeypatch):
    target, partial, old = two_indexes(tmp_path)
    before = chunk_count(target)
    real_rename = Path.rename

    def rename(self, other):  # step 2 fails, e.g. a permission error
        if self.name.endswith(".partial"):
            raise PermissionError("cannot rename " + self.name)
        return real_rename(self, other)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(PermissionError):
        replace_folder(partial, target)
    assert chunk_count(target) == before and not old.exists()   # step 1 was undone
    assert SpectRAGRetrieval(target).search("Helios")["chunks"]


def test_stopped_after_step_1_reads_the_old_index_then_recovers(tmp_path, caplog):
    target, partial, old = two_indexes(tmp_path)
    before = chunk_count(target)
    target.rename(old)                                  # the crash: step 1 done, step 2 not
    with caplog.at_level(logging.WARNING, logger="spectrafuse"):
        search = SpectRAGRetrieval(target)
    assert search.index_dir == old and search.search("Helios")["chunks"]
    assert "holds the previous index" in caplog.text
    assert old.exists() and not target.exists()          # opening an index moves nothing
    SpectRAGIngestion(target, SMALL_CHUNKS).ingest(EXAMPLES)   # the next ingest recovers
    assert chunk_count(target) > before and not old.exists() and not partial.exists()


def test_stopped_after_step_2_deletes_the_leftover(tmp_path):
    target, partial, old = two_indexes(tmp_path)
    target.rename(old)
    partial.rename(target)                              # the crash: steps 1-2 done, step 3 not
    assert readable_folder(target) == target            # the new index is read, not the old
    recover_folder(target)
    assert target.exists() and not old.exists()


def test_nothing_to_recover(tmp_path):
    recover_folder(tmp_path / "idx")                    # no index and no leftover: no error
    shutil.rmtree(tmp_path)
    assert readable_folder(tmp_path / "idx") == tmp_path / "idx"
