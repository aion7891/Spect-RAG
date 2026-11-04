"""Write an index folder safely, so there is always a complete index (DESIGN.md D63, D66).

The new index is written into '<index>.partial', then swapped in with three steps:
  1. the old index is renamed to '<index>.old'
  2. '<index>.partial' is renamed to '<index>'
  3. '<index>.old' is deleted
Renaming a folder is one instant step, so at every moment a complete index exists: the new one,
the old one, or, between steps 1 and 2 only, the old one under '<index>.old'. If the program
stops in that gap, opening the index reads '<index>.old', and the next ingest puts it back.
"""

import shutil
from pathlib import Path

from spectrafuse.errors import IndexError_
from spectrafuse.log import logger


def sibling(index_dir, suffix):
    """The folder '<index_dir><suffix>' next to the index, e.g. my_index.old."""
    target = Path(index_dir)
    return target.with_name(target.name + suffix)


def is_index(folder):
    """True if the folder holds a SpectraFuse index (its two main files)."""
    folder = Path(folder)
    return (folder / "info.json").is_file() and (folder / "chunks.json").is_file()


def recover_folder(index_dir):
    """Finish a replace that was interrupted: put '<index>.old' back, or delete it."""
    target, old = Path(index_dir), sibling(index_dir, ".old")
    if not old.exists():
        return
    if target.exists():
        shutil.rmtree(old)  # it stopped after step 2: the new index is already in place
    else:
        logger.warning("an earlier ingest stopped while replacing %s; putting the old index back", target)
        old.rename(target)  # it stopped after step 1: the old index is still complete


def prepare_folder(index_dir):
    """A fresh '<index_dir>.partial' folder to write into, so a crash never leaves half an index."""
    recover_folder(index_dir)
    target = Path(index_dir)
    if target.exists() and (not target.is_dir() or (any(target.iterdir()) and not is_index(target))):
        raise IndexError_("%s already exists and is not a SpectraFuse index; choose another folder" % target)
    partial = sibling(index_dir, ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    return partial


def replace_folder(partial, index_dir):
    """Swap the finished index into place (steps 1-3 above), never deleting the old one first."""
    target, old = Path(index_dir), sibling(index_dir, ".old")
    if target.exists():
        logger.info("replacing the existing index in %s", target)
        target.rename(old)                      # 1
    try:
        partial.rename(target)                  # 2
    except OSError:
        if old.exists() and not target.exists():
            old.rename(target)                  # undo step 1, so the old index stays usable
        raise
    if old.exists():
        shutil.rmtree(old)                      # 3


def readable_folder(index_dir):
    """The folder to open: the index, or '<index>.old' if an ingest stopped between steps 1 and 2."""
    target, old = Path(index_dir), sibling(index_dir, ".old")
    if not target.exists() and is_index(old):
        logger.warning("%s is missing, but %s holds the previous index (an ingest stopped); "
                       "reading it. Ingest again to put it back.", target, old)
        return old
    return target
