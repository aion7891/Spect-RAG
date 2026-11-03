"""Save and load the index folder (paper Table 1, "Persist").

An index is a plain dict. Each key becomes one file:
  numpy array              -> <key>.npy   (embeddings, spectral, edges, ...)
  dict of numpy arrays     -> <key>.npz   (the six raw edge signals)
  anything else            -> <key>.json  (chunks, settings, info)
Binary files keep big edge lists fast to save and small on disk. Arrays are always loaded
with allow_pickle=False, so opening an index never runs stored code.
"""

import json
from pathlib import Path

import numpy as np

from spectrag.errors import IndexError_


def save_json(path, data):
    """Write data as JSON. Small files are indented; things JSON cannot store become text."""
    with open(path, "w", encoding="utf-8") as file:
        indent = 1 if path.name in ("settings.json", "info.json") else None
        json.dump(data, file, ensure_ascii=False, indent=indent, default=str)


def load_json(path):
    """Read a JSON file."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def is_array_dict(value):
    return isinstance(value, dict) and value and all(isinstance(v, np.ndarray) for v in value.values())


def save_index(folder, index):
    """Save every item of the index dict into the folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in index.items():
        if isinstance(value, np.ndarray):
            np.save(folder / (key + ".npy"), value)
        elif is_array_dict(value):
            np.savez(folder / (key + ".npz"), **value)
        else:
            save_json(folder / (key + ".json"), value)
    return folder


def load_index(folder, skip=(), only=None):
    """Load every .npy, .npz and .json file in the folder (IndexError_ if it is missing).
    skip: keys to leave out; only: if given, load just these keys."""
    folder = Path(folder)
    if not folder.is_dir():
        raise IndexError_("No index folder: " + str(folder))
    index = {}
    for path in sorted(folder.iterdir()):
        if path.stem in skip or (only is not None and path.stem not in only):
            continue
        if path.suffix == ".npy":
            index[path.stem] = np.load(path, allow_pickle=False)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                index[path.stem] = {name: data[name] for name in data.files}
        elif path.suffix == ".json":
            index[path.stem] = load_json(path)
    return index


def edges_to_array(edges):
    """[[i, j, w], ...] -> float array of shape (E, 3). Chunk ids stay exact as floats."""
    return np.asarray(edges, dtype=np.float64).reshape(-1, 3)


def array_to_edges(array):
    """Float array (E, 3) or a JSON list (older indexes) -> [[i, j, w], ...] with int ids."""
    return [[int(i), int(j), float(w)] for i, j, w in array]


REQUIRED = ("chunks.json", "embeddings.npy", "spectral.npy", "eigenvalues.npy", "settings.json",
            "info.json")


def check_index_folder(folder):
    """Fail early, with a clear message, if the folder is not a complete, readable index."""
    if not folder.is_dir():
        raise IndexError_("No index at %s. Build one first: SpectRAGIngestion(%r).ingest(path)"
                          % (folder, str(folder)))
    missing = [name for name in REQUIRED if not (folder / name).is_file()]
    if not any((folder / name).is_file() for name in ("edges.npy", "edges.json")):
        missing.append("edges.npy")
    if missing:
        raise IndexError_("The index at %s is incomplete (missing %s). Ingest the documents again."
                          % (folder, ", ".join(missing)))
