"""Save and load the index folder (paper Table 1, "Persist").

An index is a plain dict. Each key becomes one file:
  numpy arrays  -> <key>.npy   (e.g. embeddings.npy)
  anything else -> <key>.json  (e.g. chunks.json, settings.json)
"""

import json
from pathlib import Path

import numpy as np


def save_json(path, data):
    """Write data as readable JSON. Things JSON cannot store (like functions) become text."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=1, default=str)


def load_json(path):
    """Read a JSON file."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_index(folder, index):
    """Save every item of the index dict into the folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in index.items():
        if isinstance(value, np.ndarray):
            np.save(folder / (key + ".npy"), value)
        else:
            save_json(folder / (key + ".json"), value)
    return folder


def load_index(folder):
    """Load every .npy and .json file in the folder back into a dict."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError("No index folder: " + str(folder))
    index = {}
    for path in sorted(folder.glob("*.npy")):
        index[path.stem] = np.load(path)
    for path in sorted(folder.glob("*.json")):
        index[path.stem] = load_json(path)
    return index
