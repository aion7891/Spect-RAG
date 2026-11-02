"""Download the MultiHopRAG dataset into benchmarks/data/multihoprag/ and verify it.

Run:  python benchmarks/download_multihoprag.py
The files are already in the repository; this only downloads the ones that are missing.
"""

import hashlib
import urllib.request
from pathlib import Path

FOLDER = Path(__file__).parent / "data" / "multihoprag"
URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/"


def expected_checksums():
    lines = (FOLDER / "SHA256SUMS").read_text().split("\n")
    return {line.split()[1]: line.split()[0] for line in lines if line.strip()}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name, checksum in expected_checksums().items():
        path = FOLDER / name
        if not path.is_file():
            print("downloading", name)
            urllib.request.urlretrieve(URL + name, path)
        status = "ok" if sha256(path) == checksum else "CHECKSUM MISMATCH"
        print("%-18s %s" % (name, status))


if __name__ == "__main__":
    main()
