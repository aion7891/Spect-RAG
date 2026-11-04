"""Shared test fixtures.

paper_pdf: the SpectRAG paper (arXiv:2610.36991v2), used as a real PDF in a few tests. It is not
stored in this repository (it is the authors' work), so it is found in this order:
  1. the file named by SPECTRAG_PAPER_PDF;
  2. a copy cached earlier in ~/.cache/spectrag-tests/;
  3. a fresh download from arXiv, only when SPECTRAG_DOWNLOAD_PAPER=1 (CI sets it).
Otherwise the tests that need it are skipped, so the suite still runs offline.
"""

import os
import urllib.request
from pathlib import Path

import pytest

PAPER_URL = "https://arxiv.org/pdf/2610.36991v2"
PAPER_CACHE = Path.home() / ".cache" / "spectrag-tests" / "2610.36991v2.pdf"


def download_paper(path):
    """Save the paper from arXiv to 'path'. Returns True if it worked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(PAPER_URL, timeout=30) as reply:
            data = reply.read()
    except OSError:
        return False
    if not data.startswith(b"%PDF"):
        return False
    path.write_bytes(data)
    return True


@pytest.fixture(scope="session")
def paper_pdf():
    """The path of the SpectRAG paper PDF, or skip the test if it is not available."""
    pytest.importorskip("pypdf")
    given = os.environ.get("SPECTRAG_PAPER_PDF")
    if given:
        if not Path(given).is_file():
            pytest.fail("SPECTRAG_PAPER_PDF=%s is not a file" % given)
        return Path(given)
    if PAPER_CACHE.is_file():
        return PAPER_CACHE
    if os.environ.get("SPECTRAG_DOWNLOAD_PAPER") == "1" and download_paper(PAPER_CACHE):
        return PAPER_CACHE
    pytest.skip("needs the paper PDF: set SPECTRAG_DOWNLOAD_PAPER=1 to fetch it from arXiv")
