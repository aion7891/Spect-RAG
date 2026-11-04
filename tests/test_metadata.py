"""pyproject.toml stays within the limits PyPI enforces on upload (twine check does not test them).

PyPI rejects a release with "400 Bad Request" when a project-URL label is longer than 32
characters, or the summary longer than 512 characters or more than one line.
"""

import re
from pathlib import Path

PYPROJECT = (Path(__file__).parent.parent / "pyproject.toml").read_text()


def project_urls():
    """{label: url} from the [project.urls] table (read by hand: tomllib needs Python 3.11)."""
    table = PYPROJECT.split("[project.urls]", 1)[1].split("\n[", 1)[0]
    return dict(re.findall(r'^"?([^"=\n]+?)"?\s*=\s*"([^"]+)"', table, re.MULTILINE))


def test_project_url_labels_fit_pypi():
    urls = project_urls()
    assert "Official implementation" in urls and "Paper" in urls
    too_long = [label for label in urls if len(label) > 32]
    assert not too_long, "PyPI allows 32 characters per project-URL label: %s" % too_long


def test_summary_fits_pypi():
    summary = re.search(r'^description = "(.*)"$', PYPROJECT, re.MULTILINE).group(1)
    assert len(summary) <= 512 and "\n" not in summary
    assert summary.startswith("Unofficial implementation of SpectRAG")
