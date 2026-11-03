"""The documentation stays true: every setting is documented and every local link works."""

import re
from pathlib import Path

import pytest

from spectrag import DEFAULTS

ROOT = Path(__file__).parent.parent
SITE = ROOT / "docs"  # the GitBook site root (see .gitbook.yaml)
DOCS = [ROOT / "README.md", ROOT / "CHANGELOG.md"] + sorted(SITE.rglob("*.md"))
# Developer notes that live in docs/ but are not pages of the site.
NOT_PAGES = {"SUMMARY.md", "DESIGN.md", "DEVELOPMENT_LOG.md", "PRODUCTION_READINESS.md", "PROGRESS.md"}


def summary_pages():
    """The pages listed in docs/SUMMARY.md, as paths relative to docs/."""
    return re.findall(r"^\s*\* \[[^\]]+\]\(([^)]+)\)", (SITE / "SUMMARY.md").read_text(), re.MULTILINE)


@pytest.mark.parametrize("path", [ROOT / "README.md", SITE / "guides" / "configuration.md"],
                         ids=lambda p: p.name)
def test_every_setting_is_documented(path):
    text = path.read_text()
    missing = [key for key in DEFAULTS if "`%s`" % key not in text]
    assert not missing, "%s is missing settings: %s" % (path.name, missing)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_local_links_and_images_exist(path):
    text = path.read_text()
    targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text) + re.findall(r'<img src="([^"]+)"', text)
    broken = [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))
              and not (path.parent / t).exists()]
    assert not broken, "%s has broken links: %s" % (path.name, broken)


def test_summary_lists_every_page():
    listed = summary_pages()
    assert all((SITE / page).is_file() for page in listed), "SUMMARY.md lists a missing page"
    pages = {p.relative_to(SITE).as_posix() for p in SITE.rglob("*.md") if p.name not in NOT_PAGES}
    assert pages == set(listed), "pages not in SUMMARY.md: %s" % sorted(pages - set(listed))


def test_site_links_stay_inside_the_site():
    """GitBook can only resolve links inside docs/; files elsewhere need a GitHub URL."""
    for page in summary_pages():
        path = SITE / page
        targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", path.read_text())
        outside = [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))
                   and SITE.resolve() not in (path.parent / t).resolve().parents]
        assert not outside, "%s links outside docs/: %s" % (page, outside)


@pytest.mark.parametrize("name", ["recall_at_k_chunk1200.svg", "budget_chunk1200.svg"])
def test_benchmark_charts_match_the_results(name):
    copy = SITE / "assets" / "benchmark" / name
    source = ROOT / "benchmarks" / "results" / "multihoprag" / name
    assert copy.read_bytes() == source.read_bytes(), "copy %s to docs/assets/benchmark/" % name
