"""The documentation stays true: every setting is documented and every local link works."""

import re
from pathlib import Path

import pytest

from spectrafuse import DEFAULTS

ROOT = Path(__file__).parent.parent
SITE = ROOT / "docs"  # the MkDocs docs_dir (see mkdocs.yml)
DOCS = [ROOT / "README.md", ROOT / "CHANGELOG.md"] + sorted(SITE.rglob("*.md"))
# Developer notes that live in docs/ but are not pages of the site (exclude_docs in mkdocs.yml).
NOT_PAGES = {"DESIGN.md", "DEVELOPMENT_LOG.md", "PRODUCTION_READINESS.md", "PROGRESS.md"}
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def nav_pages():
    """The Markdown pages listed under nav: in mkdocs.yml, as paths relative to docs/."""
    nav = (ROOT / "mkdocs.yml").read_text().split("\nnav:\n", 1)[1]
    return re.findall(r":\s+(\S+\.md)\s*$", nav, re.MULTILINE)


def is_local(target):
    return not target.startswith(("http://", "https://", "mailto:"))


@pytest.mark.parametrize("path", [ROOT / "README.md", SITE / "guides" / "configuration.md"],
                         ids=lambda p: p.name)
def test_every_setting_is_documented(path):
    text = path.read_text()
    missing = [key for key in DEFAULTS if "`%s`" % key not in text]
    assert not missing, "%s is missing settings: %s" % (path.name, missing)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_local_links_and_images_exist(path):
    text = path.read_text()
    targets = LINK.findall(text) + re.findall(r'<img src="([^"]+)"', text)
    broken = [t for t in targets if is_local(t) and not (path.parent / t).exists()]
    assert not broken, "%s has broken links: %s" % (path.name, broken)


def test_navigation_lists_every_page():
    listed = nav_pages()
    assert all((SITE / page).is_file() for page in listed), "mkdocs.yml lists a missing page"
    pages = {p.relative_to(SITE).as_posix() for p in SITE.rglob("*.md") if p.name not in NOT_PAGES}
    assert pages == set(listed), "pages not in the nav of mkdocs.yml: %s" % sorted(pages - set(listed))


@pytest.mark.parametrize("page", nav_pages())
def test_site_pages_stay_inside_the_site(page):
    """MkDocs can only resolve links inside docs/ (and not to the excluded notes); others need a
    GitHub URL. GitBook syntax would show as plain text."""
    path = SITE / page
    text = path.read_text()
    outside = [t for t in LINK.findall(text) if is_local(t)
               and (SITE.resolve() not in (path.parent / t).resolve().parents
                    or Path(t).name in NOT_PAGES)]
    assert not outside, "%s links outside the site: %s" % (page, outside)
    assert "{%" not in text, "%s still has GitBook syntax" % page


@pytest.mark.parametrize("copy, source", [
    ("assets/benchmark/recall_at_k_chunk1200.svg", "benchmarks/results/multihoprag/recall_at_k_chunk1200.svg"),
    ("assets/benchmark/budget_chunk1200.svg", "benchmarks/results/multihoprag/budget_chunk1200.svg"),
    ("interactive/graph-view.html", "examples/graph_view.html"),
], ids=lambda value: Path(value).name)
def test_copies_match_their_sources(copy, source):
    assert (SITE / copy).read_bytes() == (ROOT / source).read_bytes(), "copy %s to docs/%s" % (source, copy)
