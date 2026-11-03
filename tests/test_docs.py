"""The documentation stays true: every setting is documented and every local link works."""

import re
from pathlib import Path

import pytest

from spectrag import DEFAULTS

ROOT = Path(__file__).parent.parent
DOCS = [ROOT / "README.md", ROOT / "CHANGELOG.md"] + sorted((ROOT / "docs").glob("*.md"))


def test_every_setting_is_in_the_readme_table():
    readme = (ROOT / "README.md").read_text()
    missing = [key for key in DEFAULTS if "`%s`" % key not in readme]
    assert not missing, "README configuration table is missing: %s" % missing


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_local_links_and_images_exist(path):
    text = path.read_text()
    targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text) + re.findall(r'<img src="([^"]+)"', text)
    broken = [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))
              and not (path.parent / t).exists()]
    assert not broken, "%s has broken links: %s" % (path.name, broken)
