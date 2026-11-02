from spectrag.utils.sections import has_structure, is_heading, split_by_sections


def test_markdown_headings():
    assert is_heading("## 2.1 Sensors") == (2, "2.1 Sensors")
    assert is_heading("# Title #") == (1, "Title")
    assert is_heading("#hashtag") is None


def test_numbered_headings():
    assert is_heading("2. Methods") == (1, "2. Methods")
    assert is_heading("2.1 Datasets") == (2, "2.1 Datasets")
    assert is_heading("1. Buy milk and bread from the shop.") is None   # list item
    assert is_heading("2014 was a good year for the company") is None    # lowercase after number


def test_has_structure():
    assert has_structure("# Title\nSome text.")
    assert has_structure("1. Intro\nText.\n2. Methods\nMore text.")
    assert not has_structure("Just a plain paragraph. Another sentence.")
    assert not has_structure("```\n# a code comment\n```\nPlain text.")


def test_split_by_sections_builds_paths():
    text = "# Doc\n\n## 1. History\nOld.\n\n## 2. Products\n\n### 2.1 Sensors\nS1.\n\n## 3. End\nBye."
    sections = split_by_sections(text)
    assert [s["section"] for s in sections] == [
        ["Doc", "1. History"],
        ["Doc", "2. Products", "2.1 Sensors"],
        ["Doc", "3. End"],
    ]
    assert sections[1]["heading"] == "### 2.1 Sensors"
    assert sections[1]["body"] == "S1."


def test_text_before_first_heading_is_kept():
    sections = split_by_sections("Intro text.\n\n# Part A\nBody.")
    assert sections[0] == {"section": [], "heading": "", "body": "Intro text."}


def test_headings_inside_code_blocks_are_ignored():
    sections = split_by_sections("# Real\n```\n# not a heading\n```\nText.")
    assert len(sections) == 1
    assert "# not a heading" in sections[0]["body"]
