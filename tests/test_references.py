from spectrag.utils.references import add_references, extract_references, section_numbers


def test_section_references():
    refs = extract_references("See Section 2.1, Sec. 3 and §4.2. Also sections 5.")
    targets = [r["target"] for r in refs if r["type"] == "section"]
    assert targets == ["2.1", "3", "4.2", "5"]


def test_file_references_and_markdown_links():
    refs = extract_references("See notes.md and [the report](docs/report.pdf).")
    assert {"type": "doc", "target": "notes.md"} in refs
    assert {"type": "doc", "target": "report.pdf"} in refs


def test_duplicates_removed_and_plain_text_has_none():
    assert len(extract_references("Section 2. Again Section 2.")) == 1
    assert extract_references("Nothing to see here in 2019.") == []


def test_section_numbers():
    assert section_numbers(["Aurora Labs", "2. Products", "2.1 Sensors"]) == {"2", "2.1"}
    assert section_numbers(["Members"]) == set()


def test_add_references():
    chunks = [{"text": "Uses the model in Section 2.1."}]
    assert add_references(chunks)[0]["references"] == [{"type": "section", "target": "2.1"}]
