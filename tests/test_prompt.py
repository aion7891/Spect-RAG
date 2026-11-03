from spectrag.utils.prompt import build_prompt, format_chunk, format_context, group_by_document

CHUNKS = [
    {"doc_id": "b.md", "position": 3, "section": ["B", "Late"], "text": "B late.", "score": 0.4},
    {"doc_id": "a.txt", "position": 0, "section": [], "text": "A only.", "score": 0.6},
    {"doc_id": "b.md", "position": 1, "section": ["B", "Early"], "text": "B early.", "score": 0.9},
]


def test_documents_by_best_score_chunks_in_reading_order():
    groups = group_by_document(CHUNKS)
    assert [g[0]["doc_id"] for g in groups] == ["b.md", "a.txt"]
    assert [c["position"] for c in groups[0]] == [1, 3]


def test_format_context():
    assert format_context(CHUNKS) == ("### Source: b.md\n[B > Early]\nB early.\n\n"
                                      "[B > Late]\nB late.\n\n### Source: a.txt\nA only.")
    assert format_context([]) == ""


def test_section_label_only_when_heading_is_missing():
    with_heading = {"section": ["Doc", "1. History"], "text": "## 1. History\nOld."}
    without = {"section": ["Doc", "1. History"], "text": "More old text."}
    assert format_chunk(with_heading) == "## 1. History\nOld."
    assert format_chunk(without) == "[Doc > 1. History]\nMore old text."


def test_build_prompt_is_grounded():
    prompt = build_prompt("Who?", "### Source: a.txt\nA only.")
    assert "only the sources below" in prompt
    assert prompt.endswith("Question: Who?\nAnswer:")
    assert "### Source: a.txt" in prompt


def test_short_prompt_asks_for_bare_answer():
    assert "Reply with the answer only" in build_prompt("Who?", "ctx", short=True)
    assert "Reply with the answer only" not in build_prompt("Who?", "ctx")
