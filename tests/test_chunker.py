from pathlib import Path

import pytest

from spectrag.config import make_settings
from spectrag.utils.chunker import chunk_documents, merge_small_chunks, pack_units, take_overlap
from spectrag.utils.loader import load_docs, read_file
from spectrag.utils.tokens import count_tokens

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"
SENTENCES = ["Sentence number %d is here with some words." % i for i in range(30)]


def test_pack_units_respects_limit_and_keeps_sentences_whole():
    chunks = pack_units(SENTENCES, 60, 20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk) <= 60
        for line in chunk.splitlines():
            assert line in SENTENCES


def test_pack_units_overlap_repeats_last_sentences():
    chunks = pack_units(SENTENCES, 60, 20)
    first, second = chunks[0].splitlines(), chunks[1].splitlines()
    shared = [line for line in second if line in first]
    assert shared                                   # some overlap exists
    assert shared == first[-len(shared):]           # it is a suffix of the previous chunk
    assert second[:len(shared)] == shared           # and the start of the next one
    assert count_tokens("\n".join(shared)) <= 20


def test_no_overlap_when_zero():
    chunks = pack_units(SENTENCES, 60, 0)
    lines = [line for chunk in chunks for line in chunk.splitlines()]
    assert lines == SENTENCES


def test_overlap_never_copies_whole_chunk_or_tables():
    assert take_overlap([0], ["only one"], [3], 100) == []
    units = ["a.", "| t | 1 |", "b."]
    assert take_overlap([0, 1, 2], units, [2, 6, 2], 100) == [2]    # stops at the table


def test_merge_small_chunks_joins_previous():
    chunks = [{"text": "big chunk", "tokens": 50}, {"text": "tiny", "tokens": 3}]
    merged = merge_small_chunks(chunks, 32, 1200)
    assert len(merged) == 1
    assert merged[0]["text"] == "big chunk\ntiny"


def test_merge_never_exceeds_max():
    # Sizes chosen so the test holds with tiktoken and with the word estimate.
    texts = [" ".join(["alpha"] * 45), " ".join(["beta"] * 20), " ".join(["gamma"] * 8)]
    chunks = [{"text": t, "tokens": count_tokens(t)} for t in texts]
    merged = merge_small_chunks(chunks, 32, 60)
    assert [c["text"] for c in merged] == [texts[0], texts[1] + "\n" + texts[2]]
    assert all(c["tokens"] == count_tokens(c["text"]) <= 60 for c in merged)


def test_long_unit_is_cut_between_words():
    long_sentence = " ".join("word%d" % i for i in range(300))     # no full stop at all
    chunks = pack_units([long_sentence], 50, 10)
    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 50 for chunk in chunks)
    assert " ".join(chunks).split() == long_sentence.split()        # no word lost or cut


def test_paper_pdf_chunks_respect_limit():
    pytest.importorskip("pypdf")
    text = read_file(Path(__file__).parent.parent / "docs" / "2610.36991v2.pdf")
    for size, overlap in [(256, 64), (60, 20)]:
        settings = make_settings({"chunk_tokens": size, "overlap_tokens": overlap})
        chunks = chunk_documents([{"doc_id": "paper", "text": text}], settings)
        assert max(chunk["tokens"] for chunk in chunks) <= size


def test_chunks_never_exceed_limit_on_sample_docs():
    settings = make_settings({"chunk_tokens": 60, "overlap_tokens": 20})
    for chunk in chunk_documents(load_docs(EXAMPLES), settings):
        assert chunk["tokens"] <= 60


def test_sample_docs_default_settings():
    chunks = chunk_documents(load_docs(EXAMPLES))
    assert [c["id"] for c in chunks] == list(range(len(chunks)))
    aurora = [c for c in chunks if c["doc_id"] == "aurora_labs.md"]
    assert aurora[0]["mode"] == "structural"
    assert aurora[0]["section"] == ["Aurora Labs", "1. History"]
    assert [c["position"] for c in aurora] == list(range(len(aurora)))
    helios = [c for c in chunks if c["doc_id"] == "helios_energy.txt"]
    assert helios[0]["mode"] == "semantic" and helios[0]["section"] == []


def test_table_stays_in_one_chunk():
    chunks = chunk_documents(load_docs(EXAMPLES / "aurora_labs.md"),
                             make_settings({"chunk_tokens": 60, "overlap_tokens": 20}))
    with_table = [c for c in chunks if "| Aurora S1 |" in c["text"]]
    assert len(with_table) == 1
    assert "| FieldView |" in with_table[0]["text"]


def test_min_chunk_size_is_respected_with_default_size():
    chunks = chunk_documents(load_docs(EXAMPLES))
    for chunk in chunks:
        assert chunk["tokens"] >= 32
