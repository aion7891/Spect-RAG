from pathlib import Path

from spectrag.config import make_settings
from spectrag.utils.chunker import chunk_documents, merge_small_chunks, pack_units, take_overlap
from spectrag.utils.loader import load_docs
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
    assert take_overlap(["only one"], 100) == []
    assert take_overlap(["a.", "| t | 1 |", "b."], 100) == ["b."]


def test_merge_small_chunks_joins_previous():
    chunks = [{"text": "big chunk", "tokens": 50}, {"text": "tiny", "tokens": 3}]
    merged = merge_small_chunks(chunks, 32, 1200)
    assert len(merged) == 1
    assert merged[0]["text"] == "big chunk\ntiny"


def test_merge_never_exceeds_max():
    chunks = [{"text": "a", "tokens": 50}, {"text": "b", "tokens": 20}, {"text": "c", "tokens": 10}]
    merged = merge_small_chunks(chunks, 32, 60)
    assert [c["text"] for c in merged] == ["a", "b\nc"]   # 'b' cannot join 'a' (70 > 60)


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
