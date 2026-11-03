"""End-to-end tests on REAL documents: the SpectRAG paper (PDF) and MultiHopRAG news articles.

The offline tests use the hash embedder, so they always run. The live tests use Ollama
(nomic-embed-text, gemma3:4b) and are skipped when Ollama is not running.
"""

import sys
import urllib.request
from pathlib import Path

import pytest

from spectrag import SpectRAGIngestion, SpectRAGRetrieval
from spectrag.utils.metrics import contains

ROOT = Path(__file__).parent.parent
PAPER = ROOT / "docs" / "2610.36991v2.pdf"
sys.path.insert(0, str(ROOT / "benchmarks"))
from mhr_data import gold_chunks, load_docs, load_questions, split  # noqa: E402

# Questions about the paper, and text that the retrieved context must contain.
PAPER_FACTS = [
    ("How many smallest eigenvectors give each chunk its spectral coordinates?", "32 smallest eigenvectors"),
    ("What token budget does the token-budget selection fill?", "12k-token budget"),
    ("How many hops does graph expansion use and what is the node cap?", "two hops and a cap of"),
]


def ollama_up():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except OSError:
        return False


def normalised(text):
    return " ".join(text.split())


@pytest.fixture(scope="module")
def paper_index(tmp_path_factory):
    pytest.importorskip("pypdf")
    folder = tmp_path_factory.mktemp("paper")
    info = SpectRAGIngestion(folder, {"chunk_tokens": 256, "overlap_tokens": 64}).ingest(PAPER)
    assert info["documents"] == 1 and info["chunks"] > 50
    return folder


@pytest.mark.parametrize("question, fact", PAPER_FACTS)
def test_paper_pdf_questions_find_the_right_passage(paper_index, question, fact):
    result = SpectRAGRetrieval(paper_index).search(question, token_budget=2000)
    assert result["tokens"] <= 2000
    assert fact in normalised(result["context"]), "fact not in the retrieved context"


@pytest.fixture(scope="module")
def news():
    """60 MultiHopRAG articles and the test questions whose evidence is all inside them."""
    _, test = split(load_questions())
    docs = load_docs()
    wanted = []
    for q in test:
        ids = {e["doc_id"] for e in q["evidence"]}
        if len(wanted) < 60 and not ids <= set(wanted):
            wanted += [i for i in sorted(ids) if i not in wanted]
    wanted = set(wanted[:60])
    covered = [q for q in test if {e["doc_id"] for e in q["evidence"]} <= wanted]
    return [d for d in docs if d["doc_id"] in wanted], covered


def test_multihoprag_articles_end_to_end(news, tmp_path):
    docs, questions = news
    info = SpectRAGIngestion(tmp_path, {"chunk_tokens": 256, "overlap_tokens": 64}).ingest_documents(docs)
    search = SpectRAGRetrieval(tmp_path)
    found = []
    for q in questions:
        result = search.search(q["question"], token_budget=2560)
        assert result["tokens"] <= 2560 and len({c["id"] for c in result["chunks"]}) == len(result["chunks"])
        gold, _ = gold_chunks(search.index["chunks"], q["evidence"])
        picked = {c["id"] for c in result["chunks"]}
        found += [1.0 if ids & picked else 0.0 for ids in gold.values()]
    assert info["documents"] == len(docs) and len(questions) >= 20
    assert sum(found) / len(found) > 0.5   # offline hash embedder; nomic does much better


@pytest.mark.skipif(not ollama_up(), reason="Ollama is not running")
def test_live_paper_with_nomic_and_gemma(tmp_path):
    pytest.importorskip("pypdf")
    from spectrag.utils.llm import ollama_llm
    SpectRAGIngestion(tmp_path, {"embedder": "ollama", "semantic_embedder": "hash",
                                 "chunk_tokens": 256, "overlap_tokens": 64}).ingest(PAPER)
    search = SpectRAGRetrieval(tmp_path)
    for question, fact in PAPER_FACTS:
        assert fact in normalised(search.search(question, token_budget=2000)["context"])
    reply = search.answer("How many dimensions do the spectral coordinates of each chunk have?",
                          ollama_llm("gemma3:4b"), token_budget=2000)["answer"]
    assert contains(reply, ["32"]), reply
