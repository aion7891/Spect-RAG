"""Integration tests: the whole online pipeline on the sample documents."""

from pathlib import Path

import pytest

from spectrag import SpectRAGIngestion, SpectRAGRetrieval
from spectrag.utils.embedder import hash_embed

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"
QUESTION = "Which company did the co-founder of Aurora Labs join after leaving?"


@pytest.fixture(scope="module")
def search(tmp_path_factory):
    folder = tmp_path_factory.mktemp("idx")
    SpectRAGIngestion(folder).ingest(EXAMPLES)
    return SpectRAGRetrieval(folder)


def test_search_result_shape(search):
    result = search.search(QUESTION)
    assert set(result) == {"question", "context", "chunks", "prompt", "tokens", "seeds",
                           "candidates"}
    assert [c["rank"] for c in result["chunks"]] == list(range(1, len(result["chunks"]) + 1))
    assert result["tokens"] == sum(c["tokens"] for c in result["chunks"])
    assert result["chunks"][0]["id"] == 0                        # the direct match comes first
    assert "### Source: aurora_labs.md" in result["context"]
    assert result["context"] in result["prompt"]


def test_budget_is_respected(search):
    for budget in [50, 150, 300, 12000]:
        result = search.search(QUESTION, token_budget=budget)
        assert result["tokens"] <= budget
    assert search.search(QUESTION, token_budget=10)["chunks"] == []   # nothing fits


def test_big_budget_takes_every_candidate_once(search):
    result = search.search(QUESTION, token_budget=12000)
    picked = [c["id"] for c in result["chunks"]]
    assert sorted(picked) == sorted(result["candidates"])
    assert len(set(picked)) == len(picked)


def test_bridge_chunk_reached_with_few_seeds(tmp_path):
    SpectRAGIngestion(tmp_path).ingest(EXAMPLES)
    search = SpectRAGRetrieval(tmp_path, {"seed_count": 2})
    result = search.search(QUESTION, token_budget=12000)
    docs = {c["doc_id"] for c in result["chunks"]}
    assert "helios_energy.txt" in docs                          # not a seed, found via the graph
    assert all(c["hop"] == 0 for c in result["chunks"] if c["id"] in result["seeds"])


def test_search_is_deterministic(search):
    first = search.search(QUESTION, token_budget=300)
    assert search.search(QUESTION, token_budget=300)["chunks"] == first["chunks"]


def test_custom_embed_function_must_be_passed_again(tmp_path):
    embed_fn = lambda texts: hash_embed(texts, 64)
    SpectRAGIngestion(tmp_path, {"embedder": embed_fn}).ingest(EXAMPLES)
    with pytest.raises(ValueError, match="embed_fn"):
        SpectRAGRetrieval(tmp_path)
    result = SpectRAGRetrieval(tmp_path, embed_fn=embed_fn).search(QUESTION)
    assert result["chunks"][0]["id"] == 0
