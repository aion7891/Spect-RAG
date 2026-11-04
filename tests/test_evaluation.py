from pathlib import Path

import pytest

from spectrafuse import SpectRAGIngestion, SpectRAGRetrieval
from spectrafuse.utils.evaluation import (METHODS, evaluate_answers, evaluate_retrieval, fill_budget,
                                       gold_map, load_questions, score_selection)

EVAL = Path(__file__).parent.parent / "examples" / "eval"


def test_fill_budget_takes_chunks_in_order_while_they_fit():
    chunks = [{"tokens": 100}, {"tokens": 300}, {"tokens": 50}]
    assert fill_budget([1, 0, 2], chunks, 360) == [1, 2]


def test_score_selection():
    gold = {"fact a": {0}, "fact b": {2, 3}}
    scores = score_selection([5, 2, 9], gold)
    assert scores["recall"] == 0.5 and scores["complete"] == 0.0
    assert scores["precision"] == pytest.approx(1 / 3)
    assert scores["mrr"] == pytest.approx(1 / 2)
    assert score_selection([0, 3], gold)["complete"] == 1.0


def test_gold_map_fails_on_unknown_evidence():
    chunks = [{"text": "Helios Energy is based\nin Porto."}]
    assert gold_map(chunks, ["based in Porto"]) == {"based in Porto": {0}}   # across a line break
    with pytest.raises(ValueError, match="not found"):
        gold_map(chunks, ["based in Lisbon"])


@pytest.fixture(scope="module")
def search(tmp_path_factory):
    folder = tmp_path_factory.mktemp("eval")
    SpectRAGIngestion(folder, {"chunk_tokens": 128, "overlap_tokens": 32}).ingest(EVAL / "docs")
    return SpectRAGRetrieval(folder, {"seed_count": 5})


def test_every_evidence_snippet_exists_in_the_corpus(search):
    for item in load_questions(EVAL / "questions.json"):
        gold_map(search.index["chunks"], item["evidence"])         # raises if a snippet is wrong


def test_evaluate_retrieval_summary(search):
    questions = load_questions(EVAL / "questions.json")
    result = evaluate_retrieval(search, questions, budget=400)
    assert len(result["rows"]) == 12
    assert sorted(result["summary"]) == sorted(METHODS)
    for values in result["summary"].values():
        assert all(0 <= value <= 1 for value in values.values())
    for row in result["rows"]:                                      # same budget for everyone
        for ids in row["ids"].values():
            assert sum(search.index["chunks"][i]["tokens"] for i in ids) <= 400
    summary = result["summary"]
    assert summary["spectrag"]["recall"] >= summary["dense"]["recall"]


def test_evaluate_answers_with_a_fake_llm(search):
    questions = load_questions(EVAL / "questions.json")[:2]
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return "Porto"

    result = evaluate_answers(search, questions, 400, fake_llm)
    assert len(prompts) == 2 * len(METHODS)
    assert "Reply with the answer only" in prompts[0]               # short-answer prompt
    assert result["summary"]["dense"]["em"] == 0.5                  # right for Q1, wrong for Q2
