"""The MultiHopRAG benchmark's gold labels and @k metrics, checked by hand."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
from mhr_data import gold_chunks, load_questions, split  # noqa: E402
from mhr_eval import metrics  # noqa: E402

CHUNKS = [{"id": 0, "doc_id": "a", "text": "Alpha beta gamma.\nDelta epsilon.", "tokens": 10},
          {"id": 1, "doc_id": "a", "text": "Delta epsilon.\nZeta eta theta.", "tokens": 10},
          {"id": 2, "doc_id": "b", "text": "Alpha beta gamma.", "tokens": 10},
          {"id": 3, "doc_id": "b", "text": "Iota kappa.", "tokens": 10}]


def test_gold_uses_the_evidence_article_only_and_falls_back_on_boundaries():
    evidence = [{"doc_id": "b", "fact": "Alpha beta   gamma."},                 # also in doc a
                {"doc_id": "a", "fact": "Delta epsilon. Zeta eta theta."},       # one chunk
                {"doc_id": "a", "fact": "Alpha beta gamma. Delta epsilon. Zeta eta"}]  # crosses
    gold, fallbacks = gold_chunks(CHUNKS, evidence)
    assert gold[0] == {2} and gold[1] == {1}
    assert fallbacks == 1 and gold[2] == {0, 1}


def test_metrics_at_k_by_hand():
    gold = {0: {2}, 1: {1}}
    row = metrics([3, 2, 0, 1], gold, CHUNKS, ks=(1, 2, 4))
    assert row["recall@1"] == 0.0 and row["recall@2"] == 0.5 and row["recall@4"] == 1.0
    assert row["precision@2"] == 0.5 and row["precision@4"] == 0.5
    assert row["f1@2"] == pytest.approx(0.5)
    assert row["mrr"] == 0.5 and row["complete"] == 1.0 and row["tokens"] == 40


def test_split_is_fixed_and_disjoint():
    questions = load_questions()
    validation, test = split(questions)
    assert len(validation) == 300 and len(test) == 1955
    assert not {q["id"] for q in validation} & {q["id"] for q in test}
    assert [q["id"] for q in split(questions)[0]] == [q["id"] for q in validation]
