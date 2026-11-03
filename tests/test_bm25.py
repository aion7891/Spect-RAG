import math

import pytest

from spectrag.utils.bm25 import bm25_scores, build_bm25, idf, tokenize

TEXTS = ["Helios Energy builds solar farms.",
         "Aurora Labs builds sensors. Aurora sensors measure soil.",
         "The cooperative grows grapes and olives in the valley."]


def test_tokenize():
    assert tokenize("Helios Energy's S1!") == ["helios", "energy", "s", "s1"]


def test_build_bm25_postings_and_lengths():
    bm25 = build_bm25(TEXTS)
    ids, counts = bm25["postings"]["aurora"]
    assert ids.tolist() == [1] and counts.tolist() == [2]
    ids, counts = bm25["postings"]["builds"]
    assert ids.tolist() == [0, 1] and counts.tolist() == [1, 1]
    assert list(bm25["lengths"]) == [5, 8, 9]
    assert bm25["average_length"] == pytest.approx(22 / 3)


def test_idf_rare_words_count_more_and_never_negative():
    assert idf(1, 3) > idf(2, 3) > idf(3, 3) > 0
    assert idf(1, 3) == pytest.approx(math.log(1 + 2.5 / 1.5))


def test_scores_match_the_formula():
    bm25 = build_bm25(TEXTS)
    k1, b = 1.5, 0.75
    expected = idf(1, 3) * 2 * (k1 + 1) / (2 + k1 * (1 - b + b * 8 / (22 / 3)))
    assert bm25_scores(bm25, "aurora")[1] == pytest.approx(expected)


def test_scores_rank_matching_chunks():
    scores = bm25_scores(build_bm25(TEXTS), "solar farms")
    assert scores[0] > 0 and scores[1] == 0 and scores[2] == 0
    scores = bm25_scores(build_bm25(TEXTS), "builds sensors")
    assert scores[1] > scores[0] > 0                 # 'sensors' is rare, 'builds' is shared


def test_unknown_words_and_empty_index():
    assert not bm25_scores(build_bm25(TEXTS), "quantum").any()
    assert len(bm25_scores(build_bm25([]), "anything")) == 0
