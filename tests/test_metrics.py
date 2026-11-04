import pytest

from spectrafuse.utils.metrics import (contains, exact_match, normalize_answer, precision_recall_f1,
                                    reciprocal_rank, token_f1)


def test_precision_recall_f1():
    assert precision_recall_f1(2, 4, 2) == pytest.approx((0.5, 1.0, 2 / 3))
    assert precision_recall_f1(0, 4, 2) == (0.0, 0.0, 0.0)          # F1 = 0 when both are 0
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_reciprocal_rank():
    assert reciprocal_rank([7, 3, 9], {9}) == pytest.approx(1 / 3)
    assert reciprocal_rank([7, 3], {1}) == 0.0


def test_normalize_answer():
    assert normalize_answer("  The Douro Harvest-Festival! ") == "douro harvestfestival"
    assert normalize_answer("An apple, a pear") == "apple pear"


def test_exact_match_uses_best_reference():
    assert exact_match("the University of Coimbra.", ["Coimbra", "University of Coimbra"]) == 1.0
    assert exact_match("Coimbra University", ["University of Coimbra"]) == 0.0


def test_token_f1():
    assert token_f1("Coimbra University", ["University of Coimbra"]) == pytest.approx(0.8)
    assert token_f1("Porto", ["Porto"]) == 1.0
    assert token_f1("Lisbon", ["Porto"]) == 0.0


def test_contains_whole_words_only():
    assert contains("It is based in Porto, Portugal.", ["Porto"]) == 1.0
    assert contains("Portugal", ["Porto"]) == 0.0
