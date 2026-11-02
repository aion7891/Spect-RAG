import pytest

from spectrag.utils.semantic_split import (find_breakpoints, sentence_distances,
                                           split_by_meaning, word_cosine)


def test_word_cosine():
    assert word_cosine("solar farm", "solar farm") == pytest.approx(1.0)
    assert word_cosine("solar farm", "grape harvest") == 0.0
    assert word_cosine("", "anything") == 0.0


def test_breakpoint_at_highest_distance():
    distances = [0.1, 0.2, 0.9, 0.1, 0.2]
    assert find_breakpoints(distances, 90) == [3]   # new segment starts at sentence 3


def test_equal_distances_give_no_breaks():
    assert find_breakpoints([1.0, 1.0, 1.0], 90) == []


def test_split_by_meaning_finds_topic_change():
    sentences = [
        "Solar panels make solar power.",
        "Solar power farms use many solar panels.",
        "Grapes grow in the valley.",
        "The valley farms harvest grapes.",
    ]
    assert split_by_meaning(sentences) == [sentences[:2], sentences[2:]]


def test_custom_embed_fn_is_used():
    vectors = {"a": [1, 0], "b": [1, 0], "c": [0, 1]}
    distances = sentence_distances(["a", "b", "c"], lambda texts: [vectors[t] for t in texts])
    assert distances == pytest.approx([0.0, 1.0])


def test_short_inputs():
    assert split_by_meaning([]) == []
    assert split_by_meaning(["One.", "Two."]) == [["One.", "Two."]]
