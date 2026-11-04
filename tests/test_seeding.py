from pathlib import Path

import numpy as np
import pytest

from spectrafuse import SpectRAGIngestion, make_settings
from spectrafuse.utils.bm25 import build_bm25
from spectrafuse.utils.embedder import make_embedder
from spectrafuse.utils.seeding import (dense_scores, find_seeds, hybrid_relevance, min_max, ranking,
                                    rrf_fuse, top_seeds, utility_scores)
from spectrafuse.utils.store import load_index

EXAMPLES = Path(__file__).parent.parent / "examples" / "docs"


def test_dense_scores_are_cosines():
    embeddings = np.array([[1, 0], [0, 1], [0.6, 0.8]])
    assert np.allclose(dense_scores(embeddings, np.array([1, 0])), [1, 0, 0.6])


def test_utility_scores_take_best_question_per_chunk():
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8]])
    scores = utility_scores(vectors, owners=[0, 0, 2], query_vector=np.array([0, 1]), chunk_count=4)
    assert np.allclose(scores, [1.0, 0.0, 0.8, 0.0])   # chunk 0 keeps its best question
    assert not utility_scores(np.zeros((0, 0)), [], np.array([1, 0]), 3).any()


def test_ranking_is_stable_and_can_filter():
    assert ranking([0.2, 0.9, 0.2, 0.5]) == [1, 3, 0, 2]     # tie 0.2: chunk 0 before chunk 2
    assert ranking([0.0, 3.0, 1.0], keep=[False, True, True]) == [1, 2]


def test_rrf_formula_and_top_seeds():
    fused = rrf_fuse([[1, 2, 3], [2, 1]], k=60)
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63)
    assert fused[1] == pytest.approx(fused[2])               # mirror ranks, same score
    assert top_seeds(fused, 2) == [1, 2]                     # tie: lower id first


def test_min_max_and_hybrid_relevance():
    assert np.allclose(min_max([2, 4, 6]), [0, 0.5, 1])
    assert not min_max([3, 3]).any()
    relevance = hybrid_relevance([0.1, 0.9], [5.0, 0.0], bm25_weight=0.5)
    assert np.allclose(relevance, [0.5, 0.5])


@pytest.fixture(scope="module")
def sample_index(tmp_path_factory):
    folder = tmp_path_factory.mktemp("idx")
    SpectRAGIngestion(folder).ingest(EXAMPLES)
    return load_index(folder)


def run(index, question, **overrides):
    settings = make_settings(dict(index["settings"], **overrides))
    embed = make_embedder(settings)
    bm25 = build_bm25([chunk["text"] for chunk in index["chunks"]])
    return find_seeds(question, embed([question], kind="query")[0], index, bm25, settings)


def test_find_seeds_on_sample_docs(sample_index):
    result = run(sample_index, "Which company did the co-founder of Aurora Labs join?", seed_count=3)
    assert len(result["seeds"]) == 3
    assert result["seeds"][0] == 0                           # Aurora history chunk
    assert len(result["rankings"]) == 2                      # no utility questions -> 2 lists
    assert result["relevance"].shape == (len(sample_index["chunks"]),)
    assert result["relevance"].min() >= 0 and result["relevance"].max() == pytest.approx(1)


def test_seed_count_is_capped_by_corpus(sample_index):
    assert len(run(sample_index, "water", seed_count=20)["seeds"]) == len(sample_index["chunks"])


def test_utility_ranking_is_used_when_questions_exist(tmp_path):
    questions = {"Members": ["How many farms are in the cooperative?"]}

    def question_fn(text):
        return next((q for key, q in questions.items() if key in text), [])

    SpectRAGIngestion(tmp_path, question_fn=question_fn).ingest(EXAMPLES)
    index = load_index(tmp_path)
    result = run(index, "How many farms are in the cooperative?", seed_count=3)
    assert len(result["rankings"]) == 3
    owner = index["question_owners"][0]
    assert result["rankings"][2] == [owner]                  # only chunks with questions
    assert result["utility"][owner] == pytest.approx(1.0, abs=1e-5)
    assert owner in result["seeds"]
