import numpy as np
import pytest

from spectrafuse.config import make_settings
from spectrafuse.utils.graph import neighbors
from spectrafuse.utils.selectors import SELECTORS, select


def make_case(tokens, scores, vectors=None, clusters=None, docs=None):
    n = len(tokens)
    vectors = np.eye(n) if vectors is None else np.array(vectors, dtype=float)
    index = {"embeddings": vectors,
             "chunks": [{"tokens": t, "cluster": (clusters or [0] * n)[i],
                         "doc_id": (docs or ["d%d" % i for i in range(n)])[i]}
                        for i, t in enumerate(tokens)]}
    scored = {"candidates": list(range(n)), "scores": np.array(scores, dtype=float),
              "proximity": np.zeros(n)}
    return index, scored


def settings(**overrides):
    return make_settings(dict({"token_budget": 30, "core_size": 1}, **overrides))


def ids(picks):
    return [int(k) for k, _ in picks]


@pytest.mark.parametrize("name", SELECTORS)
def test_every_selector_respects_the_budget(name):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(25, 4))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index, scored = make_case(rng.integers(5, 60, 25).tolist(), rng.random(25).tolist(), vectors)
    graph = neighbors([[i, i + 1, 0.5] for i in range(24)], 25)
    picks = select(scored, index, settings(selector=name, token_budget=150), graph)
    assert sum(index["chunks"][k]["tokens"] for k in ids(picks)) <= 150
    assert len(set(ids(picks))) == len(picks)


def test_mmr_skips_near_duplicates():
    vectors = [[1, 0], [0.99, 0.14], [0, 1]]                        # 1 is a near-copy of 0
    index, scored = make_case([10, 10, 10], [1.0, 0.9, 0.6], vectors)
    assert ids(select(scored, index, settings(selector="mmr", token_budget=20))) == [0, 2]
    assert ids(select(scored, index, settings(selector="mmr", mmr_lambda=1.0,
                                              token_budget=20))) == [0, 1]   # lambda 1 = top-k


def test_protected_mmr_keeps_the_core():
    vectors = [[1, 0], [0.99, 0.14], [0, 1]]
    index, scored = make_case([10, 10, 10], [1.0, 0.9, 0.6], vectors)
    picks = select(scored, index, settings(selector="protected_mmr", core_size=2, token_budget=20))
    assert picks == [(0, "core"), (1, "core")]


def test_anchor_expand_prefers_graph_neighbours_of_anchors():
    index, scored = make_case([10] * 4, [1.0, 0.9, 0.8, 0.2])
    graph = neighbors([[0, 3, 1.0]], 4)                             # chunk 3 is linked to anchor 0
    picks = select(scored, index, settings(selector="anchor_expand_ce", token_budget=20), graph)
    assert picks == [(0, "anchor"), (3, "expand")]
    rerank = [0.0, 0.0, 1.0, 0.5]                                    # a cross-encoder's opinion
    picks = select(scored, index, settings(selector="anchor_expand_ce", token_budget=40), graph, rerank)
    assert picks == [(0, "anchor"), (3, "expand"), (2, "backfill"), (1, "backfill")]


def test_topk_is_score_order():
    index, scored = make_case([10] * 4, [0.2, 0.9, 0.5, 0.7])
    assert ids(select(scored, index, settings(selector="topk", token_budget=30))) == [1, 3, 2]


def test_unknown_selector_raises():
    index, scored = make_case([10], [1.0])
    with pytest.raises(ValueError, match="Unknown selector"):
        select(scored, index, dict(settings(), selector="best"))
