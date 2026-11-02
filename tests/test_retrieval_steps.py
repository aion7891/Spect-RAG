"""Hand-check the online algorithm without a model service."""

import unittest

import networkx as nx
import numpy as np

from spect_rag.utility.expansion import expand_candidates
from spect_rag.utility.proximity import score_candidates, spectral_support
from spect_rag.utility.ranking import fuse_rankings, normalize_positive, retrieval_settings
from spect_rag.utility.retrieval_trace import evidence_records
from spect_rag.utility.selection import select_context
from spect_rag.utility.spectral import spectral_coordinates


class RetrievalStepTests(unittest.TestCase):
    def test_positive_normalization_does_not_manufacture_zero_signal(self):
        np.testing.assert_array_equal(normalize_positive([0, 0]), [0, 0])
        np.testing.assert_array_equal(normalize_positive([-2, -1]), [0, 0])
        np.testing.assert_array_equal(normalize_positive([2, 2]), [1, 1])
        np.testing.assert_array_equal(normalize_positive([1, 2]), [0.5, 1])
        with self.assertRaises(ValueError):
            normalize_positive([float("nan")])

    def test_rrf_is_hand_computable_and_missing_entries_do_not_vote(self):
        fused, rankings = fuse_rankings({
            "dense": np.asarray([1.0, 0.5]), "bm25": np.asarray([0.0, 0.0]),
            "utility": np.asarray([0.0, 1.0]),
        }, rrf_k=60)
        np.testing.assert_allclose(fused, [1 / 61, 1 / 62 + 1 / 61])
        self.assertEqual(rankings, {"dense": [0, 1], "bm25": [], "utility": [1]})

    def test_rrf_ties_use_corpus_order(self):
        fused, rankings = fuse_rankings({"dense": np.asarray([1.0, 1.0, 1.0])}, rrf_k=0)
        np.testing.assert_allclose(fused, [1, 0.5, 1 / 3])
        self.assertEqual(rankings["dense"], [0, 1, 2])

    def test_expansion_stops_at_two_hops(self):
        graph = nx.path_graph(["a", "b", "c", "d", "e"])
        nx.set_edge_attributes(graph, 1.0, "weight")
        candidates, subgraph, hops = expand_candidates(graph, ["a"], dict.fromkeys(graph, 0.0))
        self.assertEqual(candidates, ["a", "b", "c"])
        self.assertEqual(hops, {"a": 0, "b": 1, "c": 2})
        self.assertEqual(set(subgraph), {"a", "b", "c"})

    def test_expansion_cap_keeps_seeds_and_prioritizes_relevance(self):
        graph = nx.star_graph(1000)
        nx.set_edge_attributes(graph, 1.0, "weight")
        relevance = dict.fromkeys(graph, 0.0)
        relevance[1000] = 1
        candidates, _, _ = expand_candidates(graph, [0], relevance, max_candidates=4)
        self.assertEqual(candidates, [0, 1000, 1, 2])

    def test_heap_updates_keep_strongest_frontier_edge(self):
        graph = nx.Graph()
        graph.add_nodes_from(["s1", "s2", "x", "y", "z"])
        graph.add_weighted_edges_from([
            ("s1", "x", 0.1), ("s1", "y", 0.5), ("s1", "z", 0.4),
            ("s2", "x", 0.9), ("s2", "y", 0.8),
        ])
        candidates, _, _ = expand_candidates(
            graph, ["s1", "s2"], dict.fromkeys(graph, 1.0), max_candidates=4,
        )
        self.assertEqual(candidates, ["s1", "s2", "x", "y"])

    def test_invalid_expansion_seeds_are_rejected(self):
        graph = nx.empty_graph(["a"])
        for seeds, cap in [(["a", "a"], 2), (["missing"], 1), (["a"], 0)]:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                expand_candidates(graph, seeds, {"a": 1}, max_candidates=cap)

    def test_ppr_matches_hand_solved_three_node_path(self):
        graph = nx.path_graph(["a", "b", "c"])
        nx.set_edge_attributes(graph, 1.0, "weight")
        scores, weights = score_candidates(
            graph, ["a"], list(graph), {"a": 1, "b": 0, "c": 0},
            {node: np.empty(0) for node in graph}, dict.fromkeys(graph), alpha=1,
        )
        middle = 0.85 / 1.85
        expected = {"a": 0.15 + 0.85 * middle / 2, "b": middle, "c": 0.85 * middle / 2}
        self.assertEqual(weights, {"a": 1})
        for node in graph:
            self.assertAlmostEqual(scores[node]["ppr_raw"], expected[node], places=8)
            self.assertFalse(scores[node]["spectral_available"])
            self.assertEqual(scores[node]["proximity_score"], scores[node]["ppr_score"])

    def test_spectral_kernel_and_relevance_blend_are_hand_computable(self):
        graph = nx.Graph()
        graph.add_edge("a", "b", weight=1)
        scores, _ = score_candidates(
            graph, ["a"], ["a", "b"], {"a": 0.5, "b": 0},
            {"a": np.asarray([1 / np.sqrt(2)]), "b": np.asarray([-1 / np.sqrt(2)])},
            {"a": 0, "b": 0}, alpha=0.4,
        )
        self.assertEqual(scores["a"]["spectral_raw"], 1)
        self.assertEqual(scores["a"]["proximity_score"], 1)
        self.assertAlmostEqual(scores["a"]["score"], 0.6 * 0.5 + 0.4)
        self.assertAlmostEqual(scores["b"]["spectral_raw"], np.exp(-2))
        self.assertAlmostEqual(scores["b"]["proximity_score"], 0.5 * (np.exp(-2) + 0.85), places=8)

    def test_zero_coordinate_inside_supported_component_is_not_an_isolate(self):
        graph = nx.path_graph(["a", "b", "c"])
        coordinates = spectral_coordinates(graph, dimension=1)["coordinates"]
        support = spectral_support(graph, coordinates)
        self.assertIsNotNone(support["b"])
        graph.add_node("isolate")
        coordinates = np.vstack([coordinates, np.zeros((1, 1))])
        support = spectral_support(graph, coordinates)
        self.assertIsNone(support["isolate"])

    def test_components_missing_from_spectral_basis_do_not_gain_fake_closeness(self):
        graph = nx.Graph([("a", "b"), ("c", "d")])
        coordinates = np.asarray([[1.0], [-1.0], [0.0], [0.0]])
        support = spectral_support(graph, coordinates)
        self.assertIsNone(support["c"])
        scores, _ = score_candidates(
            graph, ["c"], list(graph), {"a": 0, "b": 0, "c": 1, "d": 0},
            dict(zip(graph, coordinates)), support,
        )
        self.assertTrue(all(not value["spectral_available"] for value in scores.values()))
        self.assertEqual(scores["a"]["spectral_raw"], 0)
        self.assertEqual(scores["a"]["ppr_raw"], 0)

    def test_zero_relevance_question_seeds_get_uniform_personalization(self):
        graph = nx.empty_graph(["a", "b"])
        scores, weights = score_candidates(
            graph, ["a", "b"], ["a", "b"], {"a": 0, "b": 0},
            {"a": np.empty(0), "b": np.empty(0)}, {"a": None, "b": None},
        )
        self.assertEqual(weights, {"a": 0.5, "b": 0.5})
        self.assertTrue(all(value["score"] > 0 for value in scores.values()))

    def test_coverage_is_only_a_relevance_tie_break(self):
        records = [
            {"chunk_id": "a", "doc_id": "one", "token_count": 10},
            {"chunk_id": "b", "doc_id": "one", "token_count": 10},
            {"chunk_id": "c", "doc_id": "two", "token_count": 10},
        ]
        settings = retrieval_settings(token_budget=20, core_size=1)
        scores = {node: {"score": value} for node, value in zip("abc", [0.9, 0.8, 0.8])}
        self.assertEqual(select_context(records, scores, settings)["selected"], ["a", "c"])
        scores["b"]["score"] = 0.81
        self.assertEqual(select_context(records, scores, settings)["selected"], ["a", "b"])

    def test_core_floor_and_backfill_policies_are_separate(self):
        records = [{"chunk_id": node, "doc_id": node, "token_count": 10} for node in "ab"]
        scores = {"a": {"score": 0.7}, "b": {"score": 0.6}}
        strict = retrieval_settings(token_budget=20, core_size=0, precision_floor=0.8, backfill=False)
        self.assertEqual(select_context(records, scores, strict)["selected"], [])
        protected = select_context(records, scores, {**strict, "core_size": 1})
        self.assertEqual(protected["reasons"], {"a": "core"})
        backfilled = select_context(records, scores, {**strict, "backfill": True})
        self.assertEqual(backfilled["reasons"], {"a": "backfill", "b": "backfill"})

    def test_oversized_highest_chunk_is_skipped_without_starving_budget(self):
        records = [
            {"chunk_id": "a", "doc_id": "one", "token_count": 30},
            {"chunk_id": "b", "doc_id": "one", "token_count": 10},
            {"chunk_id": "c", "doc_id": "two", "token_count": 10},
        ]
        scores = {node: {"score": value} for node, value in zip("abc", [0.9, 0.8, 0.7])}
        selection = select_context(records, scores, retrieval_settings(token_budget=20, core_size=1))
        self.assertEqual(selection["selected"], ["b", "c"])
        self.assertEqual(selection["used_tokens"], 20)
        self.assertEqual(selection["excluded"]["a"], "over_budget")

    def test_zero_scores_are_excluded_even_with_protected_core_and_backfill(self):
        records = [{"chunk_id": "a", "doc_id": "one", "token_count": 1}]
        selection = select_context(records, {"a": {"score": 0}}, retrieval_settings(precision_floor=0))
        self.assertEqual(selection["selected"], [])
        self.assertEqual(selection["excluded"], {"a": "zero_score"})

    def test_result_rank_and_selection_order_are_distinct_and_scores_are_canonical(self):
        records = [
            {"chunk_id": "a", "doc_id": "one", "token_count": 10, "score": 999},
            {"chunk_id": "b", "doc_id": "one", "token_count": 10},
            {"chunk_id": "c", "doc_id": "two", "token_count": 10},
        ]
        scores = {node: {"score": value} for node, value in zip("abc", [0.9, 0.8, 0.8])}
        selection = select_context(records, scores, retrieval_settings(token_budget=30, core_size=1))
        evidence = evidence_records(records, scores, selection)
        self.assertEqual(selection["selection_order"], ["a", "c", "b"])
        self.assertEqual([item["chunk_id"] for item in evidence], ["a", "b", "c"])
        self.assertEqual([item["selection_order"] for item in evidence], [1, 3, 2])
        self.assertEqual(evidence[0]["score"], 0.9)

    def test_invalid_retrieval_options(self):
        for options in [
            {"token_budget": -1}, {"seed_k": 0}, {"seed_k": 21, "max_candidates": 20},
            {"core_size": True}, {"rrf_k": -1}, {"alpha": float("nan")},
            {"precision_floor": 1.1}, {"backfill": "yes"},
        ]:
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                retrieval_settings(**options)


if __name__ == "__main__":
    unittest.main()
