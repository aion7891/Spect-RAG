"""Hand-computed edge fusion and normalized-Laplacian checks."""

import unittest

import networkx as nx
import numpy as np

from spect_rag.utility.graph import DEFAULT_EDGE_WEIGHTS, build_chunk_graph
from spect_rag.utility.spectral import normalized_laplacian, spectral_coordinates

from index_fixtures import chunk


class GraphSpectralTests(unittest.TestCase):
    def test_six_signals_and_fusion_are_hand_computable(self):
        left = chunk("a", [1, 0], doc="guide", section=["Guide", "Intro"],
                     concepts=["Battery", "Solar"], questions=[[1, 0]])
        right = chunk("b", [0.6, 0.8], doc="guide", position=1, section=["Guide", "Next"],
                      concepts=["battery", "Wind"], questions=[[1, 0]])
        left["citations"] = [{
            "status": "resolved", "target_doc_id": "guide", "target_chunk_ids": ["b"],
        }]
        graph = build_chunk_graph([left, right], graph_neighbors=1, relation_neighbors=1, edge_threshold=0)
        self.assertEqual(graph.number_of_edges(), 1)
        expected = {"semantic": 0.6, "adjacency": 1, "section": 0.5,
                    "entity": 1 / 3, "utility": 1, "citation": 1}
        data = graph.edges["a", "b"]
        for name, value in expected.items():
            self.assertAlmostEqual(data[name], value, places=6)
        self.assertAlmostEqual(data["weight"], sum(DEFAULT_EDGE_WEIGHTS[k] * v for k, v in expected.items()))

    def test_section_paths_do_not_connect_unrelated_documents(self):
        records = [
            chunk("a", [1, 0], section=["Same"]),
            chunk("b", [0, 1], section=["Same"]),
        ]
        graph = build_chunk_graph(records)
        self.assertEqual(list(graph), ["a", "b"])
        self.assertEqual(graph.number_of_edges(), 0)
        self.assertEqual(graph.graph["stats"]["isolated_nodes"], 2)

    def test_utility_uses_max_pairwise_question_similarity_without_fallback(self):
        records = [
            chunk("a", [1, 0], questions=[[1, 0], [0, 1]]),
            chunk("b", [0, 1], questions=[[0.6, 0.8]]),
            chunk("missing-questions", [1, 0]),
        ]
        weights = {name: float(name == "utility") for name in DEFAULT_EDGE_WEIGHTS}
        graph = build_chunk_graph(records, edge_weights=weights, edge_threshold=0)
        self.assertAlmostEqual(graph.edges["a", "b"]["utility"], 0.8, places=6)
        self.assertEqual(graph.degree("missing-questions"), 0)

    def test_citation_neighbor_cap_keeps_stable_targets_without_self_links(self):
        records = [
            chunk("a", [1, 0]),
            chunk("b", [0, 1], doc="target"),
            chunk("c", [0, 1], doc="target", position=1),
        ]
        records[0]["citations"] = [{
            "status": "resolved", "target_doc_id": "target", "target_chunk_ids": ["b", "c"],
        }]
        graph = build_chunk_graph(records, graph_neighbors=0, relation_neighbors=1)
        self.assertEqual(graph.edges["a", "b"]["citation"], 1)
        self.assertFalse(graph.has_edge("a", "c"))

    def test_threshold_and_zero_neighbor_limits_preserve_nodes(self):
        records = [chunk("a", [1, 0], doc="doc"), chunk("b", [1, 0], doc="doc", position=1)]
        graph = build_chunk_graph(records, graph_neighbors=0, relation_neighbors=0)
        self.assertAlmostEqual(graph.edges["a", "b"]["weight"], 0.084)
        sparse_graph = build_chunk_graph(records, edge_threshold=2)
        self.assertEqual(sparse_graph.number_of_nodes(), 2)
        self.assertEqual(sparse_graph.number_of_edges(), 0)

    def test_candidate_edge_storage_is_bounded_and_ties_are_stable(self):
        records = [
            chunk(str(i), [1, 0], doc="doc", position=i, section=["Shared"],
                  concepts=["Shared"], questions=[[1, 0]])
            for i in range(40)
        ]
        graph = build_chunk_graph(records, graph_neighbors=2, relation_neighbors=2)
        # At most n*k directed choices per bounded signal, plus n-1 adjacency links.
        self.assertLessEqual(graph.graph["stats"]["candidate_edges"], 40 * (2 * 2 + 3 * 2) + 39)
        again = build_chunk_graph(records, graph_neighbors=2, relation_neighbors=2)
        self.assertEqual(list(graph.edges(data=True)), list(again.edges(data=True)))

    def test_empty_graph_and_invalid_settings(self):
        self.assertEqual(build_chunk_graph([]).number_of_nodes(), 0)
        for settings in [
            {"graph_neighbors": -1}, {"relation_neighbors": True}, {"edge_threshold": float("nan")},
            {"edge_weights": {"invented": 1}},
            {"edge_weights": {name: 0 for name in DEFAULT_EDGE_WEIGHTS}},
        ]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                build_chunk_graph([], **settings)

    def test_record_inconsistencies_are_rejected(self):
        for mutation in ("duplicate", "position", "dimension", "normalization", "citation"):
            records = [chunk("a", [1, 0]), chunk("b", [0, 1])]
            if mutation == "duplicate":
                records[1]["chunk_id"] = "a"
            elif mutation == "position":
                records[1]["position"] = 2
            elif mutation == "dimension":
                records[1]["embedding_dimension"] = 3
            elif mutation == "normalization":
                records[1]["embedding"] = np.asarray([3, 4])
            else:
                records[0]["citations"] = [{
                    "status": "resolved", "target_doc_id": "missing", "target_chunk_ids": ["missing"],
                }]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_chunk_graph(records)

    def test_empty_singleton_and_edgeless_spectra(self):
        for size in (0, 1, 5):
            with self.subTest(size=size):
                graph = nx.empty_graph(size)
                spectrum = spectral_coordinates(graph)
                self.assertEqual(spectrum["coordinates"].shape, (size, 0))
                self.assertEqual(len(spectrum["eigenvalues"]), 0)
                self.assertEqual(spectrum["zero_modes"], size)
                self.assertEqual(normalized_laplacian(graph).nnz, 0)

    def test_disconnected_graph_skips_every_zero_mode_and_keeps_isolate(self):
        graph = nx.path_graph(3)
        graph.add_node("isolated")
        result = spectral_coordinates(graph)
        np.testing.assert_allclose(result["eigenvalues"], [1, 2], atol=1e-10)
        self.assertEqual(result["zero_modes"], 2)
        self.assertEqual(result["coordinates"].shape, (4, 2))
        np.testing.assert_array_equal(result["coordinates"][-1], [0, 0])
        np.testing.assert_allclose(result["coordinates"].T @ result["coordinates"], np.eye(2), atol=1e-10)
        self.assertLess(np.max(result["residuals"]), 1e-10)

    def test_multiple_nontrivial_components_and_dimension_cap(self):
        graph = nx.Graph([(0, 1), (2, 3)])
        full = spectral_coordinates(graph)
        np.testing.assert_allclose(full["eigenvalues"], [2, 2], atol=1e-10)
        self.assertEqual(full["zero_modes"], 2)
        self.assertEqual(spectral_coordinates(graph, dimension=1)["coordinates"].shape, (4, 1))
        self.assertEqual(spectral_coordinates(graph, dimension=0)["coordinates"].shape, (4, 0))

    def test_sparse_solver_matches_known_path_eigenvalues(self):
        size = 270
        result = spectral_coordinates(nx.path_graph(size), dimension=4)
        expected = 1 - np.cos(np.arange(1, 5) * np.pi / (size - 1))
        np.testing.assert_allclose(result["eigenvalues"], expected, atol=1e-8)
        self.assertLess(np.max(result["residuals"]), 1e-6)

    def test_invalid_spectral_graphs_and_dimension(self):
        invalid = [nx.DiGraph([(0, 1)]), nx.Graph([(0, 0)]), nx.Graph([(0, 1, {"weight": -1})])]
        for graph in invalid:
            with self.assertRaises(ValueError):
                spectral_coordinates(graph)
        with self.assertRaises(ValueError):
            spectral_coordinates(nx.Graph(), dimension=-1)


if __name__ == "__main__":
    unittest.main()
