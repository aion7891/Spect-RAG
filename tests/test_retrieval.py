"""Persisted-index retrieval integration with deterministic query embeddings."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from spect_rag import SpectRAGRetrieval
from spect_rag.utility.graph import DEFAULT_EDGE_WEIGHTS, build_chunk_graph
from spect_rag.utility.indexing import load_index, save_index
from spect_rag.utility.ranking import initial_retrieval
from spect_rag.utility.spectral import spectral_coordinates

from index_fixtures import chunk


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.index_dir = self.root / "index"
        self.calls = []
        self.records = [
            chunk("a", [1, 0], doc="doc", position=0, text="Alpha evidence."),
            chunk("b", [0, 1], doc="doc", position=1, text="Bridge evidence."),
            chunk("c", [-1, 0], doc="doc", position=2, text="Conclusion evidence."),
        ]

    def save(self, records=None):
        records = self.records if records is None else records
        graph = build_chunk_graph(records, edge_weights={
            name: float(name == "adjacency") for name in DEFAULT_EDGE_WEIGHTS
        })
        save_index(
            self.index_dir, records, graph, spectral_coordinates(graph),
            chunking_settings={"chunk_size": 1200, "chunk_overlap": 100,
                               "min_chunk_tokens": 32, "semantic_chunking": False},
        )

    def embed(self, texts):
        self.calls.extend(texts)
        return [[1, 0] for _ in texts]

    def retriever(self, **options):
        return SpectRAGRetrieval(
            self.index_dir, embed_fn=self.embed, embedding_model="test-model", **options,
        )

    def test_two_hop_bridge_recovery_and_inspectable_scoring(self):
        self.save()
        retrieval = self.retriever()
        evidence = retrieval.retrieve("unseenquery", seed_k=1, max_candidates=3, core_size=1, precision_floor=0.1)
        trace = retrieval.last_trace
        self.assertEqual(trace["seed_chunk_ids"], ["a"])
        self.assertEqual(trace["candidate_chunk_ids"], ["a", "b", "c"])
        details = {item["chunk_id"]: item for item in trace["candidate_scores"]}
        self.assertEqual(details["c"]["hop"], 2)
        self.assertEqual(details["b"]["relevance_score"], 0)
        self.assertGreater(details["b"]["score"], 0)
        self.assertGreater(details["b"]["ppr_raw"], details["c"]["ppr_raw"])
        self.assertEqual({item["chunk_id"] for item in evidence}, {"a", "b", "c"})
        self.assertEqual(self.calls, ["search_query: unseenquery"])
        self.assertEqual(trace["used_tokens"], sum(item["token_count"] for item in evidence))
        json.dumps(trace, allow_nan=False)
        json.dumps(evidence, allow_nan=False)

    def test_disabling_proximity_excludes_zero_relevance_bridges(self):
        self.save()
        evidence = self.retriever().retrieve("unseenquery", seed_k=1, alpha=0)
        self.assertEqual([item["chunk_id"] for item in evidence], ["a"])

    def test_hard_budget_and_no_partial_chunks(self):
        self.save()
        retrieval = self.retriever()
        budget = self.records[0]["token_count"]
        evidence = retrieval.retrieve("unseenquery", seed_k=1, token_budget=budget)
        self.assertEqual([item["chunk_id"] for item in evidence], ["a"])
        self.assertEqual(evidence[0]["text"], self.records[0]["text"])
        self.assertLessEqual(sum(item["token_count"] for item in evidence), budget)
        self.assertEqual(retrieval.retrieve("unseenquery", seed_k=1, token_budget=1), [])
        self.assertEqual(retrieval.last_trace["status"], "no_context_selected")

    def test_no_positive_signal_returns_no_arbitrary_seeds(self):
        self.save([chunk("negative", [-1, 0]), chunk("orthogonal", [0, -1])])
        retrieval = self.retriever()
        self.assertEqual(retrieval.retrieve("unseenquery"), [])
        self.assertEqual(retrieval.last_trace["status"], "no_positive_seed")
        self.assertEqual(retrieval.last_trace["active_channels"], [])

    def test_zero_budget_does_not_load_index_or_call_a_model(self):
        retrieval = SpectRAGRetrieval(self.index_dir)
        self.assertEqual(retrieval.retrieve("valid query", token_budget=0), [])
        self.assertEqual(retrieval.last_trace["status"], "empty_budget")
        self.assertFalse(self.index_dir.exists())

    def test_isolated_chunk_without_spectral_modes_uses_ppr_only(self):
        self.save([chunk("a", [1, 0])])
        evidence = self.retriever().retrieve("unseenquery")
        self.assertEqual(len(evidence), 1)
        self.assertFalse(evidence[0]["spectral_available"])
        self.assertEqual(evidence[0]["proximity_score"], 1)
        self.assertEqual(evidence[0]["spectral_score"], 0)

    def test_question_ownership_uses_max_cosine_and_missing_questions_stay_absent(self):
        records = [
            chunk("a", [1, 0]),
            chunk("b", [0, 1], questions=[[0, 1], [0.6, 0.8]]),
        ]
        self.save(records)
        index = load_index(self.index_dir)
        initial = initial_retrieval(index, "unseenquery", np.asarray([1, 0], dtype=np.float32))
        np.testing.assert_allclose(initial["raw"]["utility"], [0, 0.6], atol=1e-6)
        self.assertEqual(initial["rankings"]["utility"], [1])
        self.assertEqual(initial["rankings"]["bm25"], [])

    def test_tiny_corpus_negative_bm25_is_clipped_and_retained_in_raw_trace(self):
        self.save([chunk("a", [1, 0], text="Common token."), chunk("b", [0, 1], text="Common token.")])
        retrieval = self.retriever()
        retrieval.retrieve("common")
        trace = retrieval.last_trace
        self.assertNotIn("bm25", trace["active_channels"])
        self.assertTrue(all(item["bm25_raw"] < 0 for item in trace["candidate_scores"]))

    def test_query_cache_and_model_dimension_compatibility(self):
        self.save()
        retrieval = self.retriever(cache_dir=self.root / "cache")
        first = retrieval.retrieve("unseenquery", seed_k=1)
        second = retrieval.retrieve("unseenquery", seed_k=1)
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)
        wrong = SpectRAGRetrieval(self.index_dir, embed_fn=self.embed, embedding_model="other")
        with self.assertRaisesRegex(ValueError, "does not match"):
            wrong.retrieve("unseenquery")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.retriever(embedding_dimension=3).retrieve("unseenquery")
        self.assertEqual(len(self.calls), 1)

    def test_reload_refreshes_snapshot_and_clears_old_trace(self):
        self.save()
        retrieval = self.retriever()
        old = retrieval.retrieve("unseenquery", seed_k=1)
        generation = retrieval.last_trace["snapshot_generation"]
        self.save([chunk("fresh", [1, 0])])
        self.assertEqual(retrieval.retrieve("unseenquery", seed_k=1), old)
        self.assertEqual(retrieval.last_trace["snapshot_generation"], generation)
        retrieval.load()
        self.assertIsNone(retrieval.last_trace)
        self.assertEqual(retrieval.retrieve("unseenquery")[0]["chunk_id"], "fresh")
        self.assertNotEqual(retrieval.last_trace["snapshot_generation"], generation)

    def test_invalid_query_or_provider_does_not_leave_a_stale_trace(self):
        self.save()
        retrieval = self.retriever()
        retrieval.retrieve("unseenquery")
        for query in ("", "  ", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                retrieval.retrieve(query)
            self.assertIsNone(retrieval.last_trace)
        with self.assertRaisesRegex(ValueError, "embed_fn"):
            SpectRAGRetrieval(self.index_dir).retrieve("valid query")
        broken = SpectRAGRetrieval(
            self.index_dir, embed_fn=lambda _: [[0, 0]], embedding_model="test-model",
        )
        with self.assertRaisesRegex(ValueError, "zero"):
            broken.retrieve("valid query")
        self.assertIsNone(broken.last_trace)


if __name__ == "__main__":
    unittest.main()
