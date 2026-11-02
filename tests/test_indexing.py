"""Persistence, artifact alignment, rebuild safety, and ingestion round trips."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import faiss
import numpy as np

from spect_rag import SpectRAGIngestion, SpectRAGRetrieval
from spect_rag.utility.graph import build_chunk_graph
from spect_rag.utility.indexing import ARTIFACTS, load_index, save_index, tokenize
from spect_rag.utility.spectral import spectral_coordinates

from index_fixtures import chunk


class IndexingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.index_dir = self.root / "index"
        self.chunks = [
            chunk("a", [1, 0], doc="doc", text="Battery stores electricity.", questions=[[1, 0], [0, 1]]),
            chunk("b", [0, 1], doc="doc", position=1, text="Panels collect sunlight."),
        ]

    def save(self, records=None):
        records = self.chunks if records is None else records
        graph = build_chunk_graph(records)
        spectrum = spectral_coordinates(graph)
        manifest = save_index(
            self.index_dir, records, graph, spectrum,
            chunking_settings={"chunk_size": 1200, "chunk_overlap": 100,
                               "min_chunk_tokens": 32, "semantic_chunking": False},
        )
        return self.index_dir / "snapshots" / manifest["generation"]

    def update_hash(self, snapshot, name):
        path = snapshot / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["artifacts"][name] = sha256((snapshot / name).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest))

    def test_round_trip_preserves_order_matrices_faiss_and_bm25(self):
        snapshot = self.save()
        loaded = SpectRAGRetrieval(self.index_dir).load()
        self.assertEqual(loaded["snapshot_dir"], snapshot)
        self.assertEqual([c["chunk_id"] for c in loaded["chunks"]], ["a", "b"])
        np.testing.assert_array_equal(loaded["embeddings"], np.eye(2, dtype=np.float32))
        np.testing.assert_array_equal(loaded["question_owners"], [0, 0])
        self.assertEqual(loaded["question_index"].ntotal, 2)
        self.assertEqual(loaded["chunks"][1]["utility_question_embeddings"].shape, (0, 2))
        _, indices = loaded["dense_index"].search(np.asarray([[0, 1]], dtype=np.float32), 2)
        self.assertEqual(indices[0, 0], 1)
        self.assertEqual(loaded["lexical_tokens"], [tokenize(c["text"]) for c in self.chunks])
        self.assertEqual(loaded["bm25"].get_scores(tokenize("battery")).shape, (2,))
        self.assertEqual(set(loaded["manifest"]["artifacts"]), set(ARTIFACTS))

    def test_empty_question_index_and_punctuation_only_bm25(self):
        self.save([chunk("only", [1, 0], text="...")])
        loaded = load_index(self.index_dir)
        self.assertEqual(loaded["question_index"].ntotal, 0)
        self.assertEqual(loaded["question_embeddings"].shape, (0, 2))
        self.assertEqual(loaded["spectral_coords"].shape, (1, 0))
        self.assertIsNone(loaded["bm25"])

    def test_model_dimension_and_prefix_compatibility(self):
        self.save()
        load_index(self.index_dir, embedding_model="test-model", embedding_dimension=2)
        for options in [
            {"embedding_model": "other"}, {"embedding_dimension": 3}, {"document_prefix": ""},
        ]:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "does not match"):
                load_index(self.index_dir, **options)

    def test_missing_or_modified_artifacts_are_rejected(self):
        snapshot = self.save()
        path = snapshot / "question_owners.npy"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or mismatched"):
            load_index(self.index_dir)
        path.write_bytes(original + b"modified")
        with self.assertRaisesRegex(ValueError, "Missing or mismatched"):
            load_index(self.index_dir)

    def test_reordered_records_are_rejected_even_with_a_new_checksum(self):
        snapshot = self.save([chunk("a", [1, 0]), chunk("b", [0, 1])])
        path = snapshot / "chunks.json"
        records = json.loads(path.read_text())
        path.write_text(json.dumps(list(reversed(records))))
        self.update_hash(snapshot, "chunks.json")
        with self.assertRaisesRegex(ValueError, "chunk order"):
            load_index(self.index_dir)

    def test_graph_fusion_mismatch_is_rejected_even_with_a_new_checksum(self):
        snapshot = self.save()
        path = snapshot / "graph.json"
        graph = json.loads(path.read_text())
        graph["edges"][0]["weight"] += 0.5
        path.write_text(json.dumps(graph))
        self.update_hash(snapshot, "graph.json")
        with self.assertRaisesRegex(ValueError, "Graph weights"):
            load_index(self.index_dir)

    def test_spectral_mismatch_is_rejected_even_with_a_new_checksum(self):
        snapshot = self.save()
        path = snapshot / "spectral_coords.npy"
        coordinates = np.load(path, allow_pickle=False)
        coordinates[0, 0] += 0.25
        np.save(path, coordinates)
        self.update_hash(snapshot, "spectral_coords.npy")
        with self.assertRaisesRegex(ValueError, "Spectral"):
            load_index(self.index_dir)

    def test_faiss_row_mismatch_is_rejected_even_with_a_new_checksum(self):
        snapshot = self.save()
        swapped = faiss.IndexFlatIP(2)
        swapped.add(np.asarray([[0, 1], [1, 0]], dtype=np.float32))
        faiss.write_index(swapped, str(snapshot / "dense.faiss"))
        self.update_hash(snapshot, "dense.faiss")
        with self.assertRaisesRegex(ValueError, "FAISS vector order"):
            load_index(self.index_dir)

    def test_lexical_mismatch_is_rejected_even_with_a_new_checksum(self):
        snapshot = self.save()
        (snapshot / "lexical_tokens.json").write_text(json.dumps([["wrong"], ["wrong"]]))
        self.update_hash(snapshot, "lexical_tokens.json")
        with self.assertRaisesRegex(ValueError, "BM25 tokens"):
            load_index(self.index_dir)

    def test_unknown_version_and_unsafe_pointer_are_rejected(self):
        snapshot = self.save()
        path = snapshot / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["version"] = 2
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "version"):
            load_index(self.index_dir)
        (self.index_dir / "CURRENT").write_text("../../not-a-snapshot")
        with self.assertRaisesRegex(ValueError, "pointer"):
            load_index(self.index_dir)

    def test_failed_rebuild_leaves_previous_snapshot_readable(self):
        first = self.save()
        original_pointer = (self.index_dir / "CURRENT").read_text()
        with patch("spect_rag.utility.indexing.faiss.write_index", side_effect=RuntimeError("simulated disk failure")):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self.save()
        self.assertEqual((self.index_dir / "CURRENT").read_text(), original_pointer)
        self.assertEqual(load_index(self.index_dir)["snapshot_dir"], first)
        self.assertEqual(list((self.index_dir / "snapshots").glob(".building-*")), [])

    def test_successful_rebuild_publishes_new_snapshot_and_keeps_old(self):
        first = self.save()
        second = self.save([chunk("new", [1, 0])])
        self.assertNotEqual(first, second)
        self.assertTrue((first / "manifest.json").is_file())
        self.assertEqual(load_index(self.index_dir)["manifest"]["chunk_ids"], ["new"])

    def test_empty_index_and_unpublished_directory_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.save([])
        self.assertFalse(self.index_dir.exists())
        with self.assertRaisesRegex(ValueError, "No published index"):
            load_index(self.index_dir)

    def test_ingestion_wires_full_pipeline_and_separate_reload(self):
        source = self.root / "input.md"
        source.write_text("# Energy\n\nBattery stores electricity.\n\n# Storage\n\nSolar panels collect sunlight.")
        ingestion = SpectRAGIngestion(
            self.index_dir, embed_fn=lambda texts: [[3, 4] for _ in texts],
            embedding_model="test", embedding_dimension=2,
        )
        built = ingestion.ingest([source])
        reloaded = SpectRAGRetrieval(self.index_dir).load()
        self.assertEqual(built["manifest"], reloaded["manifest"])
        self.assertEqual(built["manifest"]["counts"]["chunks"], 2)
        self.assertEqual(built["manifest"]["counts"]["questions"], 0)
        self.assertEqual(len(built["graph"]), 2)
        self.assertEqual(built["spectral_coords"].shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
