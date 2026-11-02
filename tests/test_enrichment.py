"""Behavioral tests for model boundaries, caching, semantics, and citations."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from spect_rag import SpectRAGIngestion
from spect_rag.utility.citations import resolve_citations
from spect_rag.utility.embeddings import embed_texts
from spect_rag.utility.extraction import extract_metadata


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.cache = self.root / "cache"
        self.calls = []

    def source(self, text, name="source.txt"):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def embed(self, texts):
        self.calls.append(texts)
        return [[3.0, 4.0] for _ in texts]

    def ingestion(self, **kwargs):
        return SpectRAGIngestion(
            embed_fn=self.embed, embedding_model="test-model-v1",
            embedding_dimension=2, **kwargs,
        )

    def test_embedding_prefix_normalization_and_duplicate_order(self):
        matrix = embed_texts(["first", "second", "first"], self.embed, model_id="test", dimension=2)
        self.assertEqual(self.calls, [["search_document: first", "search_document: second"]])
        np.testing.assert_allclose(matrix, [[0.6, 0.8]] * 3)
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1)
        self.assertEqual(matrix.dtype, np.float32)
        self.assertTrue(matrix.flags.c_contiguous)

    def test_batched_embeddings_and_query_prefix(self):
        embed_texts(["one", "two", "three"], self.embed, model_id="test",
                    dimension=2, prefix="search_query: ", batch_size=2)
        self.assertEqual(self.calls, [["search_query: one", "search_query: two"], ["search_query: three"]])

    def test_missing_only_cache_reuse_and_model_prefix_invalidation(self):
        options = dict(model_id="v1", dimension=2, cache_dir=self.cache)
        first = embed_texts(["one"], self.embed, **options)
        second = embed_texts(["one", "two"], self.embed, **options)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[-1], ["search_document: two"])
        embed_texts(["one"], self.embed, **{**options, "model_id": "v2"})
        embed_texts(["one"], self.embed, **{**options, "prefix": ""})
        self.assertEqual(len(self.calls), 4)

    def test_embedding_dimension_is_part_of_cache_identity(self):
        embed_texts(["one"], self.embed, model_id="v1", dimension=2, cache_dir=self.cache)
        with self.assertRaisesRegex(ValueError, "shape"):
            embed_texts(["one"], self.embed, model_id="v1", dimension=3, cache_dir=self.cache)
        self.assertEqual(len(self.calls), 2)

    def test_invalid_vectors_do_not_enter_cache(self):
        invalid = [
            [[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]],
            [[1, 2, 3]], [[1, 2], [3, 4]], [1, 2], [["1", "2"]],
            [[1 + 1j, 2]], [[True, False]],
        ]
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(ValueError):
                embed_texts(["one"], lambda _: output, model_id="test",
                            dimension=2, cache_dir=self.cache)
        self.assertEqual(list(self.cache.rglob("*.json")), [])

    def test_large_finite_vectors_and_empty_question_matrix(self):
        result = embed_texts(["huge"], lambda _: [[1e308, 1e308]], model_id="test", dimension=2)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1)
        empty = embed_texts([], self.embed, model_id="test", dimension=2)
        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual(self.calls, [])

    def test_corrupt_cache_fails_explicitly(self):
        embed_texts(["one"], self.embed, model_id="test", dimension=2, cache_dir=self.cache)
        path = next(self.cache.rglob("*.json"))
        path.write_text("{corrupt")
        with self.assertRaisesRegex(ValueError, "Invalid cache file"):
            embed_texts(["one"], self.embed, model_id="test", dimension=2, cache_dir=self.cache)

    def test_extraction_schema_deduplication_and_cache(self):
        calls = []
        def extract(prompt):
            calls.append(prompt)
            return {"concepts": [" Battery ", "battery"], "utility_questions": ["What is stored?"]}
        first = extract_metadata("Battery stores electricity.", extract, model_id="test", cache_dir=self.cache)
        second = extract_metadata("Battery stores electricity.", extract, model_id="test", cache_dir=self.cache)
        self.assertEqual(first, second)
        self.assertEqual(first["concepts"], ["Battery"])
        self.assertEqual(len(calls), 1)
        self.assertIn("SOURCE_CHUNK_JSON:", calls[0])
        extract_metadata("Changed text.", extract, model_id="test", cache_dir=self.cache)
        extract_metadata("Changed text.", extract, model_id="new-model", cache_dir=self.cache)
        self.assertEqual(len(calls), 3)

    def test_invalid_extraction_responses_fail(self):
        for output in [
            "not a dictionary", {}, {"concepts": [], "utility_questions": [], "extra": 1},
            {"concepts": "Battery", "utility_questions": []},
            {"concepts": [""], "utility_questions": []},
            {"concepts": [], "utility_questions": ["q"] * 4},
        ]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                extract_metadata("Text.", lambda _: output, model_id="test")

    def test_semantic_boundary_at_large_topic_change(self):
        path = self.source(
            "Solar panels collect light. Batteries store energy. Controllers regulate current. "
            "Forests contain trees. Rivers carry water. Birds build nests."
        )
        seen = []
        def topic_embeddings(texts):
            seen.extend(texts)
            return [
                [1, 0] if any(word in text for word in ("Solar", "Batteries", "Controllers"))
                else [0, 1] for text in texts
            ]
        ingestion = SpectRAGIngestion(
            embed_fn=topic_embeddings, embedding_model="test-topics", embedding_dimension=2,
        )
        chunks = ingestion.prepare([path], semantic_chunking=True)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([c["semantic_segment"] for c in chunks], [0, 1])
        self.assertTrue(all(c["chunking_mode"] == "semantic" for c in chunks))
        self.assertIn("Controllers regulate current.", chunks[0]["text"])
        self.assertNotIn("Controllers", chunks[1]["text"])
        self.assertTrue(all(text.startswith("search_document: ") for text in seen))

    def test_tied_semantic_distances_do_not_force_breaks(self):
        path = self.source("First sentence. Second sentence. Third sentence.")
        chunks = self.ingestion().prepare([path], semantic_chunking=True)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunking_mode"], "semantic")

    def test_oversized_semantic_input_fails_before_model_calls(self):
        path = self.source("word " * 80 + ". Another sentence.")
        with self.assertRaisesRegex(ValueError, "not truncated"):
            self.ingestion().prepare(
                [path], semantic_chunking=True, chunk_size=40, chunk_overlap=0,
            )
        self.assertEqual(self.calls, [])

    def test_structural_documents_do_not_use_sentence_embeddings(self):
        path = self.source("# Topic\n\nFirst sentence. Second sentence.", "source.md")
        chunks = self.ingestion().prepare([path], semantic_chunking=True)
        self.assertEqual(self.calls, [])
        self.assertEqual(chunks[0]["chunking_mode"], "structural")

    def test_semantic_chunking_does_not_embed_or_split_table_rows(self):
        path = self.source(
            "First sentence. Second sentence.\n\n| Key |\n| --- |\n| Cell |\n\n"
            "Third sentence. Fourth sentence.", "source.md",
        )
        chunks = self.ingestion().prepare([path], semantic_chunking=True)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[1]["contains_table"])
        self.assertIn("| Cell |", chunks[1]["text"])
        self.assertTrue(all("|" not in text for batch in self.calls for text in batch))

    def test_enrichment_attaches_questions_and_reuses_all_cached_outputs(self):
        path = self.source("# Topic\n\nBattery stores electricity.", "source.md")
        extraction_calls = []
        def extract(prompt):
            extraction_calls.append(prompt)
            return {"concepts": ["Battery"], "utility_questions": ["What does the battery store?"]}
        ingestion = self.ingestion(extract_fn=extract, extraction_model="extract-v1", cache_dir=self.cache)
        first = ingestion.enrich([path])
        batches = len(self.calls)
        second = ingestion.enrich([path])
        self.assertEqual(len(self.calls), batches)
        self.assertEqual(len(extraction_calls), 1)
        self.assertEqual(first[0]["concepts"], ["Battery"])
        self.assertEqual(first[0]["metadata_extraction"], "enabled")
        self.assertEqual(first[0]["utility_question_embeddings"].shape, (1, 2))
        np.testing.assert_array_equal(first[0]["embedding"], second[0]["embedding"])
        self.assertEqual(first[0]["embedding_model"], "test-model-v1")

    def test_absent_extractor_is_explicit_and_never_duplicates_dense_vectors(self):
        path = self.source("Battery stores electricity.")
        chunks = self.ingestion().enrich([path], semantic_chunking=False)
        self.assertEqual(chunks[0]["metadata_extraction"], "disabled")
        self.assertEqual(chunks[0]["utility_questions"], [])
        self.assertEqual(chunks[0]["utility_question_embeddings"].shape, (0, 2))
        self.assertEqual(chunks[0]["concepts"], [])
        self.assertEqual(chunks[0]["citations"], [])

    def test_required_model_configuration(self):
        path = self.source("Text.")
        with self.assertRaisesRegex(ValueError, "embed_fn"):
            SpectRAGIngestion().enrich([path])
        with self.assertRaisesRegex(ValueError, "embed_fn"):
            SpectRAGIngestion().prepare([path], semantic_chunking=True)
        with self.assertRaisesRegex(ValueError, "extraction_model"):
            self.ingestion(extract_fn=lambda _: {}).enrich([path])

    def test_local_fragment_external_and_numeric_citation_resolution(self):
        first = self.source(
            "# First\n\nSee [details](second.md#target), [missing](second.md#absent), "
            "[outside](https://example.org/page), and [1].", "first.md",
        )
        second = self.source("# Target\n\nSupported text.\n\n# Other\n\nOther text.", "second.md")
        chunks = self.ingestion().enrich([first, second], semantic_chunking=False)
        records = chunks[0]["citations"]
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["status"], "resolved")
        self.assertEqual(records[0]["target_chunk_ids"], [chunks[1]["chunk_id"]])
        self.assertEqual(records[1]["reason"], "fragment_not_found")
        self.assertEqual(records[2]["reason"], "external_target")
        self.assertEqual(records[3]["reason"], "bibliography_not_resolved")

    def test_citations_recompute_when_target_corpus_changes(self):
        first = self.source("[Details](second.md).", "first.md")
        second = self.source("Target.", "second.md")
        ingestion = self.ingestion(cache_dir=self.cache)
        alone = ingestion.enrich([first], semantic_chunking=False)
        together = ingestion.enrich([first, second], semantic_chunking=False)
        self.assertEqual(alone[0]["citations"][0]["status"], "unresolved")
        self.assertEqual(together[0]["citations"][0]["status"], "resolved")

    def test_citation_resolution_is_nonmutating_and_handles_malformed_links(self):
        source = str(self.root / "source.md")
        original = {
            "source": source, "doc_id": "source", "chunk_id": "source:0",
            "section_path": ["Topic"],
            "text": "[here](#topic) and [bad](http://[oops).",
        }
        records = resolve_citations([original])[0]["citations"]
        self.assertNotIn("citations", original)
        self.assertEqual(records[0]["target_chunk_ids"], ["source:0"])
        self.assertEqual(records[1]["reason"], "malformed_target")


if __name__ == "__main__":
    unittest.main()
