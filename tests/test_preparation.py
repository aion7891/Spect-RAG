"""Behavioral checks for Phase 2; run: python -m unittest discover -s tests -v."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from spect_rag import SpectRAGIngestion
from spect_rag.utility.chunking import chunk_document, count_tokens, split_sentences
from spect_rag.utility.loading import load_document, load_documents

from pdf_fixture import write_pdf


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def source(self, text, filename="source.txt"):
        path = self.root / filename
        path.write_text(text, encoding="utf-8")
        return path

    def test_sections_keep_nested_paths_and_short_sections(self):
        path = self.source("# Root\n\nIntro.\n\n## Child\n\nBody.\n\n# Next\n\nEnd.", "source.md")
        chunks = SpectRAGIngestion().prepare([path])
        self.assertEqual([c["section_path"] for c in chunks], [["Root"], ["Root", "Child"], ["Next"]])
        self.assertTrue(all(c["token_count"] < 32 for c in chunks))
        self.assertEqual(chunks[0]["pages"], [])

    def test_sentence_windows_keep_all_text_and_bounded_overlap(self):
        sentences = [f"Sentence number {i} describes a unique topic." for i in range(20)]
        path = self.source(" ".join(sentences))
        chunks = SpectRAGIngestion().prepare(
            [path], chunk_size=42, chunk_overlap=12, min_chunk_tokens=0,
        )
        self.assertGreater(len(chunks), 1)
        restored = []
        for chunk in chunks:
            self.assertLessEqual(chunk["token_count"], 42)
            self.assertEqual(chunk["token_count"], count_tokens(chunk["text"]))
            current = chunk["text"].splitlines()
            overlap = 0
            for length in range(1, min(len(restored), len(current)) + 1):
                if restored[-length:] == current[:length]:
                    overlap = length
            self.assertLessEqual(count_tokens("\n".join(current[:overlap])), 12)
            restored.extend(current[overlap:])
        self.assertEqual(restored, sentences)
        self.assertEqual(chunks, SpectRAGIngestion().prepare(
            [path], chunk_size=42, chunk_overlap=12, min_chunk_tokens=0,
        ))

    def test_zero_overlap_and_repeated_sentences_preserve_multiplicity(self):
        path = self.source("Same sentence. " * 30)
        chunks = SpectRAGIngestion().prepare(
            [path], chunk_size=20, chunk_overlap=0, min_chunk_tokens=10,
        )
        self.assertEqual(sum(c["text"].count("Same sentence.") for c in chunks), 30)

    def test_markdown_table_rows_remain_intact(self):
        rows = [f"| Item {i} | Purpose {i} |" for i in range(12)]
        path = self.source("| Item | Purpose |\n| --- | --- |\n" + "\n".join(rows), "table.md")
        chunks = SpectRAGIngestion().prepare(
            [path], chunk_size=45, chunk_overlap=0, min_chunk_tokens=0,
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["contains_table"] for c in chunks))
        self.assertEqual(
            "\n".join(c["text"] for c in chunks).splitlines(),
            ["| Item | Purpose |", "| --- | --- |", *rows],
        )

    def test_single_column_table_and_code_fence(self):
        text = "| Item |\n| --- |\n| A |\n\n" + chr(96) * 3 + "\n# Not a heading\n" + chr(96) * 3
        doc = load_document(self.source(text, "source.md"))
        self.assertEqual(doc["blocks"][0]["kind"], "table")
        self.assertTrue(all(b["section_path"] == [] for b in doc["blocks"]))

    def test_oversized_sentence_and_table_row_fail_without_truncation(self):
        for text, name in [
            ("word " * 80 + ".", "long.txt"),
            ("| Name |\n| --- |\n| " + "word " * 80 + "|", "long.md"),
        ]:
            with self.subTest(name=name):
                path = self.source(text, name)
                with self.assertRaisesRegex(ValueError, "not truncated"):
                    SpectRAGIngestion().prepare([path], chunk_size=40, chunk_overlap=0)
                self.assertEqual(path.read_text(), text)

    def test_abbreviations_initials_and_quotes(self):
        self.assertEqual(
            split_sentences('Dr. Patel met J. Smith in the U.S. office. "Ready!" Next step.'),
            ['Dr. Patel met J. Smith in the U.S. office.', '"Ready!"', 'Next step.'],
        )

    def test_literal_special_token_and_unicode(self):
        text = "Literal <|endoftext|> marker. Unicode café 中文."
        chunks = SpectRAGIngestion().prepare([self.source(text)])
        self.assertIn("<|endoftext|>", chunks[0]["text"])
        self.assertIn("中文", chunks[0]["text"])

    def test_input_errors(self):
        path = self.source("A sentence.")
        cases = [
            ([], ValueError), ([path, path], ValueError),
            (path, TypeError), ([self.root / "missing.txt"], FileNotFoundError),
            ([self.source("", "empty.txt")], ValueError),
            ([self.source("content", "file.csv")], ValueError),
        ]
        for paths, error in cases:
            with self.subTest(paths=paths), self.assertRaises(error):
                load_documents(paths)

    def test_invalid_chunk_settings(self):
        doc = load_document(self.source("A sentence."))
        for options in [
            {"chunk_size": 0}, {"chunk_overlap": -1},
            {"chunk_size": 100, "chunk_overlap": 100},
            {"min_chunk_tokens": 1201}, {"chunk_size": True}, {"chunk_overlap": 1.5},
        ]:
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                chunk_document(doc, **options)

    def test_identity_content_changes_and_no_index_side_effects(self):
        path = self.source("First text.")
        first = load_document(path)
        path.write_text("Updated text.")
        second = load_document(path)
        self.assertEqual(first["doc_id"], second["doc_id"])
        self.assertNotEqual(first["content_hash"], second["content_hash"])
        index = self.root / "index"
        chunks = SpectRAGIngestion(index).prepare([path])
        self.assertFalse(index.exists())
        self.assertEqual(chunks[0]["source"], str(path.resolve()))

    def test_pdf_pages_headings_and_cross_page_sentence(self):
        path = self.root / "text.pdf"
        write_pdf(path, [
            "BT /F1 12 Tf 50 750 Td (1 Overview) Tj 0 -20 Td "
            "(This sentence continues on) Tj ET",
            "BT /F1 12 Tf 50 750 Td (the next page. Another sentence.) Tj ET",
        ])
        chunks = SpectRAGIngestion().prepare([path])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["pages"], [1, 2])
        self.assertEqual(chunks[0]["page_start"], 1)
        self.assertEqual(chunks[0]["page_end"], 2)
        self.assertEqual(chunks[0]["section_path"], ["1 Overview"])
        self.assertIn("continues on\nthe next page.", chunks[0]["text"])
        self.assertEqual(chunks[0]["block_indices"], [0, 1])

    def test_image_only_page_in_mixed_pdf_requires_ocr(self):
        path = self.root / "scan.pdf"
        write_pdf(path, [
            "BT /F1 12 Tf 50 750 Td (Readable first page.) Tj ET",
            "q 80 0 0 80 50 650 cm /Im1 Do Q",
        ])
        with self.assertRaisesRegex(ValueError, "page 2.*OCR"):
            SpectRAGIngestion().prepare([path])

    def test_ruled_pdf_table_is_extracted_once(self):
        path = self.root / "table.pdf"
        write_pdf(path, [
            "50 700 m 350 700 l S 50 675 m 350 675 l S 50 650 m 350 650 l S "
            "50 650 m 50 700 l S 200 650 m 200 700 l S 350 650 m 350 700 l S "
            "BT /F1 12 Tf 60 682 Td (Name) Tj 150 0 Td (Value) Tj "
            "-150 -25 Td (Battery) Tj 150 0 Td (Storage) Tj ET"
        ])
        doc = load_document(path)
        self.assertEqual(len(doc["blocks"]), 1)
        self.assertEqual(doc["blocks"][0]["kind"], "table")
        chunks = chunk_document(doc)
        self.assertTrue(chunks[0]["contains_table"])
        self.assertEqual(chunks[0]["text"].count("Battery"), 1)
        self.assertIn("| Battery | Storage |", chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
