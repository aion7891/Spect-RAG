# Phase 2: document loading and chunking

Implemented: the beginning of the offline pipeline, corresponding to Section 3.3
of the supplied paper. This phase returns inspectable chunks; it does not build
an embedding, graph, spectral, FAISS, or BM25 index. Phase 3 adds embeddings and
semantic segmentation. Full indexing becomes available in Phase 4.

## Read the implementation in this order

1. `src/spect_rag/ingestion.py`: `prepare()` calls the loader and then chunks each
   document. This is the orchestration; helper logic stays in `utility/`.
2. `src/spect_rag/utility/loading.py`: loads explicit file paths, detects headings
   and tables, and produces ordered blocks with source metadata.
3. `src/spect_rag/utility/chunking.py`: turns blocks into complete sentences/table
   rows, packs those units into token-limited windows, and returns chunk records.
4. `examples/prepare_documents.py`: prints a preview and optionally saves JSON.

The package continues to export exactly two classes. Internal data is represented
with dictionaries and lists. No model calls or index writes occur in preparation.

## Loading rules

- Supported inputs: UTF-8 `.txt`, `.md`, `.markdown`, and machine-readable `.pdf`.
  UTF-8 BOMs are accepted. Pass a list or iterable of file paths, not a directory
  or a bare string. Duplicate resolved paths, missing files, unsupported types,
  empty documents, and decoding failures are reported explicitly.
- Markdown: ATX headings (`#`, `##`, etc.) define a hierarchy. Fenced code does
  not create headings. Pipe tables with a separator row become table blocks,
  including single-column tables. Setext headings are not recognized yet.
- PDF: pdfplumber extracts page text and detects ruled tables with its default
  line-based strategy. Detected tables are serialized to Markdown and removed
  from the plain text extraction to avoid duplicating their content. Short
  numbered titles are treated as section headings. This can confuse a numbered
  list with a heading; it is a documented heuristic.
- A PDF page with visible image/drawing content but no text causes an OCR-required
  error, including when other pages are readable. Truly blank pages are skipped.
  The package does not perform OCR or extract meaning from figures.
- Borderless tables, complex reading order, and mathematical notation can require
  a specialized parser. PDF headers/footers and layout artifacts are retained.
  Source extraction should be inspected before relying on the resulting chunks.

`doc_id` is a hash of the resolved source path: it is repeatable for the same
local path and changes if the document moves. `content_hash` separately records
the file bytes so later indexing can detect edits. These are local identities,
not globally portable document identifiers.

## Chunking rules

`prepare(paths, chunk_size=1200, chunk_overlap=100, min_chunk_tokens=32)`:

1. Count tokens with `tiktoken`'s `cl100k_base`, including newlines inserted
   between units. Strings resembling special tokens are treated as literal text.
2. Split text with a small English punctuation heuristic that protects common
   abbreviations and initials. A dangling PDF sentence is joined to its
   continuation on the next page when the section remains the same.
3. Treat each table row as indivisible. Keep the header and Markdown separator
   together. When a table spans multiple chunks, headers are not repeated.
4. Pack units greedily until the next would exceed the hard token limit.
5. Start the next window with the longest contiguous suffix of whole units that
   fits the overlap target and leaves room for the next new unit. Overlap can be
   zero if the last sentence/row is too long for the target. No unit is sliced.
6. Start a new window at a section change. Overlap does not cross section paths.
7. Attempt to merge a short final window with its predecessor, removing overlap,
   only when the section matches and the combined text fits the hard limit.
   Small sections and tails are retained when those conditions cannot hold.

An indivisible sentence or table row over the limit raises an error identifying
its source block and pages. Increasing the limit or preprocessing that block is
an explicit caller decision. Nothing is silently truncated. The sentence rule
is heuristic, so ambiguous punctuation and non-English text need inspection.

The token budget here covers the returned chunk text. A generation prompt's
instructions, query, and context headers are outside this preparation step.

## Returned records

| Field | Meaning |
| --- | --- |
| `chunk_id` | Document ID plus deterministic zero-based position |
| `doc_id`, `source`, `content_hash` | Source identity, resolved path, and file hash |
| `position` | Chunk order within its source document |
| `text`, `token_count` | Complete chunk text and its measured token length |
| `pages`, `page_start`, `page_end` | 1-based PDF provenance; empty list / null for text files |
| `section_path` | Heading hierarchy for this chunk |
| `block_indices` | Source blocks contributing text, in encounter order |
| `contains_table` | Whether any units came from a recognized table |
| `chunking_mode` | `structural` when a section path exists, otherwise `sentence` |

Mode `sentence` deliberately does not claim embedding-based semantic splitting.

## Run and review

```bash
uv sync
uv run python examples/prepare_documents.py examples/documents/getting_started.md --output tmp/phase2/sample_chunks.json
uv run python examples/prepare_documents.py docs/2610.36991v2.pdf --output tmp/phase2/paper_chunks.json
uv run python -m unittest discover -s tests -v
```

The optional JSON output contains the full text, not just the printed preview.
The first tokenizer use downloads its vocabulary into tiktoken's cache. To use
preparation offline, populate that cache first; `TIKTOKEN_CACHE_DIR` can select
its location.

Validation during implementation:

- 14 behavioral tests passed: section hierarchy, full sentence coverage, bounded
  overlap, repeated text, intact Markdown/PDF rows, literal special tokens,
  invalid inputs/settings, stable IDs, no index writes, page provenance,
  cross-page sentences, and an image-only page in a mixed PDF.
- The bundled Markdown example produced four chunks with the expected section
  paths and table rows.
- The supplied 25-page paper produced 63 chunks with token lengths from 3 to
  1,189. Every page contributed text. Its borderless tables were not detected as
  table blocks; they remain in the plain extracted text. These checks establish
  preparation behavior, not retrieval or answer quality.

Generated inspection files are under ignored `tmp/phase2/`. Review this phase
before adding embeddings and extracted metadata in Phase 3.
