# Phase 3: embeddings and extracted metadata

This phase implements the model-assisted preparation steps in Table 1 and the
semantic segmentation described in Section 3.3. It stops before graph building,
spectral decomposition, and retrieval. The package still exports exactly two
classes; new operations are methods on `SpectRAGIngestion`.

## Read the implementation in this order

1. `src/spect_rag/ingestion.py`: `prepare()` optionally embeds sentences for
   boundaries; `enrich()` then embeds chunks, extracts concepts/questions, embeds
   questions, and resolves citations. These methods orchestrate utility functions.
2. `utility/semantic_chunking.py`: computes adjacent-sentence cosine distances
   and selects breakpoints strictly above the document's 90th percentile.
3. `utility/embeddings.py`: adds the task prefix, batches callback requests,
   validates shape and numbers, normalizes vectors, and restores input order.
4. `utility/extraction.py`: defines the extraction prompt and validates its result.
5. `utility/citations.py`: finds explicit references and resolves local links
   against the current corpus.
6. `utility/cache.py`: hashes configuration plus text and atomically saves JSON.

## Callback contracts

`embed_fn(texts: list[str])` receives already-prefixed text. Return a rectangular
numeric matrix with one vector per input in the same order. A Python list of
lists or a NumPy array works. Do not add the same task prefix again in your
client adapter. The default prefix is `search_document: ` for sentences, chunks,
and stored questions. A model that does not use prefixes can set it to `""`.
The reusable embedding utility also accepts `search_query: ` for later retrieval.

The default dimension is 768, matching the paper's embedding setting. Supply
the actual dimension when using another model. Wrong shapes, nonnumeric values,
NaN/infinity, and zero vectors are rejected. Valid vectors are L2-normalized and
returned as contiguous `float32` arrays. Scaling before normalization avoids
overflow for very large finite inputs. Missing texts are batched in groups of
64, with duplicate requests deduplicated and output order restored.

`extract_fn(prompt: str)` receives a prompt containing the source chunk encoded
as a JSON string. Your adapter must call its model and decode the response to a
Python dictionary with exactly these two fields:

```json
{
  "concepts": ["Battery", "Solar panel"],
  "utility_questions": ["What does a battery store?"]
}
```

The prompt requests at most 12 concepts and 3 self-contained questions supported
by the chunk. These counts are implementation choices, not paper defaults.
Nonempty strings are required; whitespace is normalized and case-insensitive
duplicates are removed. Empty lists are valid. Malformed responses fail explicitly
instead of being cached or silently substituted. This checks response structure,
not the truth or grounding of model-generated statements; real outputs need review.

The package has no provider SDK or hidden model service. Callback exceptions
propagate to the caller. Model names describe the supplied callbacks; they do
not cause a model to be downloaded or called automatically.

## Configure ingestion

After defining your real client adapters as `my_embed` and `my_extract`:

```python
from spect_rag import SpectRAGIngestion

ingestion = SpectRAGIngestion(
    embed_fn=my_embed,
    embedding_model="nomic-embed-text@your-revision",
    embedding_dimension=768,
    document_prefix="search_document: ",
    extract_fn=my_extract,
    extraction_model="your-extraction-model@your-revision",
    cache_dir="tmp/my-model-cache",
)
chunks = ingestion.enrich(["docs/my_document.pdf", "docs/notes.md"])
```

Both callback names in this integration example stand for your own client code.
There is no implicit connection to the models named above. Include revisions and
settings that affect results in model identities. The paper's extraction model
is `gpt-4o-mini`; the package accepts other models with the same callback contract.

`prepare(paths)` retains its model-free Phase 2 behavior. Request semantic
segmentation with `prepare(paths, semantic_chunking=True)` once an embedding
callback is configured. `enrich(paths)` enables that option by default; all
chunk-size/overlap/minimum settings remain configurable.

Leaving `extract_fn` unset explicitly disables concept/question generation.
Records then contain `metadata_extraction="disabled"`, empty concept/question
lists, and a `(0, dimension)` question-vector matrix. Dense vectors are never
copied into a missing question channel. A complete six-signal implementation will
need those extracted signals enabled when Phase 4 builds the graph.

## Semantic boundaries

Documents with recognized heading paths keep structural segmentation; they do
not pay for sentence embeddings. Entirely unstructured documents embed complete
sentences and use `distance = 1 - cosine(previous, current)`. A boundary occurs
only when distance is strictly above the 90th percentile of eligible adjacent
sentence distances. Equal distances do not force a boundary.

Table transitions are hard boundaries. Rows are not sentence-embedded, and
distance comparisons never skip across a table. Existing token packing still
enforces the hard limit; overlap and short-tail merging cannot cross semantic
boundaries. A single sentence needs no distance computation. Oversized units
are rejected before any sentence-model call. Phase 2's English sentence-splitting
and PDF layout limitations still apply.

## Citation targets

Each chunk receives a `citations` list. References are recognized from inline
Markdown links, bare HTTP(S) URLs, and numeric brackets such as `[1]` or `[1,2]`.
Local relative links are resolved against the source directory and the loaded
corpus. A link to a document targets all its chunks; a heading fragment targets
chunks sharing its simple heading slug.

Each citation retains `reference`, `target`, and `kind`, plus `status`,
`target_doc_id`, `target_chunk_ids`, and an unresolved `reason` when needed.
No network requests or automatic reads of linked files occur. Missing files,
unmatched fragments, malformed targets, external URLs, and unmapped numeric
bibliography references remain unresolved. This avoids fabricating citation edges.

Heading slugs support common punctuation/space normalization, not every Markdown
renderer or duplicate-heading suffix. Reference-style Markdown links and PDF
link annotations are not parsed. Numeric bibliography matching is not implemented.
Citation resolution is recomputed for the current corpus, rather than cached as
though target availability never changed.

## Caching and returned data

Caching is opt-in through `cache_dir`; by default no disk cache is written.
Constructors do not create files or call models. Configured model calls write
versioned JSON files under `embeddings/` and `extractions/`. Keys include text,
model identity, embedding dimension/prefix, or the exact extraction instructions.
Writes use a temporary file and atomic replacement. Corrupt cache entries raise
an explicit error; remove the identified entry to recompute it. No pickle is used.

If a model/service changes behavior without a changed identity, the package cannot
detect that remotely: change the configured identity or clear its cache. JSON
caches contain source/model output text and vectors in the location you choose.

`enrich()` returns the Phase 2 fields plus:

| Field | Value |
| --- | --- |
| `semantic_segment` | Segment number for semantic mode, otherwise null |
| `embedding` | Normalized `(dimension,)` NumPy vector |
| `concepts` | Deduplicated concept/entity strings |
| `utility_questions` | Generated question text, preserved for inspection |
| `utility_question_embeddings` | Normalized `(number_of_questions, dimension)` matrix |
| `embedding_model`, `embedding_dimension`, `embedding_prefix` | Actual configured embedding identity/settings |
| `extraction_model`, `metadata_extraction` | Extractor identity and enabled/disabled state |
| `citations` | Resolved/unresolved explicit reference records |

Arrays stay as NumPy arrays in memory. The example converts them to JSON lists
for inspection. This cache/inspection output is not the reusable retrieval index;
that artifact contract is implemented in Phase 4.

## Run and validate

```bash
uv run python examples/enrich_documents.py examples/documents/getting_started.md examples/documents/storage_notes.txt --demo --output tmp/phase3/enriched_chunks.json
uv run python -m unittest discover -s tests -v
```

The example requires `--demo` because its callbacks are deterministic word hashes
and keyword rules. They are deliberately outside the installed package and are
not semantic embeddings or LLM extraction. The example exercises the algorithm
and cache plumbing without claiming real model quality. Run twice with the same
cache to see callback counts fall to zero.

The two-document example produced six enriched chunks and resolved the link from
the text file to the Markdown charging section. Its second run used zero model
callbacks. The supplied paper was also processed end to end with test callbacks;
this checks integration with PDF preparation, not semantic retrieval quality.

Automated validation covers normalization and prefixes, input ordering and
batching, invalid vectors/responses, cache reuse/invalidation/corruption,
handcrafted semantic topic changes and ties, preserved table/section boundaries,
question vectors and disabled extraction, and citation resolution. Real model
services and retrieval/answer quality have not been evaluated in this phase.

Review this phase before adding the graph, spectrum, and index persistence.
