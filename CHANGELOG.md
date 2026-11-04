# Changelog

## Unreleased

## 0.1.3 (2026-11-04)

- **Safe index replace:** re-ingesting into an existing index used to delete the old index
  before moving the new one into place, so a crash between the two steps left no index at all.
  The old index is now renamed to `<index>.old`, the new one is renamed into place, and only
  then is `<index>.old` deleted. If the new index can't be moved into place, the old one is put
  back. If the program stops between the two renames, `SpectRAGRetrieval` reads `<index>.old`
  (with a warning) and the next ingest renames it back. Opening an index never moves or deletes
  a folder, so searches can run while an index is replaced. Two ingests into the same folder at
  the same time are still not supported.
- **CI:** each pull-request push runs the tests once (8 jobs, 1 on macOS, instead of 26), and a
  release runs two quick Linux test jobs instead of the full matrix, so releases no longer wait
  for scarce macOS runners.

## 0.1.2 (2026-11-04)

- **Custom embedders reopen:** an index built with a `functools.partial` or a callable object
  as its embedder could not be opened again, even with `embed_fn=`, because `settings.json`
  held the object's `str()`. Embed functions are now saved by name, as `"custom:<name>"`, and
  indexes written by 0.1.1 or earlier open again when the function is passed.
- **Embedder fingerprint:** a new index file, `embedder_probe.npy`, holds the vector of one fixed
  sentence. Opening an index embeds it again and raises `EmbedderError` if the embedder gives a
  different size or the cosine is below 0.99. Before, opening an index with a different
  embedder of the same size (for example `hash` and `nomic-embed-text`, both 768-d) was
  accepted, and every search returned unrelated chunks. A wrong-size embedder is now caught
  when the index is opened, not at the first search. Indexes made before 0.1.2 have no
  fingerprint and open as before, with a log message. The index format stays 2.

## 0.1.1 (2026-11-04)

- **Every deliberate error is a `SpectRAGError`:** eight error paths that raised plain built-in
  exceptions now raise SpectRAG classes that still inherit the old built-in, so existing
  `except ValueError` (or `FileNotFoundError`, `ImportError`, `ConnectionError`) code keeps
  working, and the `spectrag` command prints one `spectrag: error: ...` line instead of a
  traceback. New classes: `DocumentNotFoundError` (a missing file or folder given to
  `ingest()`), `MissingPackageError` (`faiss-cpu`, `hnswlib` or `pypdf` not installed) and
  `LLMError` (the Ollama LLM cannot be reached). A custom embedder opened without `embed_fn`, a
  graph setting passed to `search(settings=...)` and an unknown selector or proximity mode now
  raise `SettingsError`, a missing index folder in `load_index()` raises `IndexError_`, and
  evidence found in no chunk raises `DocumentError`.
- **Releases go through TestPyPI first:** every version tag is published to TestPyPI, installed
  from there into a clean environment and tried (`spectrag --version`, one ingest and search),
  and only then published to PyPI and as a GitHub Release. A failed release run can be re-run:
  files that are already uploaded are skipped.
- **Documentation site:** built with MkDocs (Material theme) from `docs/` and published with
  GitHub Pages at https://aion7891.github.io/Spect-RAG/ by the `docs` workflow: overview,
  installation, quick start, architecture, the spectral method, configuration, tuning, custom
  models, graph view, large corpora, API and CLI reference, benchmarks, FAQ, and the interactive
  explainers in `docs/interactive/`. Pull requests build it in strict mode, and
  `tests/test_docs.py` checks the pages' links, navigation and copied files.

## 0.1.0 (2026-11-03): first public release

The base release, with every core part of SpectRAG (arXiv:2610.36991). Default settings follow
the paper where it gives values; tuned defaults are planned for a later version.

- **Ingestion:** structural and semantic chunking, six edge signals (semantic, adjacency, section,
  entity, utility questions, citation), weighted chunk graph (Eq. 10), normalised Laplacian and
  spectral coordinates (Eq. 11–13), spectral clusters.
- **Retrieval:** hybrid seeds (dense + BM25 + utility questions, RRF), two-hop graph expansion,
  proximity re-scoring (Personalized PageRank, spectral closeness, heat kernel), token-budgeted
  selection (Eq. 19–22) and the paper's alternative selectors (`mmr`, `protected_mmr`,
  `anchor_expand_ce`), answers with any LLM.
- **Scale:** exact, FAISS or HNSW vector search; sparse eigensolver above 2,000 chunks; binary
  index files.
- **Configuration:** every setting validated with clear messages; per-search overrides;
  `rebuild_graph()` changes graph settings without re-embedding; grid/random-search tuning.
- **Production:** exceptions that subclass the matching built-ins, a `spectrag` logger,
  versioned index format, safe (atomic) index saving, retries for Ollama calls, the `spectrag`
  command-line tool, `quickstart.py`.
- **Visualization:** an interactive, self-contained HTML graph view of the chunk graph and of a
  question's retrieval path (`spectrag graph`, `save_graph_view()`).
- **Evaluation:** the paper's metrics and a reproducible MultiHopRAG benchmark (`benchmarks/`).
- **Validation:** end-to-end tests on real documents (the paper PDF, MultiHopRAG articles) and
  `tools/validate_release.py`, which re-checks the mathematics on a real index; see
  `docs/PRODUCTION_READINESS.md`.
- **Release automation:** pushing a version tag on `main` runs the tests, builds the package and
  publishes it to PyPI (or TestPyPI for pre-release tags) with Trusted Publishing.
