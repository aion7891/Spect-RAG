# SpectRAG 0.1.0: production-readiness report

> **Renamed:** this project is now **SpectraFuse** (`pip install spectrafuse`), an independent
> implementation inspired by SpectRAG; until 0.1.4 it was published as `spectrag`. This page is a
> historical record and keeps the old name. SpectRAG's official implementation is
> [SightAndCite/SpectRAG](https://github.com/SightAndCite/SpectRAG).

This report records what was checked before the 0.1.0 release, how it was checked, and what is
still open. Every number comes from a command that was run on this repository; the commands
are listed so anyone can repeat them. Machine: Apple Silicon Mac (macOS 26), Python 3.12 unless
stated, Ollama with `nomic-embed-text` and `gemma3:4b`.

**Verdict: ready for a 0.1.0 release** (a first public version with paper-based defaults). The
open items below are decisions for the maintainer, or are documented limitations. None of them
is a known bug.

## 1. Summary

| Area | Result | How to repeat |
|---|---|---|
| Unit, integration, equivalence tests | **319 passed, 1 skipped** (all extras); **305 passed, 15 skipped** (numpy only; the skips need optional packages) | `uv run --all-extras pytest -q` |
| Live tests (Ollama) | **10 passed**: 5 end-to-end, and 5 with a real LLM (`SPECTRAG_LIVE_LLM=1`) | `SPECTRAG_LIVE_LLM=1 uv run --all-extras pytest -q tests/test_end_to_end.py tests/test_questions.py` |
| Mathematics on a real index | **10 of 10 checks pass** (section 2) | `python tools/validate_release.py` |
| Benchmark reproducibility | All **1,955** MultiHopRAG test questions give the same results as the committed ones (0 differences) | same script |
| Package build | wheel 75 KB, sdist 101 KB; `twine check` **PASSED** for both | `uv build && twine check dist/*` |
| Clean install | wheel installed in fresh Python 3.10 and 3.14 environments: test suite passes (318 passed, 1 skipped, before the last docs-link test was added); `quickstart.py` and the `spectrag` CLI work | section 4 |
| Install from GitHub | `pip install "spectrag[all] @ git+https://github.com/aion7891/SpectraFuse.git@base-2"` works | section 4 |
| Documentation | every README Python example and 7 shell commands run; every setting in `DEFAULTS` is documented; every local link and image exists | `pytest tests/test_docs.py` |
| Lint | no pyflakes errors (unused imports, undefined names, syntax) | `ruff check --isolated --select F,E9 src tests tools benchmarks` |

## 2. Mathematical validation on real data

`tools/validate_release.py` loads the MultiHopRAG index (609 articles, 10,335 chunks,
`nomic-embed-text`). It re-checks the formulas against independent computations and writes
[`validation/results.json`](validation/results.json). The script took 13.4 s.

| Check | Paper | Result |
|---|---|---|
| Graph is symmetric, weights in (0, 1] | Eq. 10 | pass (56,390 edges) |
| Stored eigenpairs satisfy L u = λ u | Eq. 11–12 | largest residual **5.3 × 10⁻¹⁵** |
| Eigenvectors are orthonormal | Eq. 12 | pass (tolerance 10⁻⁸) |
| Eigenvalues in [0, 2] and ascending | Eq. 11 | pass (λ₂ = 0.0077 … λ₃₃ = 0.058) |
| Coordinates exclude the trivial eigenvector D^½·1 | Eq. 13 | pass |
| PageRank iteration equals the closed form β(I − (1−β)Ã)⁻¹r on a real 200-node subgraph | Eq. 18 | largest error **1.7 × 10⁻¹⁷** |
| BM25 equals the reference formula on 400 real chunks | Table 2 | largest error **0.0** |
| Every selector (5) × proximity mode (5) × budget (500, 4,000, 12,000) on 15 questions | Eq. 19–22 | **1,125 runs**: none goes over the budget, and none picks a chunk twice or picks a non-candidate |
| Invalid settings are rejected | | 4 of 4 rejected with a clear message |
| Benchmark reproduces | | 1,955 questions, 0 differences |

The test suite also checks each formula on small cases: against hand-computed values, closed
forms (`scipy.linalg.expm` for the heat kernel, a dense eigensolver for the sparse one), and
simple loop versions of the vectorised code (71 random cases in `tests/test_equivalence.py`).

## 3. Real-document end-to-end tests

`tests/test_end_to_end.py` runs the full pipeline on real documents, not toy text:

- **The SpectRAG paper (PDF).** Three questions about the paper. Each must retrieve the passage
  with the answer ("32 smallest eigenvectors", "12k-token budget", "two hops and a cap of")
  within 2,000 tokens.
- **60 MultiHopRAG news articles.** Every test question whose evidence lies in those 60 articles
  must stay within its budget with no duplicates. Evidence recall must be above 0.5 with the
  offline hash embedder.
- **Live:** the same paper indexed with `nomic-embed-text`. `gemma3:4b` must answer "How many
  dimensions do the spectral coordinates of each chunk have?" with 32.

All pass.

## 4. Packaging and installation

| Check | Result |
|---|---|
| `requires-python` | `>=3.10`, with no upper limit |
| Dependencies | `numpy>=1.24` only; extras `tokens`, `pdf`, `scale`, `all` (alias `extras`), `dev` |
| Metadata | MIT license, author Kumar Nishant (no email address), project URLs, `spectrag` console script; `twine check` passed |
| Fresh environments | wheel installed into new Python 3.10 and 3.14 environments: `spectrag --version` gives 0.1.0; ingest, search, info and graph work; `quickstart.py` runs with no arguments |
| CI (`.github/workflows/tests.yml`) | Linux and macOS × Python 3.10 / 3.12 / 3.14 × minimal and all extras; a build job (wheel → clean environment → CLI); an allowed-to-fail Python 3.15 pre-release job. *Since 0.1.2: Linux keeps every combination, and macOS runs one job (Python 3.12, all extras), because hosted macOS runners can queue for half an hour* |
| Index files | format version 2, checked when loading; saving is atomic, through a `<folder>.partial` folder that replaces the old index only when complete. *Since 0.1.3, the old index is renamed aside rather than deleted first, so a crash during the swap no longer leaves the folder empty (DESIGN.md D66)* |

## 5. Performance (from Phases 1–3)

| Measure | Value |
|---|---|
| Search latency, MultiHopRAG index (10,335 chunks, 12k budget) | median 3.5 ms (tuned) to 4.4 ms (default) per question; embedding the question with `nomic-embed-text` adds 19 ms |
| Memory after loading that index | 256 MB |
| Phase 1 speed-ups on 10,335 chunks | search 36.4 → 4.1 ms; ingestion (after embedding) 13.2 → 7.7 s; index size 159 → 49 MB; peak memory 1,556 → 1,046 MB |
| kNN graph with FAISS HNSW at 100k chunks | 6.8× faster to build than exact search, with the same edges |

## 6. Retrieval quality (MultiHopRAG, quick mode)

Test set of 1,955 questions, `nomic-embed-text`, 12,000-token budget for every method. Settings
were tuned on 300 separate validation questions.

| Method | Evidence recall | Recall@10 |
|---|---|---|
| Dense | 0.758 | |
| BM25 | 0.788 | |
| Hybrid (dense + BM25, RRF) | 0.832 | |
| SpectRAG, default settings | 0.803 | 0.433 |
| **SpectRAG, tuned on validation** | **0.846** | **0.592** |

Tuned SpectRAG beats hybrid by +0.014 evidence recall (95% CI +0.007 to +0.021). The gain is
largest on inference questions (0.802 vs 0.757). Full tables, charts and method:
[`benchmarks/results/multihoprag/REPORT.md`](../benchmarks/results/multihoprag/REPORT.md).

## 7. Known issues and open items

| Item | Status | Detail |
|---|---|---|
| Default settings are not tuned | **by design for 0.1.0** | The defaults follow the paper, and they score below hybrid retrieval on MultiHopRAG (0.803 vs 0.832). The tuned settings (0.846) are listed in the benchmark report. Tuned defaults are planned for the next version. |
| Benchmark ran in quick mode | documented | Quick mode uses the hash embedder only to find semantic-chunking breaks (sentence distances); chunks and questions are embedded with `nomic-embed-text`. Full mode (`SPECTRAG_BENCH_FULL=1`) was not run because it takes many hours on this machine. |
| LightRAG comparison | inconclusive | The local LLM took about 225 s per article, so LightRAG ran on a 4-article subset only. On that subset every method scores 1.0. No claim is made against LightRAG. |
| numpy 2.2 on macOS warnings | harmless, not hidden | Some matrix products print a spurious `RuntimeWarning: divide by zero / overflow encountered in matmul`. The results were checked and are correct (section 2). |
| Python 3.15 | blocked upstream | Only an alpha of Python 3.15 exists, and numpy does not load on it yet. The CI job may fail and does not block a release. |
| Tests in the sdist | minor | The sdist includes `tests/`, but some tests need `benchmarks/`, `examples/` and `docs/` from the repository. Run the tests from a clone. |
| `pip install git+…Spect-RAG.git` | waits for the merge | `main` still holds only the first commit, so installing from `main` fails ("does not appear to be a Python project"). It works once this branch is merged. Until then, the README shows `@base-2`. |
| License and author | maintainer to confirm | MIT and "Kumar Nishant" were filled in as placeholders. |
| PyPI release | maintainer's decision | Not uploaded yet. The release workflow publishes on a version tag after a one-time setup (section 8). |

## 8. Releasing (continuous delivery)

Releases are automated by [`.github/workflows/release.yml`](../.github/workflows/release.yml).
Pushing a version tag on `main` starts it:

| Tag | Where it is published |
|---|---|
| `v0.1.0` (final) | TestPyPI, then PyPI once it installs from TestPyPI, plus a GitHub Release with the wheel and sdist |
| `v0.1.0rc1`, `a1`, `b1`, `.dev1` (pre-release) | TestPyPI only, as a dry run |

The jobs run in this order, and each one runs only if the one before it passed:

```
check → tests → build → testpypi → testpypi-check → pypi (final tags only) → github-release
```

1. **check:** the tag is on `main`, and it equals `__version__` in `src/spectrafuse/_version.py`
   (`v0.1.0` needs `0.1.0`);
2. **tests:** two quick Linux jobs, Python 3.10 with numpy only and Python 3.14 with all extras
   (`pytest` and `quickstart.py`). The full matrix in `tests.yml` already ran on `main` when
   the pull request was merged, so it is not repeated here;
3. **build:** the wheel and sdist pass `twine check --strict`;
4. **testpypi:** both files are uploaded to TestPyPI;
5. **testpypi-check:** `spectrag==<version>` is installed from TestPyPI into a clean environment
   (numpy from PyPI, spectrag alone from TestPyPI with `--no-deps`); `spectrag --version` must
   match the tag, and a one-document ingest and search must work;
6. **pypi** and **github-release:** for a final tag only.

Publishing uses PyPI Trusted Publishing (OpenID Connect), so no API token is stored in GitHub.

### One-time setup

1. **PyPI** → Account settings → Publishing → *Add a new pending publisher*:
   - project `spectrag`;
   - owner `aion7891`;
   - repository `Spect-RAG`;
   - workflow `release.yml`;
   - environment `pypi`.
2. **TestPyPI** (test.pypi.org; a separate account) → the same, with environment `testpypi`.
3. **GitHub** → repository Settings → Environments → create `pypi` and `testpypi`. For `pypi`,
   it is recommended to add yourself under *Required reviewers*, so each release waits for
   one click of approval, and to allow only tags matching `v*` under deployment branches and
   tags.

### Each release

```bash
# 1. Set the version in src/spectrafuse/_version.py and add the CHANGELOG entry, through a PR to main
# 2. The release: TestPyPI, the install check, then PyPI and the GitHub Release
git switch main && git pull origin main
git tag -a v0.1.0 -m "SpectRAG 0.1.0" && git push origin v0.1.0
gh run watch
# Optional: a pre-release first (set __version__ = "0.1.0rc1" in a PR), TestPyPI only
```

A tag that fails before `testpypi` publishes nothing: delete the tag
(`git push --delete origin v0.1.0 && git tag -d v0.1.0`), fix the problem, and tag again. A
run that failed later (for example a network error while publishing) can be re-run with
`gh run rerun <run id> --failed`. Files that are already uploaded are skipped, and an existing
GitHub Release gets its files replaced. A version that reached PyPI or TestPyPI cannot be
uploaded again with different files, even after it is deleted there. A broken release is fixed
with a new version, e.g. 0.1.1.
