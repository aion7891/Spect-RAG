---
description: How this documentation is built, published and kept correct.
---

# Maintaining these docs

The documentation lives in the repository, next to the code, and is published in two places:

| What | Source | Published by | URL |
|---|---|---|---|
| This documentation site | `docs/` (pages listed in `docs/SUMMARY.md`) | GitBook, through Git Sync with the `main` branch | [aion7891.gitbook.io/spectrag](https://aion7891.gitbook.io/spectrag/) |
| Interactive pages | `docs/interactive/*.html`, `examples/graph_view.html` | GitHub Pages, by `.github/workflows/pages.yml` | [aion7891.github.io/Spect-RAG](https://aion7891.github.io/Spect-RAG/) |

## Layout

```
.gitbook.yaml                 tells GitBook the site root is docs/
docs/
  README.md                   the landing page (Overview)
  SUMMARY.md                  the table of contents: only pages listed here are published
  getting-started/            installation, quick start, examples
  concepts/                   architecture, the spectral part
  guides/                     configuration, tuning, your own models, graph view, large corpora
  reference/                  Python API, command line, errors, index format
  results/                    benchmarks
  help/                       FAQ, this page
  assets/                     images, animations and benchmark charts used by the pages
  interactive/                self-contained HTML explainers (GitHub Pages)
  DESIGN.md, PROGRESS.md, …   developer notes: not part of the site, linked on GitHub
```

## Change a page

1. Edit the Markdown file and open a pull request, like any code change.
2. `pytest tests/test_docs.py` checks that every local link and image exists, every page is in
   `SUMMARY.md`, every setting is documented on the configuration page, and the benchmark charts
   match `benchmarks/results/`.
3. After the merge, GitBook updates the site from `main` automatically.

To **add a page**, create the file and add a line to `docs/SUMMARY.md`. The page's URL follows
its path: `guides/tuning.md` becomes `/guides/tuning`.

GitBook-specific blocks show as plain text on GitHub. Copy them from a page that already uses
them:

| Block | Used for | Example page |
|---|---|---|
| `hint` (styles `info`, `success`, `warning`, `danger`) | call-outs | [Configuration](../guides/configuration.md) |
| `tabs` / `tab` | alternatives, e.g. pip and uv | [Installation](../getting-started/installation.md) |
| `embed` | the interactive pages | [Architecture](../concepts/architecture.md) |
| `$$ … $$` | maths (KaTeX), inline or on its own line | [The spectral part](../concepts/spectral.md) |

Front matter at the top of a page sets its subtitle on the site:

```
---
description: One sentence under the page title.
---
```

{% hint style="warning" %}
**Edit in one place.** Git Sync works in both directions: an edit made in the GitBook editor is
committed to `main` by GitBook. Prefer editing in the repository, so that changes go through
review and the docs tests.
{% endhint %}

## When the code changes

| If you change... | Update |
|---|---|
| a setting, its default or its range | `guides/configuration.md` (the test fails until every setting is there) |
| a public function or its arguments | `reference/python-api.md` |
| a CLI command or option | `reference/cli.md` |
| the benchmark results | run `python benchmarks/mhr_report.py`, then copy the two charts: `cp benchmarks/results/multihoprag/{recall_at_k,budget}_chunk1200.svg docs/assets/benchmark/`, and update `results/benchmarks.md` |
| the release version | the version shown in `getting-started/installation.md` |

## The interactive pages

`docs/interactive/` holds self-contained HTML files with inline CSS and JavaScript: no build
step. The pages workflow publishes them, plus `examples/graph_view.html` as `graph-view.html`,
whenever they change on `main`. GitBook pages show them with embed blocks and links.

Edit a page, open it in a browser to check it, and commit it. `step-by-step.html` and
`spectral.html` explain the method with a compact teaching implementation; their notes say
where its values differ from the package.

## One-time setup

These steps are done once, by a maintainer with access to the repository settings.

**GitHub Pages** (for the interactive pages):

1. Repository → Settings → Pages → Build and deployment → Source: **GitHub Actions**.
2. Run the *interactive pages* workflow once (Actions → interactive pages → Run workflow), or
   push a change to `docs/interactive/`.

**GitBook** (for this site):

1. Sign in at [gitbook.com](https://www.gitbook.com) with GitHub. Create an organisation; its
   subdomain becomes the first part of the URL (`aion7891.gitbook.io`).
2. Create a space for SpectRAG, open its GitHub Sync setting, install the GitBook GitHub app
   for the `aion7891/Spect-RAG` repository, choose the `main` branch, and choose
   **GitHub → GitBook** for the first sync. GitBook reads `.gitbook.yaml` and imports the pages
   in `docs/SUMMARY.md`.
3. Publish the space as a public docs site, with the site path `spectrag`, so that the URL is
   `https://aion7891.gitbook.io/spectrag/`. GitBook offers free plans for open-source projects.

If the final URL is different, update it everywhere it appears:

```bash
grep -rl "aion7891.gitbook.io/spectrag" README.md docs pyproject.toml
```
