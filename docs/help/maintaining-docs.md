---
description: How this documentation is built, published and kept correct.
---

# Maintaining these docs

The documentation lives in the repository, next to the code. [MkDocs](https://www.mkdocs.org)
with the [Material](https://squidfunk.github.io/mkdocs-material/) theme builds it, and the
`docs` workflow (`.github/workflows/docs.yml`) publishes it with GitHub Pages at
[aion7891.github.io/SpectraFuse](https://aion7891.github.io/SpectraFuse/).

## Layout

```text
mkdocs.yml                    site settings and the navigation: only pages listed in nav are published
.github/docs-requirements.txt the pinned MkDocs and Material versions
docs/
  index.md                    the home page
  getting-started/            installation, quick start, examples
  concepts/                   architecture, the spectral part
  guides/                     configuration, tuning, your own models, graph view, large corpora
  reference/                  Python API, command line, errors, index format
  results/                    benchmarks
  help/                       FAQ, this page
  assets/                     images, animations and benchmark charts used by the pages
  interactive/                self-contained HTML explainers, copied to the site as they are
  stylesheets/, javascripts/  the site's extra CSS and the MathJax settings
  DESIGN.md, PROGRESS.md, …   developer notes: excluded from the site (exclude_docs), linked on GitHub
```

## Preview locally

```bash
pip install -r .github/docs-requirements.txt
mkdocs serve                  # http://127.0.0.1:8000/SpectraFuse/, reloads on every save
mkdocs build                  # strict: any broken link, anchor or missing page is an error
```

## Change a page

1. Edit the Markdown file and open a pull request, like any code change.
2. Two checks run on the pull request:
    * the `docs` workflow builds the site in strict mode, so a broken link or anchor fails it;
    * `pytest tests/test_docs.py` checks that every local link and image exists, every page is
      in the navigation, every setting is documented on the configuration page, and the copied
      benchmark charts and graph view match their sources.
3. After the merge, the `docs` workflow publishes the site.

To **add a page**, create the file and add it to `nav` in `mkdocs.yml`. The page's URL follows
its path: `guides/tuning.md` becomes `/SpectraFuse/guides/tuning/`.

Material blocks used in these pages (they show as plain text on GitHub):

| Block | Used for | Example page |
|---|---|---|
| `!!! info` / `!!! warning` (admonitions, body indented 4 spaces) | call-outs | [Configuration](../guides/configuration.md) |
| `=== "Title"` (content tabs, body indented 4 spaces) | alternatives, e.g. pip and uv | [Installation](../getting-started/installation.md) |
| `<div class="spectrafuse-embed" markdown>` with an `<iframe>` | the interactive pages | [Architecture](../concepts/architecture.md) |
| `$…$` inline, `$$` on its own lines | maths (MathJax) | [The spectral part](../concepts/spectral.md) |
| `<figure markdown="span">` | an image with a caption | [Benchmarks](../results/benchmarks.md) |

The `src` of an `<iframe>` is relative to the page's URL, which is a folder: from
`concepts/architecture.md` (served at `concepts/architecture/`) it is
`../../interactive/step-by-step.html`. Markdown links are relative to the file, as usual.

## When the code changes

| If you change... | Update |
|---|---|
| a setting, its default or its range | `guides/configuration.md` (the test fails until every setting is there) |
| a public function or its arguments | `reference/python-api.md` |
| a CLI command or option | `reference/cli.md` |
| the benchmark results | run `python benchmarks/mhr_report.py`, then copy the two charts: `cp benchmarks/results/multihoprag/{recall_at_k,budget}_chunk1200.svg docs/assets/benchmark/`, and update `results/benchmarks.md` |
| `examples/graph_view.html` | `cp examples/graph_view.html docs/interactive/graph-view.html` |
| the release version | the version shown in `getting-started/installation.md` |

## The interactive pages

`docs/interactive/` holds self-contained HTML files with inline CSS and JavaScript. MkDocs copies
them to the site unchanged, at `/SpectraFuse/interactive/`. Edit a page, open it in a browser (or
through `mkdocs serve`) to check it, and commit it. `step-by-step.html` and `spectral.html`
explain the method with a compact teaching implementation; their notes say where its values
differ from the package.

## One-time setup

Done once, by a maintainer with access to the repository settings: Repository → Settings →
Pages → Build and deployment → Source: **GitHub Actions**. Or from a terminal:

```bash
gh api -X POST repos/aion7891/SpectraFuse/pages -f build_type=workflow
gh workflow run docs.yml
```
