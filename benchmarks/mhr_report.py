"""Turn the MultiHopRAG results (JSON) into REPORT.md tables and SVG charts.

Run:  python benchmarks/mhr_report.py     (after multihoprag_benchmark.py and mhr_llm_steps.py)
Charts: grouped horizontal bars, palette slots 1-3 (validated for colour-vision deficiency in
light and dark mode); every chart has a table with the same numbers next to it.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "results" / "multihoprag"
LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
         "series": ["#2a78d6", "#eb6834", "#1baf7a"]}
DARK = {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#33322f",
        "series": ["#3987e5", "#d95926", "#199e70"]}
NAMES = {"dense": "Standard RAG (dense top-k)", "bm25": "BM25 top-k", "hybrid": "Hybrid RAG (dense + BM25, RRF)",
         "dense_mmr": "Dense + MMR", "hyde": "HyDE", "lightrag": "LightRAG (hybrid mode)",
         "spectrag": "SpectraFuse (default)", "spectrag_rho0": "SpectraFuse default, length_power 0", "spectrag_tuned": "SpectraFuse (tuned on validation)",
         "spectrag_mmr": "SpectraFuse, mmr selector", "spectrag_protected_mmr": "SpectraFuse, protected_mmr",
         "spectrag_anchor_expand": "SpectraFuse, anchor_expand", "spectrag_topk": "SpectraFuse, topk selector",
         "spectrag_heat": "SpectraFuse, heat proximity", "spectrag_no_proximity": "SpectraFuse, no proximity",
         "spectrag_no_expansion": "SpectraFuse, no graph expansion"}


def load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.is_file() else None


def style():
    """CSS variables for both modes, so the SVG follows the reader's colour scheme."""
    def block(p):
        return ("--surface:%s;--ink:%s;--muted:%s;--grid:%s;" % (p["surface"], p["ink"], p["muted"], p["grid"])
                + "".join("--s%d:%s;" % (i + 1, c) for i, c in enumerate(p["series"])))
    return ("<style>svg{%s}@media (prefers-color-scheme: dark){svg{%s}}"
            "text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif}</style>" % (block(LIGHT), block(DARK)))


def bar_chart(title, labels, series, values, path, maximum=1.0):
    """Grouped horizontal bars: one row per label, one bar per series (values in 0..maximum)."""
    bar, gap, left, width = 9, 2, 250, 420
    row = len(series) * (bar + gap) + 12
    height = 70 + row * len(labels)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
             % (left + width + 60, height, left + width + 60, height), style(),
             '<rect width="100%" height="100%" fill="var(--surface)"/>',
             '<text x="16" y="24" font-size="14" font-weight="600" fill="var(--ink)">%s</text>' % title]
    for i, name in enumerate(series):  # legend
        parts.append('<rect x="%d" y="36" width="10" height="10" rx="2" fill="var(--s%d)"/>'
                     '<text x="%d" y="45" font-size="11" fill="var(--muted)">%s</text>'
                     % (left + i * 110, i + 1, left + i * 110 + 14, name))
    for tick in np.linspace(0, maximum, 5):  # recessive grid
        x = left + width * tick / maximum
        parts.append('<line x1="%.1f" y1="56" x2="%.1f" y2="%d" stroke="var(--grid)"/>'
                     '<text x="%.1f" y="%d" font-size="10" fill="var(--muted)" text-anchor="middle">%.2g</text>'
                     % (x, x, height - 14, x, height - 2, tick))
    for r, label in enumerate(labels):
        top = 60 + r * row
        parts.append('<text x="%d" y="%d" font-size="11" fill="var(--ink)" text-anchor="end">%s</text>'
                     % (left - 8, top + row / 2, label))
        for s in range(len(series)):
            value = values[r][s]
            w = max(1.0, width * value / maximum)
            parts.append('<rect x="%d" y="%d" width="%.1f" height="%d" rx="2" fill="var(--s%d)">'
                         '<title>%s, %s: %.3f</title></rect>' % (left, top + s * (bar + gap), w, bar, s + 1,
                                                                label, series[s], value))
    path.write_text("\n".join(parts + ["</svg>"]))


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def summarise(rows):
    keys = ["recall@2", "recall@5", "recall@10", "precision@5", "f1@5", "mrr", "recall", "complete", "chunks", "ms"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def bootstrap_difference(a, b, samples=2000, seed=0):
    """95% interval of mean(a - b) over questions (paired bootstrap)."""
    diff, rng = np.array(a) - np.array(b), np.random.default_rng(seed)
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(samples)]
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def retrieval_section(config, data, lines):
    rows = data["rows"]
    summary = {m: summarise(r) for m, r in rows.items()}
    order = sorted(summary, key=lambda m: -summary[m]["recall"])
    lines += ["## Retrieval on the test set (%s, %d questions)" % (config, len(rows["dense"])), "",
              table(["Method", "Recall@2", "Recall@5", "Recall@10", "Precision@5", "F1@5", "MRR",
                     "Evidence recall (budget)", "All evidence found", "Chunks", "Latency (ms)"],
                    [[NAMES.get(m, m)] + ["%.3f" % summary[m][k] for k in
                                          ("recall@2", "recall@5", "recall@10", "precision@5", "f1@5", "mrr",
                                           "recall", "complete")]
                     + ["%.1f" % summary[m]["chunks"], "%.2f" % summary[m]["ms"]] for m in order]), ""]
    for other in ["hybrid", "dense"]:
        for best in ["spectrag", "spectrag_tuned"]:
            for key in ["recall", "recall@10"]:
                diff, low, high = bootstrap_difference([r[key] for r in rows[best]], [r[key] for r in rows[other]])
                lines.append("- %s − %s, %s: **%+.3f** (95%% CI %+.3f … %+.3f)"
                             % (NAMES[best], NAMES[other], key, diff, low, high))
    lines += ["", "By question type (evidence recall within the budget):", ""]
    kinds = sorted({r["type"] for r in rows["dense"]})
    lines.append(table(["Method"] + kinds, [[NAMES.get(m, m)] + ["%.3f" % np.mean(
        [r["recall"] for r in rows[m] if r["type"] == k]) for k in kinds] for m in order]))
    main = ["dense", "bm25", "hybrid", "dense_mmr", "spectrag", "spectrag_tuned"]
    bar_chart("Recall@k on MultiHopRAG test (%s)" % config, [NAMES[m] for m in main],
              ["Recall@2", "Recall@5", "Recall@10"],
              [[summary[m][k] for k in ("recall@2", "recall@5", "recall@10")] for m in main],
              OUT / ("recall_at_k_%s.svg" % config))
    bar_chart("Evidence within the token budget (%s)" % config, [NAMES[m] for m in main],
              ["Evidence recall", "All evidence found"],
              [[summary[m]["recall"], summary[m]["complete"]] for m in main], OUT / ("budget_%s.svg" % config))
    lines += ["", "![Recall@k](recall_at_k_%s.svg)" % config, "", "![Budget](budget_%s.svg)" % config, "",
              "Memory: %.0f MB after loading the index, %.0f MB peak during the run. Gold-label fallbacks "
              "(fact across a chunk boundary): %d." % (data["memory_mb_after_load"], data["peak_memory_mb"],
                                                       data["fallbacks"]), ""]


def main():
    lines = ["# MultiHopRAG benchmark report", "",
             "Generated by `benchmarks/mhr_report.py` from the JSON files in this folder. "
             "Embeddings: nomic-embed-text (Ollama). LLM: gemma3:4b (temperature 0).", ""]
    stats = load("index_stats.json")
    if stats:
        lines += ["## Index", "", table(["Config", "Chunks", "Graph edges", "Ingestion (s)", "Index (MB)"],
                                        [[c, s["chunks"], s["graph_edges"], s["seconds"], s["index_mb"]]
                                         for c, s in stats.items()]), ""]
    for config in ["chunk1200", "chunk256"]:
        tuning, retrieval = load("tuning_%s.json" % config), load("retrieval_%s.json" % config)
        if tuning:
            best = tuning["trials"][0]
            lines += ["## Tuning (%s, 300 validation questions, %d trials, %.0f s)" % (
                config, len(tuning["trials"]), tuning["seconds"]), "",
                      "Default settings: evidence recall %.3f. Best trial: %.3f with `%s`." % (
                          tuning["default"]["recall"], best["recall"], json.dumps(best["settings"])), ""]
        if retrieval:
            retrieval_section(config, retrieval, lines)
    for name, title in [("hyde_chunk1200.json", "HyDE (test sample)"), ("answers_chunk1200.json", "Answers")]:
        data = load(name)
        if data:
            keys = list(next(iter(data["summary"].values())))
            lines += ["## %s: %d questions" % (title, data["questions"]), "",
                      table(["Method"] + keys, [[NAMES.get(m, m)] + ["%.3f" % v[k] for k in keys]
                                                for m, v in data["summary"].items()]), ""]
    for name in ["lightrag_subset_results_2400.json", "lightrag_subset_results_matched.json"]:
        data = load(name)
        if data:
            lines += ["## LightRAG subset: %d articles, %d questions, %d-token budget for every method" % (
                data["articles"], data["questions"], data["budget"]), "", data.get("note", ""), "",
                      table(["Method", "Evidence in context", "All evidence found"],
                            [[NAMES.get(m, m), "%.3f" % v["recall"], "%.3f" % v["complete"]]
                             for m, v in data["summary"].items()]), "",
                      "LightRAG mean query time: %.1f s (includes an LLM call)." % data["lightrag_query_seconds_mean"], ""]
    for config in ["chunk1200", "chunk256"]:
        data = load("latency_%s.json" % config)
        if data:
            lines += ["## Latency and memory (%s, %d questions x %d, idle machine)" % (
                config, data["questions"], data["repeats"]), "",
                      table(["Method", "Median (ms)", "p95 (ms)"],
                            [[NAMES.get(m, m), "%.2f" % t["median_ms"], "%.2f" % t["p95_ms"]]
                             for m, t in data["timings"].items()]), "",
                      "Query embedding (nomic-embed-text via Ollama, shared by every method): %.1f ms. "
                      "Memory after loading the index: %.0f MB. Index on disk: %.1f MB." % (
                          data["query_embedding_ms"], data["memory_mb"]["after_load"], data["index_mb"]), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
