"""MultiHopRAG benchmark for SpectraFuse (paper Section 4) with real nomic-embed-text embeddings.

Run all steps:   python benchmarks/multihoprag_benchmark.py
Or one step:     python benchmarks/multihoprag_benchmark.py index|tune|evaluate
Quick mode (default): one chunk setting (1,200 tokens) and 30 tuning trials.
Full mode:       SPECTRAFUSE_BENCH_FULL=1 python benchmarks/multihoprag_benchmark.py
Needs Ollama with nomic-embed-text. Results go to benchmarks/results/multihoprag/.
Validation (300 questions) is used only for tuning; every reported number is on the test set.
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mhr_data import chunks_by_doc, gold_chunks, load_docs, load_questions, split  # noqa: E402
from mhr_eval import BASELINES, average, metrics, run  # noqa: E402
from spectrafuse import SpectRAGIngestion, SpectRAGRetrieval  # noqa: E402
from spectrafuse.utils.tuning import tune  # noqa: E402

HERE = Path(__file__).parent
INDEXES, OUT = HERE / "indexes", HERE / "results" / "multihoprag"
FULL = os.environ.get("SPECTRAFUSE_BENCH_FULL") == "1"  # quick mode by default (minutes, not hours)
CONFIGS = {"chunk1200": {"chunk_tokens": 1200, "overlap_tokens": 100, "budget": 12000}}
if FULL:
    CONFIGS["chunk256"] = {"chunk_tokens": 256, "overlap_tokens": 64, "budget": 2560}
# Chunks and questions: nomic-embed-text. Quick mode finds semantic-chunking breaks with the
# offline hash embedder (sentence distances only), which is about 60x faster to ingest.
EMBEDDER = {"embedder": "ollama", "semantic_embedder": None if FULL else "hash"}
GRID = {"alpha": [0.0, 0.15, 0.3, 0.5, 0.7], "precision_floor": [0.2, 0.4, 0.6],
        "seed_count": [10, 20, 40], "selector": ["spectrag", "protected_mmr"],
        "proximity_mode": ["hybrid", "ppr", "heat"], "bm25_weight": [0.3, 0.5, 0.7],
        "length_power": [0.0, 0.5], "edge_threshold": [0.05, 0.1], "graph_k": [5, 10, 20]}
TRIALS = 120 if FULL else 30
VARIANTS = {"spectrag": {}, "spectrag_rho0": {"length_power": 0.0}, "spectrag_mmr": {"selector": "mmr"},
            "spectrag_protected_mmr": {"selector": "protected_mmr"},
            "spectrag_anchor_expand": {"selector": "anchor_expand_ce"},
            "spectrag_topk": {"selector": "topk"},
            "spectrag_heat": {"proximity_mode": "heat"},
            "spectrag_no_proximity": {"proximity_mode": "none"},
            "spectrag_no_expansion": {"hops": 0}}


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=1))


def folder_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def step_index():
    stats = {}
    for name, config in CONFIGS.items():
        settings = dict(EMBEDDER, chunk_tokens=config["chunk_tokens"], overlap_tokens=config["overlap_tokens"])
        start = time.time()
        info = SpectRAGIngestion(INDEXES / name, settings).ingest_documents(load_docs())
        stats[name] = dict(info, seconds=round(time.time() - start, 1),
                           index_mb=round(folder_mb(INDEXES / name), 1))
        print(name, stats[name], flush=True)
    save("index_stats.json", stats)


def query_vectors(search, questions):
    """Embed every question once (cached), so all methods share the same query vectors."""
    path = INDEXES / "query_vectors.npy"
    if not path.is_file():
        texts = [q["question"] for q in load_questions()]
        np.save(path, search.embed(texts, kind="query"))
    vectors = np.load(path)
    return [vectors[q["id"]] for q in questions]


def with_gold(search, questions):
    items, fallbacks, by_doc = [], 0, chunks_by_doc(search.index["chunks"])
    for q in questions:
        gold, missed = gold_chunks(search.index["chunks"], q["evidence"], by_doc)
        items.append(dict(q, gold=gold))
        fallbacks += missed
    return items, fallbacks


def step_tune():
    validation, _ = split(load_questions())
    for name, config in CONFIGS.items():
        search = SpectRAGRetrieval(INDEXES / name)
        items, _ = with_gold(search, validation)
        start = time.time()
        results = tune(search, items, GRID, objective="recall", budget=config["budget"],
                       trials=TRIALS, seed=0)
        default = tune(search, items, {"alpha": [search.settings["alpha"]]}, budget=config["budget"])[0]
        save("tuning_%s.json" % name, {"seconds": round(time.time() - start, 1), "default": default,
                                       "trials": results})
        print(name, "best", results[0], "default recall", default["recall"], flush=True)


def evaluate_config(name, config, test):
    search = SpectRAGRetrieval(INDEXES / name)
    memory_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    tuned = json.loads((OUT / ("tuning_%s.json" % name)).read_text())["trials"][0]["settings"]
    items, fallbacks = with_gold(search, test)
    vectors = query_vectors(search, test)
    methods = {b: b for b in BASELINES}
    methods.update({v: ("spectrag", o) for v, o in VARIANTS.items()})
    methods["spectrag_tuned"] = ("spectrag", tuned)
    rows = {m: [] for m in methods}
    for method, spec in methods.items():
        if method == "spectrag_tuned" and any(k in tuned for k in ("edge_threshold", "graph_k")):
            search.rebuild_graph({k: tuned[k] for k in ("edge_threshold", "graph_k") if k in tuned})
        overrides = spec if isinstance(spec, str) else ("spectrag", {k: v for k, v in spec[1].items()
                                                                     if k not in ("edge_threshold", "graph_k")})
        for item, vector in zip(items, vectors):
            ids, seconds = run(search, overrides, item["question"], vector, config["budget"])
            rows[method].append(dict(metrics(ids, item["gold"], search.index["chunks"]),
                                     ms=1000 * seconds, type=item["type"], id=item["id"]))
        print(name, method, {k: round(v, 3) for k, v in average(rows[method]).items()
                             if k in ("recall", "complete", "recall@5", "mrr", "ms")}, flush=True)
    return {"rows": rows, "fallbacks": fallbacks, "tuned": tuned, "memory_mb_after_load": memory_after_load,
            "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}


def step_evaluate():
    _, test = split(load_questions())
    for name, config in CONFIGS.items():
        result = evaluate_config(name, config, test)
        save("retrieval_%s.json" % name, result)


def step_latency(repeats=3, size=300):
    """Clean timing run (nothing else should use the machine): per-query median and p95 latency
    of every method, query embedding excluded (it is the same for all methods), plus memory."""
    _, test = split(load_questions())
    for name, config in CONFIGS.items():
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        search = SpectRAGRetrieval(INDEXES / name)
        rss_loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        tuned = json.loads((OUT / ("tuning_%s.json" % name)).read_text())["trials"][0]["settings"]
        items = test[:size]
        vectors = query_vectors(search, items)
        embed_start = time.perf_counter()
        search.embed([q["question"] for q in items[:50]], kind="query")
        embed_ms = 1000 * (time.perf_counter() - embed_start) / 50
        methods = {b: b for b in BASELINES}
        methods.update({"spectrag": ("spectrag", {}),
                        "spectrag_tuned": ("spectrag", {k: v for k, v in tuned.items()
                                                        if k not in ("edge_threshold", "graph_k")})})
        timings = {}
        for method, spec in methods.items():
            times = [run(search, spec, q["question"], v, config["budget"])[1]
                     for _ in range(repeats) for q, v in zip(items, vectors)]
            timings[method] = {"median_ms": 1000 * float(np.median(times)),
                               "p95_ms": 1000 * float(np.percentile(times, 95))}
            print(name, method, timings[method], flush=True)
        save("latency_%s.json" % name, {"questions": size, "repeats": repeats, "timings": timings,
                                        "query_embedding_ms": embed_ms,
                                        "memory_mb": {"before_load": rss_before, "after_load": rss_loaded},
                                        "index_mb": folder_mb(INDEXES / name)})


STEPS = {"index": step_index, "tune": step_tune, "evaluate": step_evaluate, "latency": step_latency}

if __name__ == "__main__":
    for step in sys.argv[1:] or list(STEPS):
        STEPS[step]()
