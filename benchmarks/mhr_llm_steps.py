"""The MultiHopRAG steps that need an LLM (gemma3:4b through Ollama).

Run:  python benchmarks/mhr_llm_steps.py hyde|answers|lightrag
  hyde      HyDE vs dense vs SpectRAG retrieval on a fixed test sample
  answers   answer accuracy (EM, token F1, contains) of dense, hybrid and SpectRAG contexts
  lightrag  scores the contexts produced by benchmarks/lightrag_runner.py on the subset
Sample sizes are small in quick mode; set SPECTRAG_BENCH_FULL=1 for bigger samples.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mhr_data import load_docs, load_questions, normalise, split  # noqa: E402
from mhr_eval import average, metrics, run  # noqa: E402
from multihoprag_benchmark import FULL, INDEXES, OUT, query_vectors, save, with_gold  # noqa: E402
from spectrag import SpectRAGIngestion, SpectRAGRetrieval  # noqa: E402
from spectrag.utils.evaluation import prompt_for  # noqa: E402
from spectrag.utils.llm import ollama_llm  # noqa: E402
from spectrag.utils.metrics import contains, exact_match, token_f1  # noqa: E402

MODEL = "gemma3:4b"
CONFIG, BUDGET = "chunk1200", 12000
HYDE_SIZE, ANSWER_SIZE = (300, 150) if FULL else (60, 30)
HYDE_PROMPT = "Write a short news passage (3 sentences) that answers this question.\n\nQuestion: {q}\nPassage:"


def sample(test, size, seed=1):
    return sorted(random.Random(seed).sample(test, size), key=lambda q: q["id"])


def step_hyde():
    _, test = split(load_questions())
    search = SpectRAGRetrieval(INDEXES / CONFIG)
    items, _ = with_gold(search, sample(test, HYDE_SIZE))
    vectors, llm = query_vectors(search, items), ollama_llm(MODEL, max_tokens=160)
    rows, start = {"dense": [], "hyde": [], "spectrag": []}, time.time()
    for item, vector in zip(items, vectors):
        passage = llm(HYDE_PROMPT.format(q=item["question"]))
        hyde_vector = search.embed([passage], kind="document")[0]  # HyDE embeds it as a document
        for method, spec, vec in [("dense", "dense", vector), ("hyde", "hyde", hyde_vector),
                                  ("spectrag", ("spectrag", {}), vector)]:
            ids, seconds = run(search, spec, item["question"], vec, BUDGET)
            rows[method].append(dict(metrics(ids, item["gold"], search.index["chunks"]), ms=1000 * seconds))
    save("hyde_%s.json" % CONFIG, {"questions": len(items), "llm_seconds": round(time.time() - start, 1),
                                   "summary": {m: average(r) for m, r in rows.items()}})
    print({m: round(average(r)["recall"], 3) for m, r in rows.items()})


def step_answers():
    _, test = split(load_questions())
    search = SpectRAGRetrieval(INDEXES / CONFIG)
    tuned = json.loads((OUT / ("tuning_%s.json" % CONFIG)).read_text())["trials"][0]["settings"]
    retrieval_only = {k: v for k, v in tuned.items() if k not in ("edge_threshold", "graph_k")}
    items, llm = sample(test, ANSWER_SIZE, seed=2), ollama_llm(MODEL, max_tokens=40)
    vectors, rows, start = query_vectors(search, items), [], time.time()
    for item, vector in zip(items, vectors):
        row = {"id": item["id"], "type": item["type"], "gold": item["answers"][0]}
        for method, spec in [("dense", "dense"), ("hybrid", "hybrid"), ("spectrag", ("spectrag", {})),
                             ("spectrag_tuned", ("spectrag", retrieval_only))]:
            ids, _ = run(search, spec, item["question"], vector, BUDGET)
            reply = llm(prompt_for(search, item["question"], ids))
            row[method] = {"answer": reply, "em": exact_match(reply, item["answers"]),
                           "f1": token_f1(reply, item["answers"]), "contains": contains(reply, item["answers"])}
        rows.append(row)
        print(len(rows), "/", len(items), flush=True)
    methods = ["dense", "hybrid", "spectrag", "spectrag_tuned"]
    summary = {m: {k: sum(r[m][k] for r in rows) / len(rows) for k in ("em", "f1", "contains")} for m in methods}
    save("answers_%s.json" % CONFIG, {"model": MODEL, "questions": len(rows), "summary": summary,
                                      "seconds": round(time.time() - start, 1), "rows": rows})
    print(summary)


def evidence_in_context(context, evidence):
    """Share of evidence facts that appear word for word in a context (any method)."""
    text = normalise(context)
    found = [1.0 if normalise(e["fact"]) in text else 0.0 for e in evidence]
    return {"recall": sum(found) / len(found), "complete": min(found)}


def step_lightrag(work_dir, budget=2400):
    """LightRAG vs the other methods on the 4-article subset, same budget, same metric."""
    subset = json.loads((OUT / "lightrag_subset.json").read_text())
    questions = {q["id"]: q for q in load_questions()}
    items = [questions[i] for i in subset["question_ids"]]
    docs = [d for d in load_docs() if d["doc_id"] in set(subset["articles"])]
    folder = INDEXES / "lightrag_subset"
    SpectRAGIngestion(folder, {"embedder": "ollama", "chunk_tokens": 1200, "overlap_tokens": 100}).ingest_documents(docs)
    search = SpectRAGRetrieval(folder)
    contexts = {c["id"]: c for c in json.loads((Path(work_dir) / "lightrag_contexts.json").read_text())}
    rows = {m: [] for m in ["dense", "bm25", "hybrid", "spectrag", "lightrag"]}
    for item in items:
        vector = search.embed([item["question"]], kind="query")[0]
        for method in ["dense", "bm25", "hybrid", "spectrag"]:
            spec = ("spectrag", {}) if method == "spectrag" else method
            ids, _ = run(search, spec, item["question"], vector, budget)
            context = "\n".join(search.index["chunks"][i]["text"] for i in ids)
            rows[method].append(evidence_in_context(context, item["evidence"]))
        rows["lightrag"].append(evidence_in_context(contexts[item["id"]]["context"], item["evidence"]))
    seconds = [c["seconds"] for c in contexts.values()]
    save("lightrag_subset_results.json", {"articles": len(docs), "questions": len(items), "budget": budget,
                                          "lightrag_query_seconds_mean": sum(seconds) / len(seconds),
                                          "summary": {m: average(r) for m, r in rows.items()}})
    print({m: average(r) for m, r in rows.items()})


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "lightrag":
        step_lightrag(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2400)
    else:
        {"hyde": step_hyde, "answers": step_answers}[step]()
