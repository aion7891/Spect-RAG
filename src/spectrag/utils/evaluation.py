"""Compare SpectRAG with plain top-k retrieval on questions with known evidence.

Every method gets the SAME token budget:
  dense    - top-k by dense cosine only (like NaiveRAG)
  hybrid   - top-k by hybrid dense + BM25 relevance
  spectrag - the full pipeline (seeds -> graph -> re-score -> budgeted selection)

A question lists 'evidence' snippets (exact text from the documents). A chunk is gold if it
contains a snippet. Recall is the share of snippets found; "complete" means all were found,
which multi-hop questions need to be answerable.
"""

import json

import numpy as np

from spectrag.utils.metrics import (contains, exact_match, precision_recall_f1, reciprocal_rank,
                                    token_f1)
from spectrag.utils.prompt import build_prompt, format_context

METHODS = ["dense", "hybrid", "spectrag"]


def load_questions(path):
    """Read a JSON list of {"question", "answers", "evidence", "type"}."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def gold_map(chunks, evidence):
    """{snippet: set of chunk ids that contain it}. Fails loudly on a snippet with no chunk."""
    texts = [" ".join(chunk["text"].split()) for chunk in chunks]
    gold = {snippet: {i for i, text in enumerate(texts) if snippet in text} for snippet in evidence}
    missing = [snippet for snippet, ids in gold.items() if not ids]
    if missing:
        raise ValueError("Evidence not found in any chunk: " + str(missing))
    return gold


def score_selection(ids, gold):
    """Precision, recall (share of snippets found), F1, MRR and 'complete' for one ranking."""
    gold_ids = set().union(*gold.values())
    found = sum(1 for chunk_ids in gold.values() if chunk_ids & set(ids))
    precision, _, _ = precision_recall_f1(len(gold_ids & set(ids)), len(ids), len(gold_ids))
    recall = found / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "mrr": reciprocal_rank(ids, gold_ids), "complete": float(found == len(gold))}


def fill_budget(order, chunks, budget):
    """Top-k under a token budget: take chunks in 'order' while they fit."""
    picked, used = [], 0
    for i in order:
        if used + chunks[i]["tokens"] <= budget:
            picked.append(int(i))
            used += chunks[i]["tokens"]
    return picked


def method_rankings(search, question, budget):
    """The ranked chunk ids each method gives for one question, all with the same budget."""
    settings = dict(search.settings, token_budget=budget)
    seeds, scored, picks = search.retrieve(question, settings)
    chunks = search.index["chunks"]
    return {"dense": fill_budget(np.argsort(-seeds["dense"], kind="stable"), chunks, budget),
            "hybrid": fill_budget(np.argsort(-seeds["relevance"], kind="stable"), chunks, budget),
            "spectrag": [scored["candidates"][k] for k, _ in picks]}


def average(rows, method, key):
    return float(np.mean([row[method][key] for row in rows])) if rows else 0.0


def evaluate_retrieval(search, questions, budget):
    """Per-question rows and the average of every metric for every method."""
    rows = []
    for item in questions:
        gold = gold_map(search.index["chunks"], item["evidence"])
        rankings = method_rankings(search, item["question"], budget)
        row = {method: score_selection(ids, gold) for method, ids in rankings.items()}
        row.update({"question": item["question"], "type": item.get("type", ""), "ids": rankings})
        rows.append(row)
    keys = ["precision", "recall", "f1", "mrr", "complete"]
    summary = {m: {k: average(rows, m, k) for k in keys} for m in METHODS}
    return {"rows": rows, "summary": summary}


def prompt_for(search, question, ids, short=True):
    """The grounded prompt built from any list of chunk ids (best first).
    short=True asks for the bare answer, so Exact Match can be measured."""
    chunks = [dict(search.index["chunks"][i], score=1.0 / rank)
              for rank, i in enumerate(ids, start=1)]
    return build_prompt(question, format_context(chunks), short=short)


def evaluate_answers(search, questions, budget, llm_fn):
    """Answer every question with every method's context (same LLM, same short-answer
    prompt); EM, token F1 and contains."""
    rows = []
    for item in questions:
        rankings = method_rankings(search, item["question"], budget)
        row = {"question": item["question"]}
        for method, ids in rankings.items():
            reply = llm_fn(prompt_for(search, item["question"], ids)).strip()
            row[method] = {"answer": reply, "em": exact_match(reply, item["answers"]),
                           "f1": token_f1(reply, item["answers"]),
                           "contains": contains(reply, item["answers"])}
        rows.append(row)
    summary = {m: {k: average(rows, m, k) for k in ["em", "f1", "contains"]} for m in METHODS}
    return {"rows": rows, "summary": summary}
