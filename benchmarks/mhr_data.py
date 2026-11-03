"""MultiHopRAG data for the benchmark: documents, questions, a fixed split, and gold chunks.

The dataset is stored in benchmarks/data/multihoprag/ (see SOURCE.md there).
"""

import json
import random
from pathlib import Path

DATA = Path(__file__).parent / "data" / "multihoprag"
VALIDATION_SIZE = 300  # questions used for tuning; all other questions are the test set
SPLIT_SEED = 0


def load_docs():
    """609 articles as SpectRAG documents; doc_id "000".."608", text = title + body."""
    corpus = json.loads((DATA / "corpus.json").read_text())
    return [{"doc_id": "%03d" % i, "text": d["title"] + "\n\n" + d["body"]} for i, d in enumerate(corpus)]


def load_questions():
    """All 2,556 questions with the doc_id of every evidence fact."""
    corpus = json.loads((DATA / "corpus.json").read_text())
    doc_of_title = {d["title"]: "%03d" % i for i, d in enumerate(corpus)}
    questions = []
    for number, item in enumerate(json.loads((DATA / "MultiHopRAG.json").read_text())):
        evidence = [{"doc_id": doc_of_title[e["title"]], "fact": e["fact"]} for e in item["evidence_list"]]
        questions.append({"id": number, "question": item["query"], "answers": [item["answer"]],
                          "type": item["question_type"].replace("_query", ""), "evidence": evidence})
    return questions


def split(questions):
    """Fixed split of the questions that have evidence: (validation, test)."""
    with_evidence = [q for q in questions if q["evidence"]]
    order = list(range(len(with_evidence)))
    random.Random(SPLIT_SEED).shuffle(order)
    chosen = set(order[:VALIDATION_SIZE])
    validation = [q for i, q in enumerate(with_evidence) if i in chosen]
    test = [q for i, q in enumerate(with_evidence) if i not in chosen]
    return validation, test


def normalise(text):
    return " ".join(text.split())


def chunks_by_doc(chunks):
    """{doc_id: [(chunk id, normalised text), ...]}, built once per index."""
    by_doc = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], []).append((chunk["id"], normalise(chunk["text"])))
    return by_doc


def gold_chunks(chunks, evidence, by_doc=None):
    """{evidence number: set of chunk ids holding that fact} and the number of fallbacks.

    A fact counts as found in a chunk of its own article that contains it word for word. A
    fact that crosses a chunk boundary maps to the chunks holding its first or second half.
    """
    by_doc = by_doc or chunks_by_doc(chunks)
    gold, fallbacks = {}, 0
    for number, item in enumerate(evidence):
        fact, texts = normalise(item["fact"]), by_doc[item["doc_id"]]
        ids = {cid for cid, text in texts if fact in text}
        if not ids:
            fallbacks += 1
            words = fact.split()
            halves = [" ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])]
            ids = {cid for cid, text in texts if any(half and half in text for half in halves)}
        gold[number] = ids
    return gold, fallbacks


def lightrag_subset(test, articles=25, questions=150, seed=0):
    """A small corpus for the LightRAG comparison (LightRAG needs an LLM call per chunk).

    Articles are chosen greedily so that as many test questions as possible have ALL their
    evidence inside the subset; then 'questions' of those are sampled, stratified by type.
    Returns (sorted doc ids, sampled questions).
    """
    chosen, remaining = set(), list(test)
    while remaining:
        best = min(remaining, key=lambda q: (len({e["doc_id"] for e in q["evidence"]} - chosen), q["id"]))
        new = {e["doc_id"] for e in best["evidence"]} - chosen
        if len(chosen) + len(new) > articles:
            break
        chosen |= new
        remaining.remove(best)
    covered = [q for q in test if {e["doc_id"] for e in q["evidence"]} <= chosen]
    rng, sample = random.Random(seed), []
    for kind in sorted({q["type"] for q in covered}):
        group = [q for q in covered if q["type"] == kind]
        share = round(questions * len(group) / len(covered))
        sample += rng.sample(group, min(share, len(group)))
    return sorted(chosen), sorted(sample, key=lambda q: q["id"])
