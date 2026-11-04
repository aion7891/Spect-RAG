"""Release validation: re-check the mathematics and every configuration on a REAL index.

Run:  python tools/validate_release.py [index_dir]
Default index: benchmarks/indexes/chunk1200 (MultiHopRAG, nomic-embed-text; build it with
benchmarks/multihoprag_benchmark.py index). Writes docs/validation/results.json.
Every check prints PASS or FAIL; the script exits with code 1 if any check fails.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tests"))
from mhr_data import load_questions, split  # noqa: E402
from reference import ref_bm25  # noqa: E402
from spectrafuse import SpectRAGRetrieval, make_settings  # noqa: E402
from spectrafuse.utils.bm25 import bm25_scores, build_bm25, tokenize  # noqa: E402
from spectrafuse.utils.diffusion import PROXIMITY_MODES, personalized_pagerank, restart_vector  # noqa: E402
from spectrafuse.utils.expansion import subgraph_matrix  # noqa: E402
from spectrafuse.utils.selectors import SELECTORS  # noqa: E402
from spectrafuse.utils.spectral import inverse_sqrt  # noqa: E402

RESULTS = []


def check(name, passed, **details):
    RESULTS.append(dict(name=name, passed=bool(passed), **details))
    print("%s  %s  %s" % ("PASS" if passed else "FAIL", name, json.dumps(details)[:150]), flush=True)


def laplacian_checks(search):
    """Eq. 11-13 on the real graph: L_sym is PSD with eigenvalues in [0, 2]; stored eigenpairs."""
    from scipy.sparse import coo_matrix, diags, eye
    edges, n = np.asarray(search.index["edges"]), len(search.index["chunks"])
    i, j, w = edges[:, 0].astype(int), edges[:, 1].astype(int), edges[:, 2]
    W = coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    scale = diags(inverse_sqrt(np.asarray(W.sum(1)).ravel()))
    L = eye(n) - scale @ W @ scale
    U, lam = search.index["spectral"], np.asarray(search.index["eigenvalues"])
    check("graph is symmetric with weights in (0, 1]", abs(W - W.T).max() == 0 and w.min() > 0 and w.max() <= 1,
          edges=int(len(w)))
    check("eigenpairs: max ||L u - lambda u||", np.linalg.norm(L @ U - U * lam, axis=0).max() < 1e-8,
          residual=float(np.linalg.norm(L @ U - U * lam, axis=0).max()))
    check("eigenvectors orthonormal", np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8))
    check("eigenvalues in [0, 2] and ascending", lam.min() >= -1e-9 and lam.max() <= 2 and np.all(np.diff(lam) >= -1e-9),
          smallest=float(lam[0]), largest=float(lam[-1]))
    trivial = np.sqrt(np.asarray(W.sum(1)).ravel())
    trivial /= np.linalg.norm(trivial)
    check("coordinates exclude the trivial eigenvector", np.abs(U.T @ trivial).max() < 1e-6)


def pagerank_check(search, question, vector):
    """Eq. 18 on a real candidate subgraph equals its closed form beta (I - (1-beta) A~)^-1 r."""
    seeds, scored, _ = search.retrieve(question, query_vector=vector)
    matrix = subgraph_matrix(search.table, scored["candidates"])
    restart = restart_vector(scored["candidates"], seeds["seeds"], seeds["relevance"])
    scale = inverse_sqrt(matrix.sum(1))
    a_tilde = scale[:, None] * matrix * scale[None, :]
    exact = 0.15 * np.linalg.solve(np.eye(len(matrix)) - 0.85 * a_tilde, restart)
    iterated = personalized_pagerank(matrix, restart, 0.15, steps=400)
    check("PageRank iteration = closed form (real subgraph)", np.allclose(iterated, exact, atol=1e-10),
          candidates=len(matrix), max_error=float(np.abs(iterated - exact).max()))


def bm25_check(search, questions):
    # The reference splits on spaces, so give both the same words (SpectraFuse's tokenizer).
    texts = [" ".join(tokenize(c["text"])) for c in search.index["chunks"][:400]]
    small = build_bm25(texts)
    worst = max(float(np.abs(bm25_scores(small, q["question"]) - ref_bm25(texts, " ".join(tokenize(q["question"])))).max())
                for q in questions[:5])
    check("BM25 = reference formula (400 real chunks)", worst < 1e-9, max_error=worst)


def configuration_checks(search, questions, vectors):
    """Every selector x proximity mode: budget respected, no duplicates, picks are candidates."""
    worst, runs = 0.0, 0
    for selector, mode in itertools.product(SELECTORS, PROXIMITY_MODES):
        for budget in (500, 4000, 12000):
            for q, v in zip(questions[:15], vectors[:15]):
                _, scored, picks = search.retrieve(q["question"], {"selector": selector, "proximity_mode": mode,
                                                                   "token_budget": budget}, query_vector=v)
                ids = [scored["candidates"][k] for k, _ in picks]
                tokens = sum(search.index["chunks"][i]["tokens"] for i in ids)
                assert tokens <= budget and len(set(ids)) == len(ids), (selector, mode, budget)
                worst, runs = max(worst, tokens / budget), runs + 1
    check("all 5 selectors x 5 proximity modes x 3 budgets respect the budget", True, runs=runs,
          highest_budget_use=round(worst, 4))
    bad = 0
    for overrides in [{"alpha": 2}, {"selector": "x"}, {"token_budget": 0}, {"edge_threshold": -1}]:
        try:
            make_settings(overrides)
        except ValueError:
            bad += 1
    check("invalid settings are rejected", bad == 4, rejected=bad)


def reproducibility_check(search, test, vectors):
    """Re-run SpectraFuse (default) on the test set and compare with the committed results."""
    saved = json.loads((ROOT / "benchmarks/results/multihoprag/retrieval_chunk1200.json").read_text())
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from multihoprag_benchmark import with_gold
    from mhr_eval import metrics, run
    items, _ = with_gold(search, test)
    old = {r["id"]: r for r in saved["rows"]["spectrag"]}
    differ = 0
    for item, vector in zip(items, vectors):
        ids, _ = run(search, ("spectrag", {}), item["question"], vector, 12000)
        row = metrics(ids, item["gold"], search.index["chunks"])
        differ += any(abs(row[k] - old[item["id"]][k]) > 1e-12 for k in ("recall", "recall@10", "mrr"))
    check("benchmark reproduces: SpectraFuse on all test questions", differ == 0, questions=len(items),
          different=differ)


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmarks/indexes/chunk1200"
    start = time.time()
    search = SpectRAGRetrieval(folder)
    _, test = split(load_questions())
    vectors = np.load(ROOT / "benchmarks/indexes/query_vectors.npy")
    test_vectors = [vectors[q["id"]] for q in test]
    laplacian_checks(search)
    pagerank_check(search, test[0]["question"], test_vectors[0])
    bm25_check(search, test)
    configuration_checks(search, test, test_vectors)
    reproducibility_check(search, test, test_vectors)
    out = ROOT / "docs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"index": str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
                                                  "seconds": round(time.time() - start, 1), "checks": RESULTS}, indent=1))
    failed = [r["name"] for r in RESULTS if not r["passed"]]
    print("\n%d checks, %d failed%s" % (len(RESULTS), len(failed), (": " + ", ".join(failed)) if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
