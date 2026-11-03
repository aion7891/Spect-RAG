"""Interactive view of the chunk graph, like Figure 5 of the paper, as one self-contained HTML
file (no internet needed). It shows chunks as nodes, edges coloured by their strongest signal,
and, for a question, the retrieval path: seeds -> graph expansion -> selected context.

    from spectrag.utils.graph_view import save_graph_view
    save_graph_view("my_index", "graph.html", question="Who founded Aurora Labs?")
"""

import json
from pathlib import Path

import numpy as np

from spectrag.utils.graph_template import TEMPLATE
from spectrag.utils.store import array_to_edges, check_index_folder, load_index

SIGNALS = ["semantic", "section", "entity", "adjacency", "utility", "citation"]


def dominant_signals(index, edges):
    """For every fused edge, the signal that added the most to its weight (alpha_r * w_r)."""
    weights = index["info"].get("weights_used", {})
    best = {}
    for name in SIGNALS:
        for i, j, w in array_to_edges(index.get("signals", {}).get(name, [])):
            value = weights.get(name, 0.0) * w
            if value > best.get((i, j), ("", -1.0))[1]:
                best[(i, j)] = (name, value)
    return [best.get((i, j), ("semantic", 0))[0] for i, j, _ in edges]


def choose_nodes(index, edges, path, max_nodes):
    """All nodes if the graph is small. Otherwise grow outwards from the retrieval path
    (path -> neighbours -> their neighbours ...), then add the best-connected chunks."""
    count = len(index["chunks"])
    if count <= max_nodes:
        return list(range(count))
    linked, degree = {}, np.zeros(count)
    for i, j, w in edges:
        linked.setdefault(i, []).append(j)
        linked.setdefault(j, []).append(i)
        degree[i] += w
        degree[j] += w
    keep, layer = list(dict.fromkeys(path))[:max_nodes], list(dict.fromkeys(path))
    while layer and len(keep) < max_nodes:
        seen = set(keep)
        layer = [n for n in dict.fromkeys(o for node in layer for o in linked.get(node, [])) if n not in seen]
        keep += layer[:max_nodes - len(keep)]
    seen = set(keep)
    keep += [int(n) for n in np.argsort(-degree) if int(n) not in seen][:max_nodes - len(keep)]
    return sorted(keep)


def layout(index, nodes, edges, steps=150, seed=0):
    """2-D positions: start from the spectral coordinates (u2, u3), then a few force steps so
    that linked chunks sit together and others do not overlap (Fruchterman-Reingold)."""
    coords = index["spectral"][nodes]
    rng = np.random.default_rng(seed)
    pos = coords[:, :2] if coords.shape[1] >= 2 else rng.normal(size=(len(nodes), 2))
    pos = (pos - pos.mean(0)) / (pos.std(0) + 1e-9) + 0.01 * rng.normal(size=pos.shape)
    place = {node: k for k, node in enumerate(nodes)}
    pairs = np.array([(place[i], place[j], w) for i, j, w in edges if i in place and j in place] or
                     np.zeros((0, 3)))
    k = 1.0 / np.sqrt(len(nodes))
    for step in range(steps):
        delta = pos[:, None, :] - pos[None, :, :]
        distance = np.linalg.norm(delta, axis=2) + 1e-6
        force = (delta / distance[..., None] * (k * k / distance)[..., None]).sum(1)  # repel
        if len(pairs):
            a, b = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
            pull = (pos[a] - pos[b]) * (np.linalg.norm(pos[a] - pos[b], axis=1) / k * pairs[:, 2])[:, None]
            np.add.at(force, a, -pull)
            np.add.at(force, b, pull)
        force -= 2.0 * k * pos  # gravity: keeps separate pieces from drifting far away
        length = np.linalg.norm(force, axis=1, keepdims=True) + 1e-9
        pos += force / length * np.minimum(length, 0.1 * (1 - step / steps))   # cooling
    low, high = np.percentile(pos, 1, axis=0), np.percentile(pos, 99, axis=0)  # ignore stragglers
    pos = np.clip((pos - low) / (high - low + 1e-9), -0.05, 1.05)
    return (pos * 1000).round(1)


def retrieval_path(index_dir, question, settings):
    """seed / hop / selected state of every chunk for one question, plus the selected list."""
    from spectrag.retrieval import SpectRAGRetrieval
    result = SpectRAGRetrieval(index_dir, settings).search(question)
    state = {c: "hop" for c in result["candidates"]}
    state.update({c: "seed" for c in result["seeds"]})
    state.update({c["id"]: "selected" for c in result["chunks"]})
    picked = [{"id": c["id"], "rank": c["rank"], "reason": c["reason"], "doc": c["doc_id"],
               "score": round(c["score"], 3)} for c in result["chunks"]]
    return state, picked


def graph_data(index_dir, question=None, settings=None, max_nodes=1500):
    """Everything the page needs, as plain Python data (also handy for tests)."""
    folder = Path(index_dir)
    check_index_folder(folder)
    index = load_index(folder)
    edges = array_to_edges(index["edges"])
    state, picked = retrieval_path(folder, question, settings) if question else ({}, [])
    nodes = choose_nodes(index, edges, list(state), max_nodes)
    shown = set(nodes)
    edges = [e for e in edges if e[0] in shown and e[1] in shown]
    signal = dominant_signals(index, edges)
    pos = layout(index, nodes, edges)
    degree = {n: 0.0 for n in nodes}
    for i, j, w in edges:
        degree[i] += w
        degree[j] += w
    chunks = index["chunks"]
    return {"question": question or "", "picked": picked, "signals": SIGNALS,
            "nodes": [{"id": n, "x": float(x), "y": float(y), "doc": chunks[n]["doc_id"],
                       "section": " > ".join(chunks[n].get("section", [])), "cluster": chunks[n].get("cluster", 0),
                       "position": chunks[n]["position"], "degree": round(degree[n], 3),
                       "state": state.get(n, ""), "text": chunks[n]["text"][:400]}
                      for n, (x, y) in zip(nodes, pos)],
            "edges": [[i, j, round(w, 3), s] for (i, j, w), s in zip(edges, signal)],
            "info": {k: index["info"].get(k) for k in ("documents", "chunks", "graph_edges")}}


def save_graph_view(index_dir, path="graph.html", question=None, settings=None, max_nodes=1500):
    """Write the interactive graph view to an HTML file and return its path."""
    data = graph_data(index_dir, question, settings, max_nodes)
    html = TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    Path(path).write_text(html, encoding="utf-8")
    return Path(path)
