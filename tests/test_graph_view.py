"""The interactive graph view (paper Figure 5)."""

import json
import re
from pathlib import Path

import pytest

from spectrafuse import IndexError_, SpectRAGIngestion
from spectrafuse.cli import main
from spectrafuse.utils.graph_view import choose_nodes, graph_data, save_graph_view

EXAMPLES = Path(__file__).parent.parent / "examples"
QUESTION = "Which company acquired the former employer of the person who leads the Madrid office of Aurora Labs?"


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    folder = tmp_path_factory.mktemp("idx")
    SpectRAGIngestion(folder, {"chunk_tokens": 128, "overlap_tokens": 32}).ingest(EXAMPLES / "eval" / "docs")
    return folder


def test_data_has_every_node_edge_and_signal(index):
    data = graph_data(index)
    assert len(data["nodes"]) == data["info"]["chunks"] and len(data["edges"]) == data["info"]["graph_edges"]
    assert {e[3] for e in data["edges"]} <= set(data["signals"])
    for node in data["nodes"]:
        assert -60 <= node["x"] <= 1060 and -60 <= node["y"] <= 1060
    assert data["picked"] == [] and all(n["state"] == "" for n in data["nodes"])


def test_retrieval_path_is_marked(index):
    data = graph_data(index, QUESTION, {"token_budget": 600, "seed_count": 3})
    states = {n["state"] for n in data["nodes"]}
    assert "selected" in states and data["picked"]
    selected = {n["id"] for n in data["nodes"] if n["state"] == "selected"}
    assert selected == {p["id"] for p in data["picked"]}


def test_large_graphs_keep_the_retrieval_path(index):
    data = graph_data(index)
    edges = [[e[0], e[1], e[2]] for e in data["edges"]]
    fake = {"chunks": [{}] * len(data["nodes"])}
    kept = choose_nodes(fake, edges, [5, 9], max_nodes=8)
    assert len(kept) == 8 and {5, 9} <= set(kept)


def test_html_is_self_contained_and_valid(index, tmp_path):
    path = save_graph_view(index, tmp_path / "graph.html", QUESTION)
    html = path.read_text()
    assert "<script src" not in html and "https://" not in html      # no downloads
    data = json.loads(re.search(r"const D = (.*?);\n", html).group(1).replace("<\\/", "</"))
    assert data["question"] == QUESTION and data["nodes"]


def test_cli_graph_and_missing_index(index, tmp_path, capsys):
    assert main(["graph", "--index", str(index), "--out", str(tmp_path / "g.html"), "--question", QUESTION]) == 0
    assert (tmp_path / "g.html").is_file()
    with pytest.raises(IndexError_):
        graph_data(tmp_path / "nothing")
