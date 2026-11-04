"""The `spectrafuse` command.

    spectrafuse ingest docs/ --index my_index [--set chunk_tokens=256 --set embedder=ollama]
    spectrafuse search "my question" --index my_index [--budget 4000] [--json]
    spectrafuse answer "my question" --index my_index --model gemma3:4b
    spectrafuse info --index my_index
    spectrafuse graph --index my_index --out graph.html [--question "my question"]
"""

import argparse
import json
import sys

from spectrafuse._version import __version__
from spectrafuse.errors import SpectRAGError
from spectrafuse.log import enable_logging


def parse_sets(pairs):
    """["alpha=0.5", "selector=mmr"] -> {"alpha": 0.5, "selector": "mmr"} (values read as JSON)."""
    settings = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit("--set needs key=value, got %r" % pair)
        key, value = pair.split("=", 1)
        try:
            settings[key] = json.loads(value)
        except ValueError:
            settings[key] = value  # plain text, e.g. selector=mmr
    return settings


def command_ingest(args):
    from spectrafuse import SpectRAGIngestion
    info = SpectRAGIngestion(args.index, parse_sets(args.set)).ingest(args.path)
    print(json.dumps(info, indent=1))


def command_search(args):
    from spectrafuse import SpectRAGRetrieval
    result = SpectRAGRetrieval(args.index, parse_sets(args.set)).search(args.question, args.budget)
    if args.json:
        print(json.dumps({key: result[key] for key in ("question", "tokens", "chunks", "context")}, indent=1))
        return
    for chunk in result["chunks"]:
        print("%2d. %-30s score %.3f  %d tokens  (%s)" % (chunk["rank"], chunk["doc_id"], chunk["score"],
                                                         chunk["tokens"], chunk["reason"]))
    print("\n" + result["context"])


def command_answer(args):
    from spectrafuse import SpectRAGRetrieval
    from spectrafuse.utils.llm import ollama_llm
    search = SpectRAGRetrieval(args.index, parse_sets(args.set))
    result = search.answer(args.question, ollama_llm(args.model), args.budget)
    print(result["answer"])
    print("\nSources: " + ", ".join(sorted({c["doc_id"] for c in result["chunks"]})))


def command_info(args):
    from spectrafuse.utils.store import check_index_folder, load_index
    from pathlib import Path
    check_index_folder(Path(args.index))
    print(json.dumps(load_index(args.index, only=("info",))["info"], indent=1))


def command_graph(args):
    from spectrafuse.utils.graph_view import save_graph_view
    path = save_graph_view(args.index, args.out, args.question, parse_sets(args.set), args.max_nodes)
    print("Graph view written to %s (open it in a browser)" % path)


def build_parser():
    parser = argparse.ArgumentParser(prog="spectrafuse", description="SpectraFuse: spectral graph RAG, an independent implementation inspired by SpectRAG (arXiv:2610.36991). Official SpectRAG: github.com/SightAndCite/SpectRAG")
    parser.add_argument("--version", action="version", version="spectrafuse " + __version__)
    parser.add_argument("-v", "--verbose", action="store_true", help="print progress messages")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="build an index from a file or folder")
    ingest.add_argument("path")
    for name, helptext in [("search", "find the context for a question"), ("answer", "answer with an LLM")]:
        sub = commands.add_parser(name, help=helptext)
        sub.add_argument("question")
        sub.add_argument("--budget", type=int, default=None, help="token budget (default 12000)")
        if name == "search":
            sub.add_argument("--json", action="store_true", help="print JSON")
        else:
            sub.add_argument("--model", default="gemma3:4b", help="Ollama model")
    info = commands.add_parser("info", help="show what an index contains")
    graph = commands.add_parser("graph", help="write an interactive graph view (HTML)")
    graph.add_argument("--out", default="graph.html", help="HTML file to write")
    graph.add_argument("--question", default=None, help="show this question's retrieval path")
    graph.add_argument("--max-nodes", type=int, default=1500, dest="max_nodes")
    for sub in (ingest, commands.choices["search"], commands.choices["answer"], info, graph):
        sub.add_argument("--index", required=True, help="index folder")
    for sub in (ingest, commands.choices["search"], commands.choices["answer"], graph):
        sub.add_argument("--set", action="append", metavar="KEY=VALUE", help="change a setting")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verbose:
        enable_logging()
    handlers = {"ingest": command_ingest, "search": command_search, "answer": command_answer,
                "info": command_info, "graph": command_graph}
    try:
        handlers[args.command](args)
    except SpectRAGError as error:
        print("spectrafuse: error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
