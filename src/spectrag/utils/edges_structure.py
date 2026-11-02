"""Edge signals that come from document structure (paper Section 3.2.1).

adjacency - consecutive chunks of the same document
section   - chunks in the same section, sibling sections, or parent and child sections
citation  - a chunk refers to a section number or to another document by file name
"""

from spectrag.utils.edge_list import add_edge, as_list
from spectrag.utils.references import section_numbers


def adjacency_edges(chunks):
    """Weight 1.0 between chunk n and chunk n+1 of the same document."""
    edges = {}
    for a, b in zip(chunks, chunks[1:]):
        if a["doc_id"] == b["doc_id"] and b["position"] == a["position"] + 1:
            add_edge(edges, a["id"], b["id"], 1.0)
    return as_list(edges)


def shared_prefix(path_a, path_b):
    """How many headings two section paths share from the top."""
    count = 0
    for title_a, title_b in zip(path_a, path_b):
        if title_a != title_b:
            break
        count += 1
    return count


def section_weight(path_a, path_b):
    """1.0 for the same section; shared/longest for siblings or parent-child; else 0."""
    if not path_a or not path_b:
        return 0.0
    longest = max(len(path_a), len(path_b))
    shared = shared_prefix(path_a, path_b)
    if shared == 0 or shared < longest - 1:
        return 0.0  # only direct relatives count, which keeps the graph sparse
    return shared / longest


def section_edges(chunks):
    """Link chunks of the same document whose section paths are closely related."""
    edges = {}
    for a_index, a in enumerate(chunks):
        for b in chunks[a_index + 1:]:
            if a["doc_id"] == b["doc_id"]:
                add_edge(edges, a["id"], b["id"], section_weight(a["section"], b["section"]))
    return as_list(edges)


def reference_targets(chunk, ref, chunks):
    """The chunks a single reference points to."""
    if ref["type"] == "section":
        return [c for c in chunks if c["doc_id"] == chunk["doc_id"]
                and ref["target"] in section_numbers(c["section"])]
    return [c for c in chunks if c["doc_id"].split("/")[-1] == ref["target"]
            and c["doc_id"] != chunk["doc_id"]]


def citation_edges(chunks):
    """Weight 1.0 from a chunk to every chunk its references point to."""
    edges = {}
    for chunk in chunks:
        for ref in chunk.get("references", []):
            for target in reference_targets(chunk, ref, chunks):
                add_edge(edges, chunk["id"], target["id"], 1.0)
    return as_list(edges)
