"""Edge signals that come from document structure (paper Section 3.2.1).

adjacency - consecutive chunks of the same document
section   - chunks in the same section, sibling sections, or parent and child sections
citation  - a chunk refers to a section number or to another document by file name
"""

from spectrafuse.utils.edge_list import add_edge, as_list
from spectrafuse.utils.references import section_numbers


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


def group_by_doc(chunks):
    """{doc_id: [chunks of that document]}."""
    groups = {}
    for chunk in chunks:
        groups.setdefault(chunk["doc_id"], []).append(chunk)
    return groups


def section_edges(chunks):
    """Link chunks of the same document whose section paths are closely related."""
    edges = {}
    for group in group_by_doc(chunks).values():  # only compare inside one document
        for a_index, a in enumerate(group):
            for b in group[a_index + 1:]:
                add_edge(edges, a["id"], b["id"], section_weight(a["section"], b["section"]))
    return as_list(edges)


def reference_lookups(chunks):
    """Two lookup tables, built once: (doc_id, section number) -> ids, file name -> chunks."""
    by_section = {}
    by_file = {}
    for chunk in chunks:
        for number in section_numbers(chunk["section"]):
            by_section.setdefault((chunk["doc_id"], number), []).append(chunk["id"])
        by_file.setdefault(chunk["doc_id"].split("/")[-1], []).append(chunk)
    return by_section, by_file


def reference_targets(chunk, ref, by_section, by_file):
    """The chunk ids a single reference points to."""
    if ref["type"] == "section":
        return by_section.get((chunk["doc_id"], ref["target"]), [])
    return [c["id"] for c in by_file.get(ref["target"], []) if c["doc_id"] != chunk["doc_id"]]


def citation_edges(chunks):
    """Weight 1.0 from a chunk to every chunk its references point to."""
    by_section, by_file = reference_lookups(chunks)
    edges = {}
    for chunk in chunks:
        for ref in chunk.get("references", []):
            for target in reference_targets(chunk, ref, by_section, by_file):
                add_edge(edges, chunk["id"], target, 1.0)
    return as_list(edges)
