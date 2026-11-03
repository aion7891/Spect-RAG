"""Turn documents into chunks (paper Section 3.3 and Table 1).

Structural mode: split by headings and keep the section path.
Semantic mode: split where the topic changes.
Both modes pack units up to 'chunk_tokens' with 'overlap_tokens' of overlap.
"""

from spectrag.config import make_settings
from spectrag.utils.sections import has_structure, split_by_sections
from spectrag.utils.semantic_split import split_by_meaning
from spectrag.utils.tokens import count_tokens
from spectrag.utils.units import fit_units, is_table_line, split_units


def take_overlap(chunk, units, sizes, overlap_tokens):
    """Return the last unit numbers of a chunk that fit in the overlap (stops at a table).

    'chunk' is a list of unit numbers. The first unit is never copied, so a chunk is
    never repeated in full.
    """
    overlap = []
    size = 0
    for index in reversed(chunk[1:]):
        size += sizes[index]
        if size > overlap_tokens or is_table_line(units[index]):
            break
        overlap.insert(0, index)
    return overlap


def pack_units(units, max_tokens, overlap_tokens):
    """Pack units into chunk texts of at most max_tokens, with overlap between chunks."""
    units = fit_units(units, max_tokens)
    sizes = [count_tokens(unit) + 1 for unit in units]  # +1 for the joining newline
    chunks = []
    current = []  # unit numbers in the chunk being built
    size = 0
    for index, unit_size in enumerate(sizes):
        if current and size + unit_size > max_tokens:
            chunks.append(current)
            current = take_overlap(current, units, sizes, overlap_tokens)
            size = sum(sizes[i] for i in current)
            if size + unit_size > max_tokens:
                current, size = [], 0  # no room for overlap next to this unit
        current.append(index)
        size += unit_size
    if current:
        chunks.append(current)
    return ["\n".join(units[i] for i in chunk) for chunk in chunks]


def structural_pieces(text, settings):
    """Return (section_path, chunk_text) pairs for a document with headings."""
    pieces = []
    for section in split_by_sections(text):
        units = split_units(section["body"], settings["chunk_tokens"])
        if section["heading"]:
            units.insert(0, section["heading"])
        for chunk_text in pack_units(units, settings["chunk_tokens"], settings["overlap_tokens"]):
            pieces.append((section["section"], chunk_text))
    return pieces


def semantic_pieces(text, settings, embed_fn=None):
    """Return (section_path, chunk_text) pairs for a document without headings."""
    pieces = []
    units = split_units(text, settings["chunk_tokens"])
    for segment in split_by_meaning(units, settings["semantic_percentile"], embed_fn):
        for chunk_text in pack_units(segment, settings["chunk_tokens"], settings["overlap_tokens"]):
            pieces.append(([], chunk_text))
    return pieces


def merge_small_chunks(chunks, min_tokens, max_tokens):
    """Join chunks shorter than min_tokens onto the previous chunk, if the result still fits."""
    merged = []
    for chunk in chunks:
        if merged and chunk["tokens"] < min_tokens:
            text = merged[-1]["text"] + "\n" + chunk["text"]
            tokens = count_tokens(text)  # exact count of the joined text
            if tokens <= max_tokens:
                merged[-1]["text"], merged[-1]["tokens"] = text, tokens
                continue
        merged.append(chunk)
    return merged


def chunk_document(doc, settings, embed_fn=None):
    """Split one document into chunk dicts."""
    if has_structure(doc["text"]):
        mode, pieces = "structural", structural_pieces(doc["text"], settings)
    else:
        mode, pieces = "semantic", semantic_pieces(doc["text"], settings, embed_fn)
    chunks = []
    for section, text in pieces:
        chunks.append({"doc_id": doc["doc_id"], "section": section, "mode": mode,
                       "text": text, "tokens": count_tokens(text)})
    chunks = merge_small_chunks(chunks, settings["min_chunk_tokens"], settings["chunk_tokens"])
    for position, chunk in enumerate(chunks):
        chunk["position"] = position  # order inside the document (used for adjacency edges)
    return chunks


def chunk_documents(docs, settings=None, embed_fn=None):
    """Split all documents. Every chunk gets a global integer 'id'."""
    settings = settings or make_settings()
    all_chunks = []
    for doc in docs:
        for chunk in chunk_document(doc, settings, embed_fn):
            chunk["id"] = len(all_chunks)
            all_chunks.append(chunk)
    return all_chunks
