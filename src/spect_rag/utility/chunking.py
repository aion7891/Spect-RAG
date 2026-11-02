"""Sentence-aware structural chunking (paper Section 3.3).

Unstructured text optionally uses Phase 3 embedding-based semantic breakpoints.
Without an embedding callback, the Phase 2 sentence-window behavior is preserved.
"""

from functools import lru_cache
import re

import tiktoken

from .semantic_chunking import semantic_segments


@lru_cache(maxsize=1)
def _tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count literal document text, allowing strings resembling special tokens."""
    return len(_tokenizer().encode(text, disallowed_special=()))


def split_sentences(text: str) -> list[str]:
    """Split at punctuation, protecting common abbreviations and initials.

    This lightweight rule is intended for English prose. It is not a linguistic
    parser; ambiguous abbreviations, equations, and other languages need review.
    Whitespace inside a returned sentence is preserved.
    """
    abbreviations = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.",
                     "etc.", "e.g.", "i.e.", "fig.", "eq.", "sec.", "no.", "st.", "al."}
    sentences = []
    start = 0
    for match in re.finditer(r"""[.!?]["'”’)\]]*\s+""", text):
        prefix = text[start:match.start() + 1]
        token = prefix.split()[-1]
        if token.lower() in abbreviations or re.fullmatch(r"(?:[A-Za-z]\.)+", token):
            continue
        sentences.append(text[start:match.end()].strip())
        start = match.end()
    if remainder := text[start:].strip():
        sentences.append(remainder)
    return sentences


def _units(document: dict) -> list[dict]:
    """Make indivisible sentences/table rows and retain block provenance."""
    units = []
    for block in document["blocks"]:
        if block["kind"] == "table":
            rows = block["text"].splitlines()
            texts = ["\n".join(rows[:2]), *rows[2:]]
        else:
            texts = split_sentences(block["text"])
        for text in texts:
            if text.strip():
                unit = {
                    **block, "text": text,
                    "pages": [block["page"]] if block["page"] is not None else [],
                    "block_indices": [block["block_index"]],
                }
                # A page boundary is not a sentence boundary. Join a dangling
                # PDF sentence to its continuation, retaining both source pages.
                if (
                    units and unit["kind"] == units[-1]["kind"] == "text"
                    and unit["pages"] and units[-1]["pages"]
                    and unit["pages"][0] != units[-1]["pages"][-1]
                    and unit["section_path"] == units[-1]["section_path"]
                    and not re.search(r"""[.!?]["'”’)\]]*$""", units[-1]["text"])
                ):
                    units[-1]["text"] += "\n" + text
                    units[-1]["pages"].extend(unit["pages"])
                    units[-1]["block_indices"].extend(unit["block_indices"])
                else:
                    units.append(unit)
    # Unique positions distinguish repeated identical sentences during merging.
    for position, unit in enumerate(units):
        unit["unit_index"] = position
    return units


def _join(units: list[dict]) -> str:
    return "\n".join(unit["text"] for unit in units)


def _overlap(window: list[dict], next_unit: dict, target: int, limit: int) -> list[dict]:
    """Take the longest contiguous suffix that fits both overlap and size caps."""
    suffix = []
    for unit in reversed(window):
        candidate = [unit, *suffix]
        if count_tokens(_join(candidate)) > target:
            break
        if count_tokens(_join([*candidate, next_unit])) > limit:
            break
        suffix = candidate
    return suffix


def chunk_document(
    document: dict, *, chunk_size: int = 1_200, chunk_overlap: int = 100,
    min_chunk_tokens: int = 32, sentence_embed_fn=None, embedding_dimension: int = 768,
) -> list[dict]:
    """Pack complete sentences/rows, keeping section boundaries and source IDs.

    A sentence or row larger than chunk_size raises ValueError. Overlap is a
    best-effort suffix of complete units, and never crosses a section boundary.
    Small sections/tails remain small when merging would exceed the hard limit
    or lose a section/semantic boundary. Table headers are not repeated.
    """
    for name, value in (("chunk_size", chunk_size), ("chunk_overlap", chunk_overlap),
                        ("min_chunk_tokens", min_chunk_tokens)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer.")
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")
    if not 0 <= min_chunk_tokens <= chunk_size:
        raise ValueError("Require 0 <= min_chunk_tokens <= chunk_size.")

    units = _units(document)
    # Reject oversized inputs before making any optional sentence-model calls.
    for unit in units:
        size = count_tokens(unit["text"])
        if size > chunk_size:
            raise ValueError(
                f"{document['source']}, block {unit['block_index']}, pages {unit['pages']}: "
                f"a sentence/table row has {size} tokens, exceeding chunk_size={chunk_size}. "
                "Increase chunk_size or preprocess this block; text was not truncated."
            )
    semantic = sentence_embed_fn is not None and not any(unit["section_path"] for unit in units)
    segments = semantic_segments(units, sentence_embed_fn, embedding_dimension) if semantic else [None] * len(units)
    for unit, segment in zip(units, segments):
        unit["semantic_segment"] = segment

    windows: list[list[dict]] = []
    window: list[dict] = []
    for unit in units:
        if window and (
            unit["section_path"] != window[-1]["section_path"]
            or unit["semantic_segment"] != window[-1]["semantic_segment"]
        ):
            windows.append(window)
            window = []
        if window and count_tokens(_join([*window, unit])) > chunk_size:
            windows.append(window)
            window = _overlap(window, unit, chunk_overlap, chunk_size)
        window.append(unit)
    if window:
        windows.append(window)

    # Try to merge a small final window without duplicating its overlap.
    if len(windows) > 1 and count_tokens(_join(windows[-1])) < min_chunk_tokens:
        previous, tail = windows[-2:]
        if (
            previous[-1]["section_path"] == tail[0]["section_path"]
            and previous[-1]["semantic_segment"] == tail[0]["semantic_segment"]
        ):
            overlap = 0
            for length in range(1, min(len(previous), len(tail)) + 1):
                if previous[-length:] == tail[:length]:
                    overlap = length
            merged = previous + tail[overlap:]
            if count_tokens(_join(merged)) <= chunk_size:
                windows[-2:] = [merged]

    chunks = []
    for position, units in enumerate(windows):
        text = _join(units)
        pages = sorted({page for unit in units for page in unit["pages"]})
        chunks.append({
            "chunk_id": f"{document['doc_id']}:{position:05d}",
            "doc_id": document["doc_id"],
            "source": document["source"],
            "content_hash": document["content_hash"],
            "position": position,
            "text": text,
            "token_count": count_tokens(text),
            "pages": pages,
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
            "section_path": list(units[0]["section_path"]),
            "block_indices": list(dict.fromkeys(
                index for unit in units for index in unit["block_indices"]
            )),
            "contains_table": any(unit["kind"] == "table" for unit in units),
            "chunking_mode": "semantic" if semantic else (
                "structural" if units[0]["section_path"] else "sentence"
            ),
            "semantic_segment": units[0]["semantic_segment"],
        })
    return chunks
