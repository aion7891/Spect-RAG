"""Find references inside a chunk, for citation edges (paper Section 3.2.1, "Citation").

Two kinds of references are found:
  section - "Section 2.1", "Sec. 3", "§4.2"        -> {"type": "section", "target": "2.1"}
  doc     - "notes.md", "[report](report.pdf)"     -> {"type": "doc", "target": "notes.md"}
"""

import re

SECTION_REF = re.compile(r"(?:\bSections?|\bSec\.|§)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
FILE_REF = re.compile(r"[\w./-]+\.(?:md|markdown|txt|pdf)\b", re.IGNORECASE)
NUMBER_AT_START = re.compile(r"^(\d+(?:\.\d+)*)")


def extract_references(text):
    """Return the list of references found in the text (no duplicates, in order)."""
    found = []
    for number in SECTION_REF.findall(text):
        found.append({"type": "section", "target": number.rstrip(".")})
    for name in FILE_REF.findall(text):
        found.append({"type": "doc", "target": name.split("/")[-1]})
    unique = []
    for ref in found:
        if ref not in unique:
            unique.append(ref)
    return unique


def add_references(chunks):
    """Store each chunk's references as chunk['references'] (paper Table 1)."""
    for chunk in chunks:
        chunk["references"] = extract_references(chunk["text"])
    return chunks


def section_numbers(section_path):
    """Numbers of the headings in a path: ['Doc', '2. Products', '2.1 Sensors'] -> {'2', '2.1'}."""
    numbers = set()
    for title in section_path:
        match = NUMBER_AT_START.match(title)
        if match:
            numbers.add(match.group(1).rstrip("."))
    return numbers
