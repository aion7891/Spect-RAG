"""Read source documents into ordered text/table blocks (paper Section 3.3).

Records are plain dictionaries. Page numbers are 1-based for PDFs and None for
text files. A section path is a list of heading titles, from outermost to inner.
"""

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
import re

import pdfplumber


def _table_separator(line: str) -> bool:
    cells = line.strip().strip("|").split("|")
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _heading(line: str, *, numbered: bool) -> tuple[int, str] | None:
    """Recognize Markdown headings, or short numbered PDF section titles."""
    match = re.fullmatch(r" {0,3}(#{1,6})\s+(.+?)\s*#*\s*", line)
    if match:
        return len(match[1]), match[2]
    if numbered:
        match = re.fullmatch(r"(\d+(?:\.\d+)*)(?:\.)?\s+(.+)", line.strip())
        if match and len(match[2].split()) <= 12 and not match[2].endswith((".", "!", "?")):
            return match[1].count(".") + 1, line.strip()
    return None


def _text_blocks(text: str, *, markdown: bool) -> list[dict]:
    """Group paragraphs and Markdown tables, preserving heading paths."""
    blocks = []
    sections: dict[int, str] = {}
    paragraph: list[str] = []
    table: list[str] = []
    fenced = False
    fence_marker = ""

    def flush(lines: list[str], kind: str) -> None:
        if lines:
            blocks.append({
                "text": "\n".join(lines),
                "kind": kind,
                "page": None,
                "section_path": list(sections.values()),
            })
            lines.clear()

    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if markdown and re.match(r"^\s*(\x60{3,}|~{3,})", line):
            flush(table, "table")
            if not fenced:
                flush(paragraph, "text")
                fence_marker = stripped[0]
                fenced = True
            elif stripped.startswith(fence_marker * 3):
                fenced = False
            paragraph.append(line)
            continue
        if fenced:
            paragraph.append(line)
            continue

        heading = _heading(line, numbered=False) if markdown else None
        if heading:
            flush(paragraph, "text")
            flush(table, "table")
            level, title = heading
            sections = {k: v for k, v in sections.items() if k < level}
            sections[level] = title
            paragraph.append(line)
        elif markdown and "|" in line and (
            table
            or (
                i + 1 < len(lines)
                and _table_separator(lines[i + 1])
            )
        ):
            flush(paragraph, "text")
            table.append(line)
        else:
            flush(table, "table")
            if stripped:
                paragraph.append(line)
            else:
                flush(paragraph, "text")
    flush(paragraph, "text")
    flush(table, "table")
    return blocks


def _markdown_table(rows: list[list[str | None]]) -> str:
    """Serialize extracted PDF cells without putting newlines inside a row."""
    width = max(len(row) for row in rows)
    lines = []
    for i, row in enumerate(rows):
        cells = [" ".join((cell or "").split()).replace("|", r"\|") for cell in row]
        cells.extend([""] * (width - len(cells)))
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _pdf_blocks(path: Path) -> list[dict]:
    """Extract PDF lines and ruled tables in page order; reject image-only pages.

    Table detection uses pdfplumber's default line-based strategy. Borderless
    tables and multi-column reading order can require a specialized parser.
    Numbered PDF headings are a heuristic, not a layout/model classification.
    """
    blocks = []
    sections: dict[int, str] = {}
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            if not any(char.get("text", "").strip() for char in page.chars):
                if page.images or page.curves or page.rects or page.lines:
                    raise ValueError(f"{path}, page {number}: no extractable text; OCR is required.")
                continue
            tables = page.find_tables()
            boxes = [table.bbox for table in tables]

            def outside_tables(obj: dict) -> bool:
                x = (obj.get("x0", 0) + obj.get("x1", 0)) / 2
                y = (obj.get("top", 0) + obj.get("bottom", 0)) / 2
                return not any(x0 <= x <= x1 and top <= y <= bottom for x0, top, x1, bottom in boxes)

            lines = page.filter(outside_tables).extract_text_lines(return_chars=False)
            items = [(line["top"], line["x0"], "text", line["text"]) for line in lines]
            for table in tables:
                rows = table.extract()
                if not rows:
                    raise ValueError(f"{path}, page {number}: detected table could not be extracted.")
                items.append((table.bbox[1], table.bbox[0], "table", _markdown_table(rows)))
            items.sort(key=lambda item: (item[0], item[1]))

            # Accumulate lines so wrapped PDF sentences stay intact within a page.
            paragraph: list[str] = []

            def flush() -> None:
                if paragraph:
                    blocks.append({
                        "text": "\n".join(paragraph), "kind": "text",
                        "page": number, "section_path": list(sections.values()),
                    })
                    paragraph.clear()

            for _, _, kind, text in items:
                if kind == "table":
                    flush()
                    blocks.append({
                        "text": text, "kind": kind, "page": number,
                        "section_path": list(sections.values()),
                    })
                elif heading := _heading(text, numbered=True):
                    flush()
                    level, title = heading
                    sections = {k: v for k, v in sections.items() if k < level}
                    sections[level] = title
                    paragraph.append(text)
                else:
                    paragraph.append(text)
            flush()
            page.close()
    return blocks


def load_document(path: str | Path) -> dict:
    """Load one UTF-8 .txt/.md/.markdown or machine-readable .pdf document.

    Identity is derived from the resolved source path. The separate content hash
    detects edits. Extraction fails explicitly for empty files or image-only PDFs.
    """
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Expected a file: {source}")
    suffix = source.suffix.lower()
    if suffix not in {".txt", ".md", ".markdown", ".pdf"}:
        raise ValueError(f"Unsupported document type {suffix!r}: {source}")
    content = source.read_bytes()
    if suffix == ".pdf":
        blocks = _pdf_blocks(source)
    else:
        blocks = _text_blocks(content.decode("utf-8-sig"), markdown=suffix != ".txt")
    if not blocks or not any(block["text"].strip() for block in blocks):
        raise ValueError(f"No extractable text in {source}; scanned PDFs require OCR.")
    for i, block in enumerate(blocks):
        block["block_index"] = i
    return {
        "doc_id": sha256(str(source).encode()).hexdigest()[:16],
        "source": str(source),
        "content_hash": sha256(content).hexdigest(),
        "format": suffix[1:],
        "blocks": blocks,
    }


def load_documents(paths: Iterable[str | Path]) -> list[dict]:
    """Load an explicit file list in caller order, rejecting duplicate sources."""
    if isinstance(paths, (str, Path)):
        raise TypeError("Pass a collection of file paths, for example [path].")
    documents = []
    seen = set()
    for path in paths:
        source = Path(path).expanduser().resolve(strict=True)
        if source in seen:
            raise ValueError(f"Duplicate document path: {source}")
        documents.append(load_document(source))
        seen.add(source)
    if not documents:
        raise ValueError("Provide at least one document path.")
    return documents
