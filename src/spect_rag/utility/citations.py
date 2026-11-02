"""Find explicit textual citations and resolve local links within this corpus.

No network requests or linked-file reads are performed. Numeric citations and
external URLs are retained as unresolved evidence, not guessed graph edges.
"""

from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlsplit


def _slug(title: str) -> str:
    title = unicodedata.normalize("NFKC", title).casefold()
    title = re.sub(r"[^\w\s-]", "", title)
    return re.sub(r"\s+", "-", title.strip())


def find_citations(text: str) -> list[dict]:
    """Recognize inline Markdown links, bare HTTP(S) URLs, and numeric brackets."""
    found = []
    link_spans = []
    pattern = r"""\[[^\]]+\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+['"][^)]*['"])?\s*\)"""
    for match in re.finditer(pattern, text):
        found.append((match.start(), "link", match.group(), match[1].strip("<>")))
        link_spans.append(match.span())
    for pattern, kind in (
        (r"https?://[^\s<>\])]+", "url"),
        (r"\[\d+(?:\s*[,;-]\s*\d+)*\]", "numeric"),
    ):
        for match in re.finditer(pattern, text):
            if any(start <= match.start() < end for start, end in link_spans):
                continue
            target = match.group().rstrip(".,;:") if kind == "url" else match.group()
            found.append((match.start(), kind, match.group(), target))
    result = {}
    for _, kind, reference, target in sorted(found):
        result.setdefault((kind, target), {"kind": kind, "reference": reference, "target": target})
    return list(result.values())


def resolve_citations(chunks: list[dict]) -> list[dict]:
    """Attach citation targets without changing the input chunks or their order.

    Local links resolve to all chunks in a target document or section. Unmatched
    fragments stay unresolved. Only simple heading slugs are supported.
    """
    sources = {}
    for chunk in chunks:
        sources.setdefault(chunk["source"], []).append(chunk)
    result = []
    for chunk in chunks:
        citations = []
        for reference in find_citations(chunk["text"]):
            record = {
                **reference, "status": "unresolved", "target_doc_id": None,
                "target_chunk_ids": [], "reason": "target_not_in_corpus",
            }
            try:
                target = urlsplit(reference["target"]) if reference["kind"] != "numeric" else None
                path = None
                if target is not None and not target.scheme and not target.netloc:
                    source = Path(chunk["source"])
                    path = (source.parent / unquote(target.path)).resolve() if target.path else source
            except (ValueError, OSError):
                record["reason"] = "malformed_target"
                citations.append(record)
                continue
            if reference["kind"] == "numeric":
                record["reason"] = "bibliography_not_resolved"
            elif target.scheme or target.netloc:
                record["reason"] = "external_target"
            else:
                candidates = sources.get(str(path), [])
                if candidates:
                    record["target_doc_id"] = candidates[0]["doc_id"]
                    if target.fragment:
                        anchor = unquote(target.fragment).casefold()
                        candidates = [
                            other for other in candidates
                            if any(_slug(title) == anchor for title in other["section_path"])
                        ]
                    if candidates:
                        record.update(
                            status="resolved", reason=None,
                            target_chunk_ids=[other["chunk_id"] for other in candidates],
                        )
                    else:
                        record["reason"] = "fragment_not_found"
            citations.append(record)
        result.append({**chunk, "citations": citations})
    return result
