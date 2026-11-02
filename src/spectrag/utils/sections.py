"""Find headings and split a structured document into sections (paper Section 3.3)."""

import re

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z].*)$")


def is_numbered_heading(line):
    """True for short title lines like '2.1 Datasets' (not list items or sentences)."""
    match = NUMBERED_HEADING.match(line)
    if not match:
        return False
    title = match.group(2)
    return len(title.split()) <= 10 and not title.endswith((".", ":", ",", ";"))


def is_heading(line, markdown_only=False):
    """Return (level, title) if the line is a heading, otherwise None.

    In Markdown files only '#' headings count, so numbered lists ('1. Install') stay text.
    """
    line = line.strip()
    match = MARKDOWN_HEADING.match(line)
    if match:
        return len(match.group(1)), match.group(2)
    if not markdown_only and is_numbered_heading(line):
        number = NUMBERED_HEADING.match(line).group(1)
        return number.count(".") + 1, line  # '2.1' -> level 2
    return None


def heading_lines(text):
    """Return every heading found outside ``` code blocks."""
    found = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        elif not in_code and is_heading(line):
            found.append(line.strip())
    return found


def has_markdown_headings(text):
    """True if the text has at least one '#' heading."""
    return any(line.startswith("#") for line in heading_lines(text))


def has_structure(text):
    """Use structural mode for any Markdown heading, or at least two numbered headings."""
    return has_markdown_headings(text) or len(heading_lines(text)) >= 2


def add_section(sections, path, heading, body_lines, carried):
    """Save a finished section. Returns the heading lines still waiting for a body.

    A heading with no text of its own (e.g. '## 2. Products') is carried forward and put
    in front of the next section, so no line of the document is ever lost.
    """
    heading_text = [line for line in carried + [heading] if line]
    body = "\n".join(body_lines).strip()
    if not body:
        return heading_text
    titles = [title for _, title in path]
    sections.append({"section": titles, "heading": "\n".join(heading_text), "body": body})
    return []


def split_by_sections(text):
    """Split text into [{'section': [path], 'heading': lines, 'body': text}, ...]."""
    markdown_only = has_markdown_headings(text)
    sections = []
    path = []        # list of (level, title) from the top heading down
    heading = ""
    body = []
    carried = []     # headings that had no body yet
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        found = None if in_code else is_heading(line, markdown_only)
        if found:
            carried = add_section(sections, path, heading, body, carried)
            level, title = found
            # Drop headings at the same or deeper level, then add the new one.
            path = [item for item in path if item[0] < level] + [(level, title)]
            heading = line.strip()
            body = []
        else:
            body.append(line)
    carried = add_section(sections, path, heading, body, carried)
    if carried:  # headings at the very end with no text after them
        if sections:
            sections[-1]["body"] += "\n" + "\n".join(carried)
        else:
            sections.append({"section": [], "heading": "", "body": "\n".join(carried)})
    return sections
