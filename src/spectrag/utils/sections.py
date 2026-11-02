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


def is_heading(line):
    """Return (level, title) if the line is a heading, otherwise None."""
    line = line.strip()
    match = MARKDOWN_HEADING.match(line)
    if match:
        return len(match.group(1)), match.group(2)
    if is_numbered_heading(line):
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


def has_structure(text):
    """Use structural mode for any Markdown heading, or at least two numbered headings."""
    headings = heading_lines(text)
    if any(line.startswith("#") for line in headings):
        return True
    return len(headings) >= 2


def add_section(sections, path, heading, body_lines):
    """Save a finished section, skipping sections that have only a heading."""
    body = "\n".join(body_lines).strip()
    if body:
        titles = [title for _, title in path]
        sections.append({"section": titles, "heading": heading, "body": body})


def split_by_sections(text):
    """Split text into [{'section': [path], 'heading': line, 'body': text}, ...]."""
    sections = []
    path = []        # list of (level, title) from the top heading down
    heading = ""
    body = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        found = None if in_code else is_heading(line)
        if found:
            add_section(sections, path, heading, body)
            level, title = found
            # Drop headings at the same or deeper level, then add the new one.
            path = [item for item in path if item[0] < level] + [(level, title)]
            heading = line.strip()
            body = []
        else:
            body.append(line)
    add_section(sections, path, heading, body)
    return sections
