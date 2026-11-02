"""Break text into "units": whole sentences and whole tables.

Chunks are built from units, so a chunk never cuts a sentence or a table row in half.
"""

from spectrag.utils.tokens import count_tokens, split_sentences


def is_table_line(line):
    """Markdown table rows start with '|'."""
    return line.strip().startswith("|")


def is_separator_row(line):
    """The '|---|---|' row under a table header."""
    return set(line.strip()) <= set("|-: ")


def split_table(table, max_tokens):
    """Split a too-large table into row groups. Each group repeats the header."""
    lines = table.splitlines()
    header = lines[:2] if len(lines) > 1 and is_separator_row(lines[1]) else lines[:1]
    groups = []
    current = list(header)
    for row in lines[len(header):]:
        if len(current) > len(header) and count_tokens("\n".join(current + [row])) > max_tokens:
            groups.append("\n".join(current))
            current = list(header)
        current.append(row)
    groups.append("\n".join(current))
    return groups


def add_table(units, table_lines, max_tokens):
    """Add a finished table as one unit, or several if it is too large."""
    table = "\n".join(line.strip() for line in table_lines)
    if count_tokens(table) > max_tokens:
        units.extend(split_table(table, max_tokens))
    else:
        units.append(table)


def split_units(text, max_tokens):
    """Return the text as a list of sentences and tables, in their original order."""
    units = []
    prose = []
    table = []
    for line in text.splitlines():
        if is_table_line(line):
            if prose:
                units.extend(split_sentences("\n".join(prose)))
                prose = []
            table.append(line)
        else:
            if table:
                add_table(units, table, max_tokens)
                table = []
            prose.append(line)
    if table:
        add_table(units, table, max_tokens)
    if prose:
        units.extend(split_sentences("\n".join(prose)))
    return units
