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


def halve_until_fits(text, max_tokens):
    """Cut text in half between words until every piece fits (exact token counts)."""
    words = text.split()
    if len(words) < 2 or count_tokens(text) <= max_tokens:
        return [text]
    middle = len(words) // 2
    return (halve_until_fits(" ".join(words[:middle]), max_tokens)
            + halve_until_fits(" ".join(words[middle:]), max_tokens))


def split_long_unit(unit, max_tokens):
    """Last resort for one 'sentence' longer than a whole chunk (common in PDF text that has
    no full stops): cut it between words.

    Words are added while their token counts fit; the exact count of each piece is then
    checked, because tokens of joined words can differ slightly from the sum of the parts.
    """
    pieces = []
    words = []
    size = 0
    for word in unit.split():
        word_size = count_tokens(" " + word)
        if words and size + word_size > max_tokens:
            pieces.append(" ".join(words))
            words, size = [], 0
        words.append(word)
        size += word_size
    if words:
        pieces.append(" ".join(words))
    return [part for piece in pieces for part in halve_until_fits(piece, max_tokens)]


def fit_units(units, max_tokens):
    """Make sure no single unit is longer than max_tokens."""
    fitted = []
    for unit in units:
        if count_tokens(unit) > max_tokens:
            fitted.extend(split_long_unit(unit, max_tokens))
        else:
            fitted.append(unit)
    return fitted


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
