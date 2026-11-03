from spectrag.utils.tokens import count_tokens
from spectrag.utils.units import split_table, split_units

TABLE = "| Name | Year |\n|------|------|\n| S1 | 2017 |\n| S2 | 2020 |"


def test_sentences_and_table_keep_order():
    text = "Before the table. Second line.\n\n" + TABLE + "\n\nAfter the table."
    assert split_units(text, 1200) == [
        "Before the table.",
        "Second line.",
        TABLE,
        "After the table.",
    ]


def test_small_table_is_one_unit():
    assert split_units(TABLE, 1200) == [TABLE]


def test_large_table_splits_by_rows_and_repeats_header():
    rows = ["| Row %d | value %d |" % (i, i) for i in range(40)]
    table = "| Name | Value |\n|---|---|\n" + "\n".join(rows)
    groups = split_table(table, 60)
    assert len(groups) > 1
    for group in groups:
        assert group.startswith("| Name | Value |\n|---|---|")
    # Every original row appears exactly once, never cut in half.
    body_rows = [line for g in groups for line in g.splitlines()[2:]]
    assert body_rows == rows


def test_table_groups_respect_limit():
    rows = ["| Row %d | value %d |" % (i, i) for i in range(40)]
    table = "| Name | Value |\n|---|---|\n" + "\n".join(rows)
    for group in split_table(table, 60):
        assert count_tokens(group) <= 60
