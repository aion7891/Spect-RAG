from spectrafuse.utils import tokens
from spectrafuse.utils.tokens import count_tokens, estimate_tokens, split_sentences


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") == 4   # 3 words / 0.75


def test_count_tokens_falls_back_without_tiktoken(monkeypatch):
    monkeypatch.setattr(tokens, "get_encoder", lambda: None)
    assert count_tokens("one two three") == 4


def test_count_tokens_is_positive():
    assert count_tokens("Aurora Labs was founded in 2014.") > 0


def test_split_basic_sentences():
    text = "Aurora Labs makes sensors. It was founded in 2014! Is it big? Yes."
    assert split_sentences(text) == [
        "Aurora Labs makes sensors.",
        "It was founded in 2014!",
        "Is it big?",
        "Yes.",
    ]


def test_abbreviations_do_not_split():
    text = "It was led by Dr. Maya Chen. See Fig. 2 for details."
    assert split_sentences(text) == ["It was led by Dr. Maya Chen.", "See Fig. 2 for details."]


def test_blank_line_ends_sentence_and_whitespace_is_tidied():
    text = "# Heading\n\nFirst   line\ncontinues here.\n\nSecond paragraph."
    assert split_sentences(text) == ["# Heading", "First line continues here.", "Second paragraph."]


def test_decimals_do_not_split():
    assert split_sentences("Version 2.1 is out. Good.") == ["Version 2.1 is out.", "Good."]


def test_accented_capital_starts_a_sentence():
    assert split_sentences("Helios is in Porto. Évora is big.") == ["Helios is in Porto.", "Évora is big."]
