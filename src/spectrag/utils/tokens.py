"""Count tokens and split text into sentences."""

import re

# The paper measures chunks with the cl100k_base tokenizer (tiktoken).
_ENCODER = None
_ENCODER_LOADED = False

# Words that end with a period but do not end a sentence.
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "e.g.", "i.e.", "etc.",
                 "vs.", "fig.", "eq.", "no.", "inc.", "ltd.", "st.", "approx."}


def get_encoder():
    """Return the tiktoken encoder, or None if tiktoken is not available."""
    global _ENCODER, _ENCODER_LOADED
    if not _ENCODER_LOADED:
        _ENCODER_LOADED = True
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None  # not installed, or the encoding could not be downloaded
    return _ENCODER


def estimate_tokens(text):
    """Rough token count: about 0.75 words per token."""
    words = len(text.split())
    return round(words / 0.75)


def count_tokens(text):
    """Count tokens with tiktoken when possible, otherwise estimate."""
    encoder = get_encoder()
    if encoder is None:
        return estimate_tokens(text)
    return len(encoder.encode(text))


def merge_abbreviations(parts):
    """Join pieces that were wrongly split after an abbreviation like 'Dr.'."""
    merged = []
    for part in parts:
        last_word = merged[-1].split()[-1].lower() if merged else ""
        if last_word in ABBREVIATIONS:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


def split_sentences(text):
    """Split text into sentences. Blank lines always end a sentence."""
    sentences = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())  # tidy whitespace
        if not paragraph:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])", paragraph)
        sentences.extend(merge_abbreviations(parts))
    return sentences
