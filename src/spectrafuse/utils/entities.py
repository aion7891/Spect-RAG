"""Simple entity extraction for entity edges (paper Section 3.2.1, "Entity").

The paper uses gpt-4o-mini. This default uses patterns instead, so no API is needed:
  - capitalised phrases: "Maya Chen", "Helios Grid Planner", "Aurora S1"
  - each word of those phrases: "helios", "grid", "planner"
  - acronyms and codes: "CTO", "S2"
  - years: "2019"
"""

import re

from spectrafuse.utils.tokens import split_sentences

PHRASE = re.compile(r"\b[A-Z][\w'-]*(?:[ \t]+[A-Z][\w'-]*)*")   # never crosses a line break
YEAR = re.compile(r"\b(?:1[89]\d\d|20\d\d)\b")
HEADING_START = re.compile(r"^#*\s*(?:\d+(?:\.\d+)*\.?\s+)?")    # '## 2.1 ' at line start

# Capitalised words that are not entities on their own.
STOPWORDS = {"the", "a", "an", "in", "on", "at", "of", "and", "or", "but", "by", "for",
             "to", "it", "its", "this", "that", "these", "those", "he", "she", "they",
             "we", "you", "i", "his", "her", "their", "our", "see", "since", "under",
             "after", "before", "both", "dr", "mr", "mrs", "ms", "prof",
             "section", "sections", "sec", "table", "figure", "fig", "chapter", "new",
             "january", "february", "march", "april", "may", "june", "july", "august",
             "september", "october", "november", "december", "monday", "tuesday",
             "wednesday", "thursday", "friday", "saturday", "sunday", "week", "today"}


def clean_phrase(phrase):
    """Drop leading/trailing stopwords: 'The Aurora S1' -> 'aurora s1'."""
    words = phrase.lower().split()
    while words and words[0] in STOPWORDS:
        words.pop(0)
    while words and words[-1] in STOPWORDS:
        words.pop()
    return " ".join(words)


def sentence_phrases(sentence):
    """Capitalised phrases in one sentence. A single word that starts the sentence is skipped,
    because every sentence starts with a capital letter ("Farmers check ...")."""
    found = []
    text = HEADING_START.sub("", sentence.strip())  # '## 1. History' -> 'History'
    for match in PHRASE.finditer(text):
        phrase = clean_phrase(match.group())
        one_word_at_start = match.start() == 0 and " " not in match.group().strip()
        if phrase and not one_word_at_start:
            found.append(phrase)
    return found


def add_entities(chunks, entity_fn=None):
    """Store each chunk's entities as chunk['entities'] (paper: kept for reuse)."""
    entity_fn = entity_fn or extract_entities
    for chunk in chunks:
        chunk["entities"] = sorted(set(entity_fn(chunk["text"])))
    return chunks


def extract_entities(text):
    """Return a sorted list of lowercased entities found in the text."""
    entities = set(YEAR.findall(text))
    sentences = [s for line in text.splitlines() for s in split_sentences(line)]
    for sentence in sentences:  # chunks keep one sentence/heading per line
        for phrase in sentence_phrases(sentence):
            if len(phrase) < 2:
                continue  # one letter, e.g. the "U" of "U.S."
            entities.add(phrase)
            for word in phrase.split():
                if len(word) > 1 and word not in STOPWORDS:
                    entities.add(word)
    return sorted(entities)
