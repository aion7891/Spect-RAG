"""Evaluation metrics from the paper (Section 4.5).

Retrieval level (Eq. 23-24): precision, recall, F1 of the retrieved set; MRR of the ranking.
Answer level (Eq. 25-27): exact match, token F1, and "contains" (Table 7), after normalising
answers (lower case, no punctuation, no articles). With several reference answers the best
score is used.
"""

import re
import string
from collections import Counter


def precision_recall_f1(found, retrieved_count, gold_count):
    """Precision = found / retrieved, recall = found / gold, F1 = their harmonic mean."""
    precision = found / retrieved_count if retrieved_count else 0.0
    recall = found / gold_count if gold_count else 0.0
    if precision + recall == 0:
        return 0.0, 0.0, 0.0  # paper: F1 is 0 when both are 0
    return precision, recall, 2 * precision * recall / (precision + recall)


def reciprocal_rank(ranked_ids, gold_ids):
    """1 / rank of the first gold item (ranks start at 1); 0 if none was retrieved."""
    for rank, item in enumerate(ranked_ids, start=1):
        if item in gold_ids:
            return 1.0 / rank
    return 0.0


def normalize_answer(text):
    """'The Douro Festival!' -> 'douro festival'."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, references):
    """Eq. 25: 1 if the normalised prediction equals a normalised reference."""
    guess = normalize_answer(prediction)
    return float(any(guess == normalize_answer(ref) for ref in references))


def token_f1(prediction, references):
    """Eq. 26-27: overlap of the word multisets of prediction and reference."""
    best = 0.0
    guess = normalize_answer(prediction).split()
    for ref in references:
        truth = normalize_answer(ref).split()
        common = sum((Counter(guess) & Counter(truth)).values())
        if common:
            p, r = common / len(guess), common / len(truth)
            best = max(best, 2 * p * r / (p + r))
    return best


def contains(prediction, references):
    """Table 7 'Contains': a reference appears as whole words inside the prediction."""
    guess = " " + normalize_answer(prediction) + " "
    return float(any(" " + normalize_answer(ref) + " " in guess for ref in references))
