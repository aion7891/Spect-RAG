"""Deterministic wiring examples, NOT semantic embeddings or LLM extraction.

Keep fake model behavior outside the installed package. Replace these callbacks
with real model clients for actual document understanding.
"""

from hashlib import sha256
import json
import re

import numpy as np


CALLS = {"embedding_batches": 0, "extractions": 0}


def demo_embed(texts: list[str]) -> list[list[float]]:
    """Hash words into 16 counters solely to exercise the callback interface."""
    CALLS["embedding_batches"] += 1
    result = []
    for text in texts:
        vector = np.zeros(16)
        for word in re.findall(r"\w+", text.casefold()):
            vector[sha256(word.encode()).digest()[0] % 16] += 1
        result.append(vector.tolist())
    return result


def demo_extract(prompt: str) -> dict:
    """Use literal keyword rules; this does not implement the paper's LLM step."""
    CALLS["extractions"] += 1
    text = json.loads(prompt.split("SOURCE_CHUNK_JSON:\n", 1)[1]).casefold()
    concepts = [term for term in ("battery", "solar", "controller") if term in text]
    questions = ["What does a battery store?"] if "battery stores" in text else []
    return {"concepts": concepts, "utility_questions": questions}
