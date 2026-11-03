"""Turn texts into L2-normalised vectors (paper Table 1, "Embedding").

Three choices, picked with settings["embedder"]:
  "hash"   - offline word-hashing vectors (default, no model or internet needed)
  "ollama" - the paper's nomic-embed-text model through a local Ollama server
  a function - your own embed function: texts -> list of vectors
"""

import functools
import hashlib
import json
import re
import urllib.request

import numpy as np

from spectrag.errors import EmbedderError

# nomic-embed-text expects these task prefixes (paper: "search_document:").
PREFIXES = {"document": "search_document: ", "query": "search_query: "}


def normalize(vectors):
    """Scale every row to length 1, so a dot product equals cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)  # all-zero rows stay zero


def text_features(text):
    """Lowercase words plus word pairs, e.g. 'solar', 'farm', 'solar farm'."""
    words = re.findall(r"\w+", text.lower())
    pairs = [a + " " + b for a, b in zip(words, words[1:])]
    return words + pairs


@functools.lru_cache(maxsize=500_000)  # the same words come back again and again
def hash_feature(feature, dims):
    """Map a feature to a fixed (position, sign). md5 gives the same answer every run."""
    number = int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16)
    return number % dims, 1.0 if (number // dims) % 2 == 0 else -1.0


def hash_embed(texts, dims=768):
    """Offline embedder: each feature adds +1 or -1 at its hashed position."""
    vectors = np.zeros((len(texts), dims), dtype=np.float32)
    for row, text in enumerate(texts):
        for feature in text_features(text):
            position, sign = hash_feature(feature, dims)
            vectors[row, position] += sign
    return normalize(vectors)


def ollama_request(texts, model, url, retries=2):
    """Send one batch to Ollama's /api/embed and return its list of vectors (with retries)."""
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    for attempt in range(retries + 1):
        request = urllib.request.Request(url.rstrip("/") + "/api/embed", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read())["embeddings"]
        except OSError as error:
            if attempt == retries:
                raise EmbedderError("Cannot reach Ollama at " + url + ". Is 'ollama serve' "
                                      "running and 'ollama pull " + model + "' done?") from error


def ollama_embed(texts, kind="document", model="nomic-embed-text",
                 url="http://localhost:11434", batch_size=32):
    """Embed texts with nomic-embed-text, adding the task prefix the model expects."""
    prefixed = [PREFIXES[kind] + text for text in texts]
    vectors = []
    for start in range(0, len(prefixed), batch_size):
        vectors.extend(ollama_request(prefixed[start:start + batch_size], model, url))
    return normalize(vectors)


def make_embedder(settings):
    """Return embed(texts, kind="document") for the embedder named in settings."""
    choice = settings["embedder"]
    if callable(choice):
        return lambda texts, kind="document": normalize(choice(texts))
    if choice == "hash":
        return lambda texts, kind="document": hash_embed(texts, settings["embed_dims"])
    if choice == "ollama":
        return lambda texts, kind="document": ollama_embed(
            texts, kind, settings["ollama_model"], settings["ollama_url"])
    raise EmbedderError("Unknown embedder: %r (use 'hash', 'ollama' or a function)" % (choice,))
