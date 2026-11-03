"""Remember which embedder built an index, so the same one is used when the index is opened.

settings.json cannot store a function, so an embedder is saved by name:
  "hash", "ollama"          - the built-in embedders, saved as they are
  "custom:<function name>"  - your own function; pass it again with embed_fn=...
A name alone cannot tell two embedders apart, so the index also keeps a fingerprint: the
vector of one fixed sentence (embedder_probe.npy). Opening the index embeds it again and
compares (DESIGN.md D65).
"""

import numpy as np

from spectrag.errors import EmbedderError
from spectrag.log import logger

BUILT_IN = ("hash", "ollama")
CUSTOM = "custom:"
PROBE_TEXT = "SpectRAG embedder check: Helios Energy builds solar farms near Porto."
SAME_EMBEDDER = 0.99  # assumed: lowest cosine for "the same embedder" (GPUs vary a little)


def function_name(fn):
    """A readable name for a function, a functools.partial or a callable object."""
    inner = getattr(fn, "func", fn)  # functools.partial keeps the wrapped function in .func
    return getattr(inner, "__qualname__", None) or type(inner).__qualname__


def embedder_name(choice):
    """What settings.json stores for an embedder: its name, or 'custom:<name>' for a function."""
    return CUSTOM + function_name(choice) if callable(choice) else choice


def saved_settings(settings):
    """A copy of the settings with every embedder function replaced by its name."""
    return dict(settings, embedder=embedder_name(settings["embedder"]),
                semantic_embedder=embedder_name(settings["semantic_embedder"]))


def read_embedder_name(value):
    """An embedder name read from settings.json. Indexes made before 0.1.2 saved str(fn) for
    your own function (e.g. 'functools.partial(<function ...>)'); that text becomes 'custom:...'."""
    if value is None or value in BUILT_IN or str(value).startswith(CUSTOM):
        return value
    return CUSTOM + str(value)


def read_saved_settings(settings):
    """The saved settings, with old-style names of your own functions turned into 'custom:...'."""
    return dict(settings, embedder=read_embedder_name(settings["embedder"]),
                semantic_embedder=read_embedder_name(settings.get("semantic_embedder")))


def is_saved_function(choice):
    """True if the embedder is the saved name of your own function (not the function itself)."""
    return isinstance(choice, str) and choice.startswith(CUSTOM)


def embed_probe(embed):
    """The embedder's vector for PROBE_TEXT: the fingerprint saved with the index."""
    return np.asarray(embed([PROBE_TEXT], kind="document")[0], dtype=np.float32)


def check_embedder(embed, saved_probe, built_with):
    """Raise EmbedderError if embed() is not the embedder that built the index ('built_with')."""
    if saved_probe is None:
        logger.info("this index has no embedder_probe.npy (made before 0.1.2): embedder not checked")
        return
    probe = embed_probe(embed)
    fix = "open the index with the embedder it was built with (%s)" % embedder_name(built_with)
    if len(probe) != len(saved_probe):
        raise EmbedderError("The embedder gives %d dimensions but the index has %d: %s."
                            % (len(probe), len(saved_probe), fix))
    similarity = float(probe @ saved_probe)  # both vectors have length 1: this is the cosine
    if similarity < SAME_EMBEDDER:
        raise EmbedderError("This index was built with a different embedder (the check sentence "
                            "has cosine %.3f, below %.2f): %s." % (similarity, SAME_EMBEDDER, fix))
