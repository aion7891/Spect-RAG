"""Remember which embedder built an index, so the same one is used when the index is opened.

settings.json cannot store a function, so an embedder is saved by name:
  "hash", "ollama"          - the built-in embedders, saved as they are
  "custom:<function name>"  - your own function; pass it again with embed_fn=...
"""

BUILT_IN = ("hash", "ollama")
CUSTOM = "custom:"


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
