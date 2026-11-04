"""Check settings before they are used, with messages that say how to fix them."""

from spectrafuse.errors import SettingsError
from spectrafuse.utils.embedder_id import BUILT_IN, is_saved_function

# Whole numbers: name -> smallest allowed value.
AT_LEAST = {"chunk_tokens": 16, "overlap_tokens": 0, "min_chunk_tokens": 0, "embed_dims": 8,
            "hnsw_m": 2, "hnsw_ef_construction": 1, "hnsw_ef_search": 1, "dense_top_n": 1,
            "semantic_k": 1, "graph_k": 1, "spectral_dims": 1, "seed_count": 1, "hops": 0,
            "max_candidates": 1, "token_budget": 1, "rrf_k": 1, "ppr_steps": 1, "core_size": 0}
# Numbers: name -> (low, high); None means no limit on that side.
BETWEEN = {"semantic_percentile": (0, 100), "edge_threshold": (0, None), "bm25_k1": (0, None),
           "bm25_b": (0, 1), "bm25_weight": (0, 1), "alpha": (0, 1), "ppr_beta": (0, 1),
           "diffusion_t": (0, None), "mmr_lambda": (0, 1), "precision_floor": (0, 1),
           "length_power": (0, 1)}
CHOICES = {"ann_backend": ("auto", "exact", "faiss", "hnsw"),
           "proximity_mode": ("hybrid", "ppr", "spectral", "heat", "none"),
           "selector": ("spectrag", "mmr", "protected_mmr", "anchor_expand_ce", "topk")}
EMBEDDERS = BUILT_IN
SIGNALS = ("semantic", "adjacency", "section", "entity", "utility", "citation")
SELECT_WEIGHTS = ("a", "b", "c", "d", "lambda")


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_whole_numbers(settings):
    for name, low in AT_LEAST.items():
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < low:
            raise SettingsError("%s must be a whole number >= %d, got %r" % (name, low, value))


def check_ranges(settings):
    for name, (low, high) in BETWEEN.items():
        value = settings[name]
        if not is_number(value) or value < low or (high is not None and value > high):
            limit = "between %s and %s" % (low, high) if high is not None else ">= %s" % low
            raise SettingsError("%s must be a number %s, got %r" % (name, limit, value))


def check_choices(settings):
    for name, allowed in CHOICES.items():
        if settings[name] not in allowed:
            raise SettingsError("%s must be one of %s, got %r" % (name, ", ".join(allowed), settings[name]))


def check_embedders(settings):
    choice = settings["embedder"]
    if not (callable(choice) or choice in EMBEDDERS or is_saved_function(choice)):
        raise SettingsError("embedder must be 'hash', 'ollama' or a function texts -> vectors, got %r" % (choice,))
    split = settings["semantic_embedder"]
    if not (split is None or callable(split) or split in EMBEDDERS or is_saved_function(split)):
        raise SettingsError("semantic_embedder must be None, 'hash', 'ollama' or a function, got %r" % (split,))


def check_weights(settings):
    weights = settings["edge_weights"]
    if not isinstance(weights, dict) or set(weights) - set(SIGNALS) or not weights:
        raise SettingsError("edge_weights must be a dict with keys from %s" % ", ".join(SIGNALS))
    if any(not is_number(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise SettingsError("edge_weights must be numbers >= 0 with a positive sum, got %r" % (weights,))
    select = settings["select_weights"]
    if not isinstance(select, dict) or set(select) != set(SELECT_WEIGHTS) \
            or any(not is_number(w) or w < 0 for w in select.values()):
        raise SettingsError("select_weights needs keys %s with numbers >= 0" % ", ".join(SELECT_WEIGHTS))


def check_relations(settings):
    if settings["overlap_tokens"] >= settings["chunk_tokens"]:
        raise SettingsError("overlap_tokens (%d) must be smaller than chunk_tokens (%d)"
                            % (settings["overlap_tokens"], settings["chunk_tokens"]))
    clusters = settings["n_clusters"]
    if clusters is not None and (not isinstance(clusters, int) or isinstance(clusters, bool) or clusters < 1):
        raise SettingsError("n_clusters must be None (automatic) or a whole number >= 1, got %r" % (clusters,))


def validate_settings(settings):
    """Raise SettingsError with a clear message if any setting is not usable."""
    for check in (check_whole_numbers, check_ranges, check_choices, check_embedders,
                  check_weights, check_relations):
        check(settings)
    return settings
