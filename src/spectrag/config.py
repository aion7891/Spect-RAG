"""Default settings for SpectRAG.

Values marked "paper" come from the SpectRAG paper (Tables 1-3).
Values marked "assumed" are not given in the paper, so we picked a sensible value.
"""

import copy

# Edge-signal weights from the paper's "signal configuration 1" (Table 9).
EDGE_WEIGHTS_CONFIG_1 = {
    "semantic": 0.300,
    "section": 0.250,
    "entity": 0.200,
    "adjacency": 0.084,
    "utility": 0.083,
    "citation": 0.083,
}

# Edge-signal weights from the paper's "signal configuration 2" (Table 10).
EDGE_WEIGHTS_CONFIG_2 = {
    "semantic": 0.150,
    "section": 0.200,
    "entity": 0.250,
    "adjacency": 0.100,
    "utility": 0.280,
    "citation": 0.020,
}

DEFAULTS = {
    # --- Chunking (paper) ---
    "chunk_tokens": 1200,
    "overlap_tokens": 100,
    "min_chunk_tokens": 32,
    "semantic_percentile": 90,
    # --- Embedding ---
    "embedder": "hash",          # "hash" (offline), "ollama" (nomic-embed-text), or your own function
    "embed_dims": 768,           # paper: nomic-embed-text is 768-d (used by the hash embedder)
    "ollama_model": "nomic-embed-text",
    "ollama_url": "http://localhost:11434",
    # --- Graph (weights: paper, the rest: assumed) ---
    "edge_weights": EDGE_WEIGHTS_CONFIG_1,
    "semantic_k": 10,            # assumed
    "edge_threshold": 0.05,      # assumed
    # --- Spectral (paper) ---
    "spectral_dims": 32,
    "n_clusters": None,          # assumed: None means about sqrt(n / 2) clusters
    # --- Retrieval (paper) ---
    "seed_count": 20,
    "hops": 2,
    "max_candidates": 200,
    "token_budget": 12000,
    # --- Retrieval (assumed) ---
    "rrf_k": 60,
    "alpha": 0.3,
    "ppr_beta": 0.15,
    "ppr_steps": 30,
    "core_size": 3,
    "precision_floor": 0.4,
    "select_weights": {"a": 1.0, "b": 0.3, "c": 0.2, "d": 0.1, "lambda": 0.5},
    "length_power": 0.5,
}


def make_settings(overrides=None):
    """Return a fresh copy of DEFAULTS with the user's overrides applied."""
    settings = copy.deepcopy(DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if key not in settings:
                raise KeyError("Unknown setting: " + key)
            settings[key] = copy.deepcopy(value)
    return settings
