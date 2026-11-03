"""Build the weighted graph and its spectral coordinates from the six edge signals.

Used by SpectRAGIngestion, and by SpectRAGRetrieval.rebuild_graph() so that graph settings
(edge_threshold, graph_k, edge_weights, spectral_dims, n_clusters) can be changed and tuned
without embedding the documents again.
"""

from spectrag.utils.clusters import cluster_coords
from spectrag.utils.graph import active_weights, degrees, fuse_edges, keep_strongest
from spectrag.utils.spectral import compute_spectral


def build_graph(signals, settings):
    """Fuse the signals (Eq. 10), drop weak edges, keep each chunk's strongest graph_k.
    Returns (edges, weights used)."""
    weights = active_weights(settings["edge_weights"], signals)
    edges = fuse_edges(signals, weights, settings["edge_threshold"])
    chain = [(i, j) for i, j, _ in signals["adjacency"]]  # keep documents connected
    return keep_strongest(edges, settings["graph_k"], always=chain), weights


def build_spectral(edges, node_count, settings):
    """Laplacian (Eq. 11) -> spectral coordinates (Eq. 12-13) -> clusters.
    Returns (coords, eigenvalues, clusters)."""
    coords, eigenvalues = compute_spectral(edges, node_count, settings["spectral_dims"])
    clusters = cluster_coords(coords, settings["n_clusters"], degrees(edges, node_count))
    return coords, eigenvalues, clusters
