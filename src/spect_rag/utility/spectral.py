"""Normalized-Laplacian coordinates with explicit disconnected-graph policy."""

import networkx as nx
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


def normalized_laplacian(graph: nx.Graph) -> sparse.csr_matrix:
    """Use zero rows/diagonal for isolates; positive-degree blocks use Equation 11."""
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Spectral coordinates require a simple undirected graph without self-loops.")
    if any(not np.isfinite(data.get("weight", 1)) or data.get("weight", 1) <= 0
           for _, _, data in graph.edges(data=True)):
        raise ValueError("Graph edge weights must be finite and positive.")
    if not graph:
        return sparse.csr_matrix((0, 0), dtype=np.float64)
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=list(graph), weight="weight", dtype=np.float64, format="csr",
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros(len(degree))
    np.divide(1.0, np.sqrt(degree), out=inverse, where=degree > 0)
    scale = sparse.diags(inverse)
    return sparse.csr_matrix(sparse.diags((degree > 0).astype(float)) - scale @ adjacency @ scale)


def spectral_coordinates(
    graph: nx.Graph, *, dimension: int = 32, zero_tolerance: float = 1e-9,
) -> dict:
    """Select globally lowest positive modes from component-wise decompositions.

    All component zero modes are excluded. Isolates have no positive mode and
    retain zero coordinate rows. Small components use dense eigh; large ones use
    sparse eigsh. Signs use a stable largest-magnitude pivot convention.
    """
    if type(dimension) is not int or dimension < 0:
        raise ValueError("Spectral dimension must be a non-negative integer.")
    if not np.isfinite(zero_tolerance) or not 0 < zero_tolerance < 1:
        raise ValueError("Spectral zero tolerance must lie strictly between zero and one.")
    laplacian = normalized_laplacian(graph)
    rows = {node: row for row, node in enumerate(graph)}
    components = sorted(
        [sorted(rows[node] for node in component) for component in nx.connected_components(graph)],
        key=lambda indices: indices[0],
    )
    candidates = []
    if dimension:
        for indices in components:
            size = len(indices)
            if size == 1:
                continue
            block = laplacian[indices][:, indices]
            if size <= 256 or dimension >= size - 1:
                values, vectors = linalg.eigh(block.toarray())
            else:
                try:
                    values, vectors = eigsh(
                        block, k=dimension + 1, which="SM",
                        v0=np.linspace(1.0, 2.0, size), tol=1e-10,
                    )
                except ArpackNoConvergence as exc:
                    raise RuntimeError("Spectral eigensolver did not converge; no index was saved.") from exc
            modes = [int(i) for i in np.argsort(values) if values[i] > zero_tolerance][:dimension]
            for mode in modes:
                vector = vectors[:, mode]
                if vector[np.argmax(np.abs(vector))] < 0:
                    vector = -vector
                candidates.append((float(values[mode]), indices[0], mode, indices, vector))
    selected = sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[:dimension]
    coordinates = np.zeros((len(graph), len(selected)), dtype=np.float64)
    values = np.asarray([item[0] for item in selected], dtype=np.float64)
    for column, (_, _, _, indices, vector) in enumerate(selected):
        coordinates[indices, column] = vector
    residuals = np.linalg.norm(laplacian @ coordinates - coordinates * values, axis=0)
    if np.any(residuals > 1e-6):
        raise RuntimeError("Spectral eigenpair residuals exceed tolerance; no index was saved.")
    return {
        "coordinates": coordinates, "eigenvalues": values, "residuals": residuals,
        "requested_dimension": dimension, "zero_tolerance": float(zero_tolerance),
        "zero_modes": len(components),
    }
