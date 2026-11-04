"""Spectral coordinates (paper Eq. 11-13, Sections 3.2.3 and 3.4).

Small graphs: exact dense solver (numpy).
Large graphs: sparse solver (scipy, optional), as the paper suggests for scale (Section 7).
"""

import numpy as np

from spectrafuse.utils.graph import build_matrix, degrees

DENSE_LIMIT = 2000  # above this many chunks, use scipy's sparse solver if it is installed


def inverse_sqrt(degree):
    """1 / sqrt(degree), with 0 for nodes that have no edges (instead of 1 / sqrt(0))."""
    result = np.zeros_like(degree, dtype=float)
    result[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    return result


def normalized_laplacian(matrix):
    """L_sym = I - D^-1/2 W D^-1/2 (paper Eq. 11)."""
    scale = inverse_sqrt(matrix.sum(axis=1))
    return np.eye(len(matrix)) - scale[:, None] * matrix * scale[None, :]


def spectral_coords(laplacian, dims):
    """Eigenvectors 2 .. dims+1 of L_sym (the first one is trivial), one row per chunk.

    Returns (coords, eigenvalues). Small graphs give fewer than 'dims' columns.
    """
    values, vectors = np.linalg.eigh(laplacian)  # sorted from smallest to largest
    used = min(dims, len(values) - 1)
    if used <= 0:
        return np.zeros((len(values), 0)), values[:0]
    return vectors[:, 1:used + 1], values[1:used + 1]


def sparse_spectral_coords(edges, node_count, dims):
    """Same result as spectral_coords, without building an n x n matrix.

    The smallest eigenvalues of L_sym are 1 minus the largest of S = D^-1/2 W D^-1/2,
    and scipy's eigsh finds the largest ones of a sparse matrix quickly.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh
    rows = [i for i, j, w in edges] + [j for i, j, w in edges]
    cols = [j for i, j, w in edges] + [i for i, j, w in edges]
    weights = [w for i, j, w in edges] * 2
    matrix = coo_matrix((weights, (rows, cols)), shape=(node_count, node_count)).tocsr()
    scale = diags(inverse_sqrt(degrees(edges, node_count)))
    values, vectors = eigsh(scale @ matrix @ scale, k=dims + 1, which="LA",
                            v0=np.ones(node_count))  # fixed start vector: same result every run
    order = np.argsort(-values)
    return vectors[:, order][:, 1:], 1.0 - values[order][1:]


def has_scipy():
    """True if the optional scipy package is installed."""
    try:
        import scipy.sparse.linalg  # noqa: F401
        return True
    except ImportError:
        return False


def compute_spectral(edges, node_count, dims, dense_limit=DENSE_LIMIT):
    """Pick the dense or the sparse solver. Returns (coords, eigenvalues)."""
    dims = min(dims, node_count - 1)
    if node_count > dense_limit and dims + 1 < node_count and has_scipy():
        return sparse_spectral_coords(edges, node_count, dims)
    return spectral_coords(normalized_laplacian(build_matrix(edges, node_count)), dims)
