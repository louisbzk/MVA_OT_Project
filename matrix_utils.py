import numpy as np
from typing import Union, List
import sys


def logsumexp(x: np.ndarray, axis: Union[int, None] = None) -> Union[np.ndarray, float]:
    """
    Compute log-sum-exp of x along an axis, stable numerically.
    WARNING : only supports 1D and 2D arrays !

    :param x: a vector or matrix
    :param axis: the axis to sum along
    :return: log(sum(exp(x))) along axis
    """
    xm = np.amax(x, axis, keepdims=True)  # keepdims=True ensures correct broadcasting when subtracting
    return xm.flatten() + np.log(np.sum(np.exp(x - xm), axis))


def rowsum(mat: np.ndarray, log_domain: bool = False) -> np.ndarray:
    """
    Compute the sum of rows of a matrix

    :param mat: 2D matrix
    :param log_domain: whether the input is in log-domain or not
    :return: A vector whose i-th element is the i-th row sum of mat
    """
    if log_domain:
        return logsumexp(mat, axis=1)

    return np.sum(mat, axis=1)


def rowsum_k(mat: np.ndarray, k: int, log_domain: bool = False) -> Union[float, int, np.int8, np.float64, np.ndarray]:
    if log_domain:
        return logsumexp(mat[k])

    return np.sum(mat[k])


def colsum(mat: np.ndarray, log_domain: bool = False) -> np.ndarray:
    """
    Compute the sum of columns of a matrix

    :param mat: 2D matrix
    :param log_domain: whether the input is in log-domain or not
    :return: A vector whose i-th element is the i-th column sum of mat
    """
    if log_domain:
        return logsumexp(mat.T, axis=1)

    return rowsum(mat.T)


def colsum_k(mat: np.ndarray, k: int, log_domain: bool = False) -> Union[float, int, np.int8, np.float64, np.ndarray]:
    if log_domain:
        return logsumexp(mat.T[k])

    return np.sum(mat.T[k])


def conditioning_number(mat: np.ndarray) -> float:
    """
    Compute the conditioning number of a nonnegative matrix
    """
    flattened_mat = np.ravel(mat)
    min_nonzero_coeff = np.min(flattened_mat[flattened_mat != 0])
    return np.sum(mat) / min_nonzero_coeff


def compute_diameter(mat: np.ndarray) -> int:
    """
    Compute the diameter of the graph associated to mat (its adjacency matrix is mat).
    It is the longest path among all the shortest paths that connects all pairs of nodes.

    This is computed using the Dijkstra algorithm ran on all nodes (costly).
    :param mat: a non-negative square matrix
    """
    n = mat.shape[0]
    diameter = -1
    for node in range(n):
        max_shortest_path = np.max(_dijkstra(mat, node))
        if diameter < max_shortest_path:
            diameter = max_shortest_path
    return diameter


def _dijkstra(mat: np.ndarray, start_node: int):
    """
    Dijkstra algorithm to find the shortest path from a node to all others.
    :param mat: the adjacency matrix of the graph
    :param start_node: the starting point
    :return: an array containing the shorted paths from start_node to others.
    """
    n = mat.shape[0]
    shortest_paths = np.full(shape=n, fill_value=sys.float_info.max)
    shortest_paths[start_node] = 0
    unvisited_nodes = list(range(n))
    while unvisited_nodes:
        nearest_node = None
        for node in unvisited_nodes:
            if nearest_node is None:
                nearest_node = node
            elif shortest_paths[node] < shortest_paths[nearest_node]:
                nearest_node = node

        # Find neighbors of the nearest node
        neighbors = mat[nearest_node].nonzero()[0]
        for neighbor in neighbors:
            path_length = shortest_paths[nearest_node] + mat[nearest_node, neighbor]
            if path_length < shortest_paths[neighbor]:
                shortest_paths[neighbor] = path_length

        unvisited_nodes.remove(nearest_node)

    return shortest_paths


def generate_matrix_from_cond(n: int, m: int, target_cond: float, scale: float = 1.):
    """
    Generate a sparse random matrix of a given size with a target conditioning number.
    :param n: size of the matrix
    :param m: number of nonzero elements (size of support)
    :param target_cond: target conditioning number
    :param scale: scale of coefficients (they range from 0 to scale)
    :return: a matrix with m nonzero coefficients with the specified conditioning
    """
    if target_cond <= m:
        raise ValueError('Conditioning cannot be less than the number of non-zeros')

    nonzero_coeffs = np.random.rand(m)
    while np.min(nonzero_coeffs) == 0.0:  # very unlikely...
        nonzero_coeffs = np.random.rand(m)
    min_coeff_idx = np.argmin(nonzero_coeffs)

    # construct the matrix
    mat = np.zeros(shape=(n, n), dtype=float)
    non_diagonal_indices = list(range(n * n))
    for i in range(n):
        non_diagonal_indices.remove(i * n + i)
    nonzero_indices = np.random.choice(non_diagonal_indices, m, replace=False)
    min_coeff_position = None
    for nonzero_idx, mat_idx in enumerate(nonzero_indices):
        if nonzero_idx == min_coeff_idx:
            min_coeff_position = (mat_idx // n, mat_idx % n)
        mat[mat_idx // n][mat_idx % n] = nonzero_coeffs[nonzero_idx]

    # tweak conditioning using an additive scaling constant, then rescale using a multiplicative constant to
    # adjust the matrix coefficients to the requested scale
    alpha = (np.sum(mat) - target_cond * mat[min_coeff_position]) / float(target_cond - m)
    nonzero_mask = mat != 0.0
    mat[nonzero_mask] += alpha  # conditioning is now target_cond
    mat = mat / np.max(mat) * scale  # rescale coefficients (conditioning is unchanged)

    return mat


def qr_algorithm(mat: np.ndarray, tol: float = 1e-8, max_iter: int = 5000):
    """
    Compute eigenvalues & eigenvectors of a matrix.
    Source : http://madrury.github.io/jekyll/update/statistics/2017/10/04/qr-algorithm.html
    J. G. F. Francis, The QR Transformation A Unitary Analogue to the LR Transformation—Part 1,
    The Computer Journal, Volume 4, Issue 3, 1961, Pages 265–271, https://doi.org/10.1093/comjnl/4.3.265

    :param mat: the input matrix
    :param tol: stopping criterion
    :param max_iter: stop after this number of iterations
    :return: an array containing the eigenvalues of mat and an array containing the corresponding eigenvectors
    """
    q, r = np.linalg.qr(mat)
    previous = np.eye(len(mat))
    for i in range(max_iter):
        previous = previous @ q
        x = r @ q
        q, r = np.linalg.qr(x)
        if np.allclose(x, np.triu(x), atol=tol):
            break
    else:
        print('[WARNING] qr_algorithm did not converge')
    return np.diagonal(x), (previous @ q).T


def zerodiag(mat: np.ndarray) -> np.ndarray:
    """
    Return the same matrix with 0 on the diagonal
    """
    _m = mat.copy()
    np.fill_diagonal(_m, 0.)
    return _m


if __name__ == '__main__':
    A = generate_matrix_from_cond(5, 21, target_cond=2000, scale=100)
    diam = compute_diameter(A)
    print(diam)
