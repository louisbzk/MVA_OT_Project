import numpy as np
from typing import Union
import sys


def rowsum(mat: np.ndarray) -> np.ndarray:
    """
    Compute the sum of rows of a matrix

    :param mat: 2D matrix
    :return: A vector whose i-th element is the i-th row sum of mat
    """
    return np.sum(mat, axis=1)


def rowsum_k(mat: np.ndarray, k: int) -> Union[float, int, np.int8, np.float64, np.ndarray]:
    return np.sum(mat[k])


def colsum(mat: np.ndarray) -> np.ndarray:
    """
    Compute the sum of columns of a matrix

    :param mat: 2D matrix
    :return: A vector whose i-th element is the i-th column sum of mat
    """
    return rowsum(mat.T)


def colsum_k(mat: np.ndarray, k: int) -> Union[float, int, np.int8, np.float64, np.ndarray]:
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
    if target_cond <= 1.:
        raise ValueError('Conditioning cannot be below 1')

    nonzero_coeffs = scale * np.random.rand(m)
    while np.min(nonzero_coeffs) == 0.0:  # very unlikely...
        nonzero_coeffs = scale * np.random.rand(m)
    min_coeff_idx = np.argmin(nonzero_coeffs)

    # construct the matrix
    mat = np.zeros(shape=(n, n), dtype=float)
    nonzero_indices = np.random.choice(list(range(n * n)), m, replace=False)
    min_coeff_position = None
    for nonzero_idx, mat_idx in enumerate(nonzero_indices):
        if nonzero_idx == min_coeff_idx:
            min_coeff_position = (mat_idx // n, mat_idx % n)
        mat[mat_idx // n][mat_idx % n] = nonzero_coeffs[nonzero_idx]

    # tweak conditioning
    mat[min_coeff_position] = (np.sum(mat) - mat[min_coeff_position]) / (target_cond - 1)

    return mat


if __name__ == '__main__':
    A = generate_matrix_from_cond(5, 21, target_cond=2, scale=100)
    diam = compute_diameter(A)
    print(diam)
