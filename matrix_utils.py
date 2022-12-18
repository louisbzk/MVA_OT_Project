import numpy as np
from typing import Union


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
