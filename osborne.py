import numpy as np


def osborne(
        mat: np.ndarray,
        method: str,
):
    """
    Perform matrix balancing on the matrix 'mat' using the Osborne algorithm.

    :param mat: The matrix to balance
    :param method: The update rule for the Osborne update. One of :
        * `cycle` (default) : cycle through the coordinates in order (1, ..., n, 1, ..., n, etc.),
        * `greedy` : select the coordinate with the most descent,
        * `random-cycle` : choose random permutations to cycle through the coordinates
        * `random` : at every step, the update coordinate is chosen randomly
    :return:
    """
    raise NotImplementedError
