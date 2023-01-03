import numpy as np
from matrix_utils import rowsum, rowsum_k, colsum, colsum_k, logsumexp
from itertools import cycle
from typing import Union, Tuple, List


def osborne(
        mat: np.ndarray,
        epsilon: float,
        method: str,
        history: bool = False,
        log_domain: bool = False,
):
    """
    Perform matrix balancing on a square matrix.
    This does NOT check that the matrix is square.
    This does NOT use a log-domain implementation by default (i.e. the algorithm expects a raw nonnegative matrix,
    not the log of the target matrix).

    :param mat: The matrix to balance
    :param epsilon: Precision criterion
    :param method: The update rule for the Osborne update. One of :
        * `cycle` (default) : cycle through the coordinates in order (1, ..., n, 1, ..., n, etc.),
        * `greedy` : select the coordinate with the most descent,
        * `random-cycle` : choose random permutations to cycle through the coordinates
        * `random` : at every step, the update coordinate is chosen randomly
    :param history: if True, return a history of the scalings at each step
    :param log_domain: if True, uses log-domain computational tricks to improve numerical precision. If so, the
    entry-wise log of mat has to be input.
    :return: A vector representing the diagonal balance, and a history of increments if history=True. The history
    is a list of tuples of the form (coordinate, increment)
    """
    if method == 'cycle':
        return cyclic_osborne(mat, epsilon, history, log_domain)

    if method == 'greedy':
        return greedy_osborne(mat, epsilon, history, log_domain)

    if method == 'random-cycle':
        return random_cyclic_osborne(mat, epsilon, history, log_domain)

    if method == 'random':
        return random_osborne(mat, epsilon, history, log_domain)


def osborne_update(balanced_mat: np.ndarray,
                   balance: np.ndarray,
                   coordinate: int,
                   log_domain: bool) -> float:
    """
    Modifies the current balance in-place and returns the increment applied.
    """
    r_k = rowsum_k(balanced_mat, coordinate, log_domain)
    c_k = colsum_k(balanced_mat, coordinate, log_domain)

    if not log_domain:
        increment = 0.5 * (np.log(c_k) - np.log(r_k))

    else:  # in that case, r_k := log(r_k), c_k := log(c_k)
        increment = 0.5 * (c_k - r_k)

    balance[coordinate] = balance[coordinate] + increment
    return increment


def compute_imbalance(balanced_mat, log_domain, order=1):
    """
    Compute the current imbalance

    :param balanced_mat: the matrix with the current balance applied
    :param log_domain: whether the input is in log-domain or not
    :param order: order of the norm to measure imbalance with
    :return: the measure of imbalance
    """
    r = rowsum(balanced_mat, log_domain)
    if not log_domain:
        imbalance = np.linalg.norm(r - colsum(balanced_mat), order) / np.sum(r)
    else:
        imbalance = np.linalg.norm(np.exp(r) - np.exp(colsum(balanced_mat, log_domain)), order) / np.exp(logsumexp(r))

    return imbalance


def compute_balanced_mat(mat, bal, log_domain):
    """
    Apply the current balancing to the matrix and return the result

    :param mat: the matrix to balance
    :param bal: the current balance
    :param log_domain: whether the matrix is in log-domain
    :return: the balanced matrix in the appropriate domain
    """
    if not log_domain:
        balanced_mat = np.diag(np.exp(bal)) @ mat @ np.diag(np.exp(-bal))
    else:
        # use broadcasting rules to compute (x_i - x_j) + log mat_{i, j}
        balanced_mat = mat + bal[:, None] - bal[None, :]

    return balanced_mat


def cyclic_osborne(mat: np.ndarray, epsilon: float, history: bool, log_domain: bool):
    n = mat.shape[0]
    balance = np.zeros(n, dtype=float)
    balanced_mat = compute_balanced_mat(mat, balance, log_domain)
    if history:
        balance_hist = []

    coord_cycle = cycle(range(n))
    while compute_imbalance(balanced_mat, log_domain) > epsilon:
        update_coord = next(coord_cycle)
        balanced_mat = compute_balanced_mat(mat, balance, log_domain)
        increment = osborne_update(balanced_mat, balance, update_coord, log_domain)
        if history:
            balance_hist.append((update_coord, increment))

    if history:
        return balance, balance_hist

    return balance


def greedy_osborne(mat: np.ndarray, epsilon: float, history: bool, log_domain: bool):
    n = mat.shape[0]
    balance = np.zeros(n, dtype=float)
    balanced_mat = compute_balanced_mat(mat, balance, log_domain)
    if history:
        balance_hist = []

    while compute_imbalance(balanced_mat, log_domain) > epsilon:
        r, c = rowsum(balanced_mat, log_domain), colsum(balanced_mat, log_domain)
        if not log_domain:
            update_coord = np.argmax(np.abs(np.sqrt(r) - np.sqrt(c)))
        else:
            update_coord = np.argmax(np.abs(np.exp(0.5 * r) - np.exp(0.5 * c)))

        increment = osborne_update(balanced_mat, balance, update_coord, log_domain)
        balanced_mat = compute_balanced_mat(mat, balance, log_domain)
        if history:
            balance_hist.append((update_coord, increment))

    if history:
        return balance, balance_hist

    return balance


def random_cyclic_osborne(mat: np.ndarray, epsilon: float, history: bool, log_domain: bool):
    n = mat.shape[0]
    balance = np.zeros(n, dtype=float)
    balanced_mat = compute_balanced_mat(mat, balance, log_domain)
    if history:
        balance_hist = []

    coords = np.array(range(n), dtype=int)
    np.random.shuffle(coords)
    i = 0
    while compute_imbalance(balanced_mat, log_domain) > epsilon:
        if i == n:  # Reshuffle array
            i = 0
            np.random.shuffle(coords)

        update_coord = coords[i]
        increment = osborne_update(balanced_mat, balance, update_coord, log_domain)
        if history:
            balance_hist.append((update_coord, increment))
        balanced_mat = compute_balanced_mat(mat, balance, log_domain)
        i += 1

    if history:
        return balance, balance_hist

    return balance


def random_osborne(mat: np.ndarray, epsilon: float, history: bool, log_domain: bool):
    n = mat.shape[0]
    balance = np.zeros(n, dtype=float)
    balanced_mat = compute_balanced_mat(mat, balance, log_domain)
    if history:
        balance_hist = []

    while compute_imbalance(balanced_mat, log_domain) > epsilon:
        update_coord = np.random.randint(0, n)
        increment = osborne_update(balanced_mat, balance, update_coord, log_domain)
        if history:
            balance_hist.append((update_coord, increment))
        balanced_mat = compute_balanced_mat(mat, balance, log_domain)

    if history:
        return balance, balance_hist

    return balance


def potential(scaling: np.ndarray, mat: np.ndarray) -> float:
    """
    Compute the potential associated to a given scaling and matrix.

    :param scaling: the 1D-vector representing the diagonal scaling, in log-domain
    :param mat: the matrix that is being balanced
    :return: the potential value
    """
    n = mat.shape[0]
    v = np.zeros(n * n, dtype=float)
    for i in range(n):
        for j in range(n):
            v[i * n + j] = scaling[i] - scaling[j]

    return np.log(np.sum(np.exp(v) * mat.flatten()))


def potential_from_increment_history(history: List[Tuple], mat: np.ndarray) -> np.ndarray:
    """
    Compute the history of potentials given a history of the scalings, represented as (coordinate, increment) pairs.

    :param history: increment history of the scalings through the Osborne iterations
    :param mat: the matrix that was balanced
    :return: a history of the values of the potential
    """
    n = mat.shape[0]
    scaling = np.zeros(n, dtype=float)
    potential_history = np.zeros(shape=len(history) + 1, dtype=float)
    potential_history[0] = potential(scaling, mat)

    for i, (update_coord, increment) in enumerate(history):
        scaling[update_coord] = scaling[update_coord] + increment
        potential_history[i + 1] = potential(scaling, mat)

    return potential_history


def test_osborne_variants(test_mat: Union[None, np.ndarray] = None) -> Tuple[bool, bool]:
    """
    Test the different algorithm variants : all should return the same result

    :param test_mat: The matrix on which to perform the test. If None, a random matrix is used
    :return: Two booleans : the first indicates if the algorithms all converged (i.e. the resulting matrix is
    indeed balanced for all variants), and the second indicates if the row  (resp. column) sums for all variants are
    the same.
    """

    if test_mat is None:
        _test_mat = 10. * np.random.rand(10, 10)
        for i in range(10):
            _test_mat[i, i] = 0.
    else:
        _test_mat = test_mat

    eps = 1e-12
    balance_cyclic = osborne(_test_mat, eps, method='cycle')
    balance_random_cyclic = osborne(_test_mat, eps, method='random-cycle')
    balance_greedy = osborne(_test_mat, eps, method='greedy')
    balance_random = osborne(_test_mat, eps, method='random')

    balanced_cyclic = np.diag(np.exp(balance_cyclic)
                              @ _test_mat @ np.diag(np.exp(-balance_cyclic)))
    r_cyclic = rowsum(balanced_cyclic)
    c_cyclic = colsum(balanced_cyclic)

    balanced_random_cyclic = np.diag(np.exp(
        balance_random_cyclic) @ _test_mat @ np.diag(np.exp(-balance_random_cyclic)))
    r_random_cyclic = rowsum(balanced_random_cyclic)
    c_random_cyclic = colsum(balanced_random_cyclic)

    balanced_greedy = np.diag(np.exp(balance_greedy)
                              @ _test_mat @ np.diag(np.exp(-balance_greedy)))
    r_greedy = rowsum(balanced_greedy)
    c_greedy = colsum(balanced_greedy)

    balanced_random = np.diag(np.exp(balance_random)
                              @ _test_mat @ np.diag(np.exp(-balance_random)))
    r_random = rowsum(balanced_random)
    c_random = colsum(balanced_random)

    all_variants_converged = np.allclose(
        np.array([
            np.linalg.norm(r_cyclic - c_cyclic, 1),
            np.linalg.norm(r_random_cyclic - c_random_cyclic, 1),
            np.linalg.norm(r_greedy - c_greedy, 1),
            np.linalg.norm(r_random - c_random, 1),
        ]), 0.
    )

    results_are_the_same = (
        np.allclose(r_cyclic, r_greedy) and np.allclose(c_cyclic, c_greedy) and
        np.allclose(r_cyclic, r_random_cyclic) and np.allclose(c_cyclic, c_random_cyclic) and
        np.allclose(r_cyclic, r_random) and np.allclose(c_cyclic, c_random)
    )

    print(
        f'With A = diag(exp(balance)) @ mat @ diag(exp(-balance)) : \n\n'
        f'Cyclic : \n'
        f'\t||r(A) - c(A)||_1 = {np.linalg.norm(r_cyclic - c_cyclic, 1)},\n'
        f'Random cyclic : \n'
        f'\t||r(A) - c(A)||_1 = {np.linalg.norm(r_random_cyclic - c_random_cyclic, 1)},\n'
        f'Greedy : \n'
        f'\t||r(A) - c(A)||_1 = {np.linalg.norm(r_greedy - c_greedy, 1)},\n'
        f'Random : \n'
        f'\t||r(A) - c(A)||_1 = {np.linalg.norm(r_random - c_random, 1)},\n\n'
        f'All results are the same ? {results_are_the_same}'
    )
    return all_variants_converged, results_are_the_same


if __name__ == '__main__':
    test_osborne_variants()
