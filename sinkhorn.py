import numpy as np
from matrix_utils import rowsum, rowsum_k, colsum, colsum_k, logsumexp


def rho(a, b):
    """
    The penalty function used in Greenkhorn. Analogous to KL-divergence
    """
    return b - a + a * np.log(a / b)


def sinkhorn_log(mat: np.ndarray,
                 a: np.ndarray,
                 b: np.ndarray,
                 max_iter: int = 1000,
                 check_every: int = 100,
                 tol: float = 1e-9):
    """
    Sinkhorn algorithm for matrix scaling, in stable log-domain implementation.

    The algorithm updates vector scalings u and v such that diag(u) @ K @ diag(v) has rows that sum to a, and
    columns that sum to b, where K is the Gibbs kernel associated to the (regularized) OT problem.

    :param mat: The matrix to scale input in log-domain. Typically a kernel -C/epsilon with C a cost matrix.
    :param a: the row-wise objective (the distribution to send mass from)
    :param b: the column-wise objective (the distribution to send mass to)
    :param max_iter: max number of scaling iterations
    :param check_every: period of check of early stopping criterion ||r(A) - r||_1 + ||c(A) - c||_1 < tol
    :param tol: stopping criterion threshold
    :return: the diagonal scalings u, v, and a history if requested.
    """
    n = mat.shape[0]
    log_v = np.zeros(n)  # log-domain
    log_a = np.log(a)
    log_b = np.log(b)

    for it in range(1, max_iter + 1):
        log_u = log_a - logsumexp(mat + log_v[None, :], 1)
        log_v = log_b - logsumexp(mat + log_u[:, None], 0)

        if it % check_every == 0:
            log_scaled_mat = log_u[:, None] + mat + log_v[None, :]
            r = np.exp(rowsum(log_scaled_mat, log_domain=True))
            c = np.exp(colsum(log_scaled_mat, log_domain=True))
            if np.linalg.norm(r - a, 1) + np.linalg.norm(c - b, 1) < tol:
                break

    else:
        print('[WARNING] sinkhorn_log did not converge')

    return np.exp(log_u), np.exp(log_v), it


def greenkhorn(cost_mat: np.ndarray,
               reg: float,
               a: np.ndarray,
               b: np.ndarray,
               max_iter: int = 1000,
               check_every: int = 100,
               tol: float = 1e-9):
    """
    Greedy Sinkhorn algorithm.

    From "Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration", by Altschuler et al.
    https://arxiv.org/abs/1705.09634

    :param cost_mat: the cost matrix of the OT problem
    :param reg: the regularization parameter
    :param a: the row-wise objective (the distribution to send mass from)
    :param b: the column-wise objective (the distribution to send mass to)
    :param max_iter: max number of scaling iterations
    :param check_every: period of check of early stopping criterion ||r(A) - r||_1 + ||c(A) - c||_1 < tol
    :param tol: stopping criterion threshold
    :return: the diagonal scalings u, v
    """
    n = cost_mat.shape[0]
    u, v = np.ones(n, dtype=float), np.ones(n, dtype=float)
    kernel = np.exp(-cost_mat / reg)
    kernel /= np.linalg.norm(kernel, 1)
    r = rowsum(kernel)
    c = colsum(kernel)
    err_r, err_c = rho(a, r), rho(b, c)

    for it in range(1, max_iter + 1):
        imax = np.argmax(err_r)
        jmax = np.argmax(err_c)
        if err_r[imax] > err_c[jmax]:
            # record scaling
            u_old = u[imax]
            u[imax] = a[imax] / np.dot(kernel[imax, :], v)
            err_r[imax] = 0.

            # update the column sums
            c = c + (u[imax] - u_old) * kernel[imax, :] * v
            err_c = rho(b, c)

            # update the row sum
            r[imax] = a[imax]
        else:
            # record scaling
            v_old = v[jmax]
            v[jmax] = b[jmax] / np.dot(kernel[:, jmax], u)
            err_c[jmax] = 0.

            # update the row sums
            r = r + (v[jmax] - v_old) * kernel[:, jmax] * u
            err_r = rho(a, r)

            # update the column sum
            c[jmax] = b[jmax]

        if it % check_every == 0:
            if np.linalg.norm(r - a, 1) + np.linalg.norm(c - b, 1) < tol:
                break
    else:
        print('[WARNING] greenkhorn did not converge')

    return u, v, it
