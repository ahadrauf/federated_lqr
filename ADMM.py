# Source: https://github.com/cvxgrp/lfd_lqr/blob/master/algorithms.py
import warnings

import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are


def _ADMM(L, r, xs, us, A, B, P0=None, Q0=None, R0=None, niter=50, rho=1):
    """
    Policy fitting with a Kalman constraint.
    Args:
        - L: function that takes in a cvxpy Variable
            and returns a cvxpy expression representing the objective.
        - r: function that takes in a cvxpy Variable
            and returns a cvxpy expression and a list of constraints
            representing the regularization function.
        - xs: N x n matrix of states.
        - us_observed: N x m matrix of inputs.
        - A: n x n dynamics matrix.
        - B: n x m dynamics matrix.
        - P: n x n PSD matrix, the initial PSD cost-to-go coefficient.
        - Q: n x n PSD matrix, the initial state cost coefficient.
        - R: n x n PD matrix, the initial input cost coefficient.
        - niter: int (optional). Number of iterations (default=50).
        - rho: double (optional). Penalty parameter (default=1).
    Returns:
        - K: m x n gain matrix found by policy fitting with a Kalman constraint.
    """
    n, m = B.shape

    try:
        import mosek
        solver = cp.MOSEK
    except:
        warnings.warn("Solver MOSEK is not installed, falling back to SCS.")
        solver = cp.SCS

    # Using warm start to hopefully speed up solve times --> need to define K and (P, Q, R) step problems
    # K step
    Kcp_K = cp.Variable((m, n))
    Pcp_K = cp.Parameter((n, n), PSD=True)
    Qcp_K = cp.Parameter((n, n), PSD=True)
    Rcp_K = cp.Parameter((m, m), PSD=True)
    Ycp = cp.Parameter((n + m, n))
    M_K = cp.vstack([
        Qcp_K + A.T@Pcp_K@(A + B@Kcp_K) - Pcp_K,
        Rcp_K@Kcp_K + B.T@Pcp_K@(A + B@Kcp_K)
    ])
    objective_K = cp.Minimize(L(Kcp_K) + r(Kcp_K) + cp.trace(Ycp.T@M_K) + rho/2*cp.sum_squares(M_K))
    prob_K = cp.Problem(objective_K)

    # PQR step
    Kcp_PQR = cp.Parameter((m, n))
    Pcp_PQR = cp.Variable((n, n), PSD=True)
    Qcp_PQR = cp.Variable((n, n), PSD=True)
    Rcp_PQR = cp.Variable((m, m), PSD=True)
    M_PQR = cp.vstack([
        Qcp_PQR + A.T@Pcp_PQR@(A + B@Kcp_PQR) - Pcp_PQR,
        Rcp_PQR@Kcp_PQR + B.T@Pcp_PQR@(A + B@Kcp_PQR)
    ])
    objective_PQR = cp.Minimize(cp.trace(Ycp.T@M_PQR) + rho/2*cp.sum_squares(M_PQR))
    prob_PQR = cp.Problem(objective_PQR)

    # Initialize K step parameters
    def rand_initialization(m, n):
        A = 1./np.sqrt(n)*np.random.randn(n, n)
        return A.T@A
        # return (A + A.T)/2
    Pcp_K.value = rand_initialization(n, n) if P0 is None else P0
    Qcp_K.value = rand_initialization(n, n) if Q0 is None else Q0
    Rcp_K.value = rand_initialization(m, m) if R0 is None else R0
    Ycp.value = np.zeros((n + m, n))

    for k in range(niter):
        # K step
        prob_K.solve(solver=solver)
        Kcp_PQR.value = Kcp_K.value

        # P, Q, R step
        try:
            prob_PQR.solve(solver=solver)
        except:
            warnings.warn("Defaulting to SCS solver for PQR step")
            prob_PQR.solve(solver=cp.SCS, acceleration_lookback=0, max_iters=10000)
        Pcp_K.value = Pcp_PQR.value
        Qcp_K.value = Qcp_PQR.value
        Rcp_K.value = Rcp_PQR.value

        # Y step
        residual = np.vstack([
            Qcp_PQR.value + A.T@Pcp_PQR.value@(A + B@Kcp_K.value) - Pcp_PQR.value,
            Rcp_PQR.value@Kcp_K.value + B.T@Pcp_PQR.value@(A + B@Kcp_K.value)
        ])
        Ycp.value = Ycp.value + rho*residual

    return Kcp_K.value, Pcp_PQR.value, Qcp_PQR.value, Rcp_PQR.value


def policy_fitting(L, r, xs, us):
    """
    Traditional policy fitting (no ADMM)
    :param L: L(K), Loss function
    :param r: r(K), regularization term
    :param xs: Array of observed states (N x n)
    :param us: Array of observed inputs (N x m)
    :return: Kcp (gain matrix found by policy fitting) (m x n)
    """
    n = xs.shape[1]
    m = us.shape[1]
    Kcp = cp.Variable((m, n))
    cp.Problem(cp.Minimize(L(Kcp) + r(Kcp))).solve()
    return Kcp.value

def policy_fitting_with_kalman_constraint(L, r, xs, us, A, B, P0=None, Q0=None, R0=None, niter=50, rho=1):
    """
    Traditional policy fitting (no ADMM)
    :param L: L(K), Loss function
    :param r: r(K), regularization term
    :param xs: Array of observed states (N x n)
    :param us: Array of observed inputs (N x m)
    :return: Kcp (gain matrix found by policy fitting) (m x n)
    """
    return _ADMM(L, r, xs, us, A, B, P0, Q0, R0, niter, rho)