# Source: https://github.com/cvxgrp/lfd_lqr/blob/master/algorithms.py
import warnings

import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are
import sys


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

    # The matrix formulations as in the paper are DCP compliant but not DPP compliant, which slows down operation
    # Below is a slight reformulation to make them DPP compliant for faster solving using CVXPY
    # (the problematic parts are the cp.trace(Ycp.T@M) terms, which involves multiplying a parameter with other
    # parameters)
    # More details: https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming
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
    M_Ktemp = cp.Variable((n + m, n))
    objective_K = cp.Minimize(L(Kcp_K, Qcp_K, Rcp_K) + r(Kcp_K, Qcp_K, Rcp_K) + cp.trace(Ycp.T@M_K) +
                              rho/2*cp.sum_squares(M_K))
    constraint_K = []
    # print("K step:", objective_K.is_dcp(dpp=True), objective_K.is_dcp(dpp=False))
    # print("K step:", objective_K.is_dgp(dpp=True), objective_K.is_dgp(dpp=False))
    prob_K = cp.Problem(objective_K, constraint_K)

    # PQR step
    Kcp_PQR = cp.Parameter((m, n))
    Pcp_PQR = cp.Variable((n, n), PSD=True)
    Qcp_PQR = cp.Variable((n, n), PSD=True)
    Rcp_PQR = cp.Variable((m, m), PSD=True)
    M_PQR = cp.vstack([
        Qcp_PQR + A.T@Pcp_PQR@(A + B@Kcp_PQR) - Pcp_PQR,
        Rcp_PQR@Kcp_PQR + B.T@Pcp_PQR@(A + B@Kcp_PQR)
    ])
    M_PQRtemp = cp.Variable((n + m, n))
    objective_PQR = cp.Minimize(L(Kcp_K, Qcp_K, Rcp_K) + r(Kcp_K, Qcp_K, Rcp_K) + cp.trace(Ycp.T@M_PQR) + \
                                rho/2*cp.sum_squares(M_PQR))
    constraint_PQR = [Pcp_PQR>>0, Qcp_PQR>>0, Rcp_PQR>>np.eye(m)]
    # print("PQR step:", objective_PQR.is_dcp(dpp=True), objective_PQR.is_dcp(dpp=False))
    # print("PQR step:", objective_PQR.is_dgp(dpp=True), objective_PQR.is_dgp(dpp=False))
    prob_PQR = cp.Problem(objective_PQR, constraint_PQR)

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
        # prob_K.solve(solver=solver)
        try:
            prob_K.solve(solver=solver)
        except:
            try:
                print("Defaulting to SCS solver for K step", flush=True)
                prob_K.solve(solver=cp.SCS, acceleration_lookback=0, max_iters=10000)
            except:
                print("SCS solver failed", flush=True)
                Kinf = np.inf*np.ones((m, n))
                Pinf = np.inf*np.ones((n, n))
                Qinf = np.inf*np.ones((n, n))
                Rinf = np.inf*np.ones((m, m))
                return Kinf, Pinf, Qinf, Rinf
        Kcp_PQR.value = Kcp_K.value

        # P, Q, R step
        print('L', L(Kcp_K.value, Qcp_K.value, Rcp_K.value))
        print('r', r(Kcp_K.value, Qcp_K.value, Rcp_K.value))
        try:
            prob_PQR.solve(solver=solver)
        except:
            try:
                print("Defaulting to SCS solver for PQR step", flush=True)
                prob_PQR.solve(solver=cp.SCS, acceleration_lookback=0, max_iters=10000)
            except:
                print("SCS solver failed", flush=True)
                Kinf = np.inf*np.ones((m, n))
                Pinf = np.inf*np.ones((n, n))
                Qinf = np.inf*np.ones((n, n))
                Rinf = np.inf*np.ones((m, m))
                return Kinf, Pinf, Qinf, Rinf
        # print(Pcp_PQR.value, Qcp_PQR.value, Rcp_PQR.value)
        Pcp_K.value = _ensure_symmetric_psd(Pcp_PQR.value, threshold=0)
        Qcp_K.value = _ensure_symmetric_psd(Qcp_PQR.value, threshold=0)
        Rcp_K.value = _ensure_symmetric_psd(Rcp_PQR.value, threshold=1e-6)

        # Y step
        residual = np.vstack([
            Qcp_PQR.value + A.T@Pcp_PQR.value@(A + B@Kcp_K.value) - Pcp_PQR.value,
            Rcp_PQR.value@Kcp_K.value + B.T@Pcp_PQR.value@(A + B@Kcp_K.value)
        ])
        Ycp.value = Ycp.value + rho*residual

    return Kcp_K.value, Pcp_PQR.value, Qcp_PQR.value, Rcp_PQR.value


def _policy_fitting_with_a_kalman_constraint_extra(LK, LQR, rK, rQR, xs, us_observed, A, B, P, Q, R, niter=50, rho=1):
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

    K = np.zeros((m, n))
    Y = np.zeros((n + m, n))

    try:
        import mosek
        solver = cp.MOSEK
    except:
        warnings.warn("Solver MOSEK is not installed, falling back to SCS.")
        solver = cp.SCS

    for k in range(niter):
        # K step
        Kcp = cp.Variable((m, n))
        r_obj, r_cons = rK(Kcp)
        M = cp.vstack([
            Q + A.T@P@(A + B@Kcp) - P,
            R@Kcp + B.T@P@(A + B@Kcp)
        ])
        objective = cp.Minimize(LK(Kcp) + r_obj + cp.trace(Y.T@M) + rho/2*cp.sum_squares(M))
        prob = cp.Problem(objective, r_cons)
        # prob.solve(solver=solver)
        try:
            prob.solve(solver=solver)
        except:
            try:
                print("Defaulting to SCS solver for K step", flush=True)
                prob.solve(solver=cp.SCS, acceleration_lookback=0, max_iters=10000)
            except:
                print("SCS solver failed", flush=True)
                Kinf = np.inf*np.ones((m, n))
                Pinf = np.inf*np.ones((n, n))
                Qinf = np.inf*np.ones((n, n))
                Rinf = np.inf*np.ones((m, m))
                return Kinf, Pinf, Qinf, Rinf
        K = Kcp.value

        # P, Q, R step
        Pcp = cp.Variable((n, n), PSD=True)
        Qcp = cp.Variable((n, n), PSD=True)
        Rcp = cp.Variable((m, m), PSD=True)
        M = cp.vstack([
            Qcp + A.T@Pcp@(A + B@K) - Pcp,
            Rcp@K + B.T@Pcp@(B@K + A)
        ])
        objective = cp.Minimize(LQR(Qcp, Rcp) + cp.trace(Y.T@M) + rho/2*cp.sum_squares(M))
        prob = cp.Problem(objective, [Pcp>>0, Qcp>>0, Rcp>>np.eye(m)])
        try:
            prob.solve(solver=solver)
        except:
            try:
                print("Defaulting to SCS solver for PQR step", flush=True)
                prob.solve(solver=cp.SCS, acceleration_lookback=0, max_iters=10000)
            except:
                print("SCS solver failed", flush=True)
                Kinf = np.inf*np.ones((m, n))
                Pinf = np.inf*np.ones((n, n))
                Qinf = np.inf*np.ones((n, n))
                Rinf = np.inf*np.ones((m, m))
                return Kinf, Pinf, Qinf, Rinf
        P = Pcp.value
        Q = Qcp.value
        R = Rcp.value

        # Y step
        residual = np.vstack([
            Q + A.T@P@(A + B@K) - P,
            R@K + B.T@P@(A + B@K)
        ])
        Y = Y + rho*residual

    R = (R + R.T)/2
    Q = (Q + Q.T)/2

    w, v = np.linalg.eigh(R)
    w[w < 1e-6] = 1e-6
    R = v@np.diag(w)@v.T

    w, v = np.linalg.eigh(Q)
    w[w < 0] = 0
    Q = v@np.diag(w)@v.T

    P = solve_discrete_are(A, B, Q, R)

    return -np.linalg.solve(R + B.T@P@B, B.T@P@A), P, Q, R


def policy_fitting_with_a_kalman_constraint_extra(LK, LQR, rK, rQR, xs, us_observed, A, B, n_random=5, niter=50, rho=1,
                                                  P0 = None, Q0 = None, R0 = None):
    """
    Wrapper around _policy_fitting_with_a_kalman_constraint.
    """
    n, m = B.shape

    def evaluate_L(K, Q, R):
        Kcp = cp.Variable((m, n))
        Qcp = cp.Variable((n, n))
        Rcp = cp.Variable((m, m))
        Kcp.value = K
        Qcp.value = Q
        Rcp.value = R
        return LK(Kcp).value + LQR(Qcp, Rcp).value

    # solve with zero initialization
    P = np.zeros((n, n)) if P0 is None else P0
    Q = np.zeros((n, n)) if Q0 is None else Q0
    R = np.zeros((m, m)) if R0 is None else R0
    K, P, Q, R = _policy_fitting_with_a_kalman_constraint_extra(
        LK, LQR, rK, rQR, xs, us_observed, A, B, P, Q, R, niter=niter, rho=rho)

    best_K, bP, bQ, bR = K, P, Q, R
    best_L = evaluate_L(K, Q, R)

    # run n_random random initializations; keep best
    for _ in range(n_random):
        P = 1./np.sqrt(n)*np.random.randn(n, n)
        Q = 1./np.sqrt(n)*np.random.randn(n, n)
        R = 1./np.sqrt(m)*np.random.randn(m, m)
        K, P, Q, R = _policy_fitting_with_a_kalman_constraint_extra(LK, LQR, rK, rQR, xs, us_observed, A, B, P.T@P, Q.T@Q, R.T@R,
                                                            niter=niter,
                                                     rho=rho)
        L_K = evaluate_L(K, Q, R)
        if L_K < best_L:
            best_L = L_K
            best_K, bP, bQ, bR = K, P, Q, R

    return best_K, bP, bQ, bR


def _ensure_symmetric_psd(A, threshold=0.):
    R = (A + A.T)/2

    w, v = np.linalg.eigh(R)
    w[w < threshold] = threshold
    R = v@np.diag(w)@v.T

    return R


def policy_fitting(L, r, xs, us, Q, R):
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
    # if sys.version[:3] == '3.6':
    #     Kcp = cp.Variable(m, n)
    # else:
    Kcp = cp.Variable((m, n))
    cp.Problem(cp.Minimize(L(Kcp, Q, R) + r(Kcp, Q, R))).solve()
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
