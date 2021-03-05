# Source: https://github.com/cvxgrp/lfd_lqr/blob/master/algorithms.py
import warnings

import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are

def _ADMM(L, LPQR, r, rPQR, xs, us, A, B, P, Q, R, niter=50, rho=1):
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
        print("Solver MOSEK is not installed, falling back to SCS.", flush=True)
        solver = cp.SCS

    for k in range(niter):
        # K step
        Kcp = cp.Variable((m, n))
        r_obj, r_cons = r(Kcp)
        M = cp.vstack([
            Q + A.T@P@(A + B@Kcp) - P,
            R@Kcp + B.T@P@(A + B@Kcp)
        ])
        objective = cp.Minimize(L(Kcp) + r_obj + cp.trace(Y.T@M) + rho/2*cp.sum_squares(M))
        prob = cp.Problem(objective, r_cons)
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
        K = Kcp.value

        # P, Q, R step
        Pcp = cp.Variable((n, n), PSD=True)
        Qcp = cp.Variable((n, n), PSD=True)
        Rcp = cp.Variable((m, m), PSD=True)
        M = cp.vstack([
            Qcp + A.T@Pcp@(A + B@K) - Pcp,
            Rcp@K + B.T@Pcp@(B@K + A)
        ])
        objective = cp.Minimize(LPQR(Qcp, Rcp) + rPQR(Qcp, Rcp) + cp.trace(Y.T@M) +
                                rho/2*cp.sum_squares(M))
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
    Kpf = cp.Variable((m, n))
    r_obj, r_cons = r(Kpf)
    cp.Problem(cp.Minimize(L(Kpf) + r_obj), r_cons).solve()

    return Kpf.value


def policy_fitting_with_a_kalman_constraint(L, r, xs, us_observed, A, B, n_random=5, niter=50, rho=1,
                                            P0=None, Q0=None, R0=None, LPQR=None, rPQR=None):
    """
    Wrapper around _ADMM.
    """
    n, m = B.shape

    def evaluate_L(K):
        Kcp = cp.Variable((m, n))
        Kcp.value = K
        loss = L(K)
        return loss.value

    if LPQR is None:
        LPQR = lambda Q, R: cp.Constant(0)
    if rPQR is None:
        rPQR = lambda Q, R: cp.Constant(0)

    # solve with zero initialization
    P = np.zeros((n, n))
    Q = np.zeros((n, n))
    R = np.zeros((m, m))
    K, P, Q, R = _ADMM(L, LPQR, r, rPQR, xs, us_observed, A, B, P, Q, R, niter=niter, rho=rho)

    best_K, bP, bQ, bR = K, P, Q, R
    best_L = evaluate_L(K)

    # run n_random random initializations; keep best
    for iter in range(n_random):
        # if iter < n_random / 2 and P0 is not None:
        #     dP = 1./n/n*np.random.randn(n, n)
        #     dQ = 1./n/n*np.random.randn(n, n)
        #     dR = 1./m/m*np.random.randn(m, m)
        #     P = P0 + dP.T@dP
        #     Q = Q0 + dQ.T@dQ
        #     R = R0 + dR.T@dR
        # else:
        P = 1./np.sqrt(n)*np.random.randn(n, n)
        Q = 1./np.sqrt(n)*np.random.randn(n, n)
        R = 1./np.sqrt(m)*np.random.randn(m, m)
        P = P.T@P
        Q = Q.T@Q
        R = R.T@R
        K, P, Q, R = _ADMM(L, LPQR, r, rPQR, xs, us_observed, A, B, P, Q, R,
                           niter=niter, rho=rho)
        L_K = evaluate_L(K)
        if L_K < best_L:
            best_L = L_K
            best_K, bP, bQ, bR = K, P, Q, R

    return best_K, bP, bQ, bR
