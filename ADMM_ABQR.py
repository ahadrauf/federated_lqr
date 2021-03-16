# Source: https://github.com/cvxgrp/lfd_lqr/blob/master/algorithms.py
import warnings

import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are


def _ADMM(LK, LPQR, rK, rPQR, rAB, rs123, A, B, P, Q, R, niter=50, rho=1):
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

    Y = np.zeros((n + m, n))
    F = P@A
    G = P@B
    s1 = np.zeros((n, n))
    s2 = np.zeros((n, m))
    s3 = np.zeros((m, m))

    try:
        import mosek
        solver = cp.MOSEK
    except:
        print("Solver MOSEK is not installed, falling back to SCS.", flush=True)
        solver = cp.SCS

    def solve_subproblem(prob):
        try:
            prob.solve(solver=solver)
        except:
            try:
                print("Defaulting to SCS solver for PQR step", flush=True)
                prob.solve(solver=cp.SCS, acceleration_lookback=0, max_iters=10000)
            except:
                print("SCS solver failed", flush=True)
                Kinf = np.inf*np.ones((m, n))
                Ainf = np.inf*np.ones((n, n))
                Binf = np.inf*np.ones((n, m))
                Pinf = np.inf*np.ones((n, n))
                Qinf = np.inf*np.ones((n, n))
                Rinf = np.inf*np.ones((m, m))
                return Ainf, Binf, Kinf, Pinf, Qinf, Rinf

    for k in range(niter):
        # K step
        Kcp = cp.Variable((m, n))
        M = cp.vstack([
            Q + A.T@F + A.T@G@Kcp - P,
            R@Kcp + B.T@F + B.T@G@Kcp - s3
        ])
        objective = cp.Minimize(LK(Kcp) + rK(Kcp) + cp.trace(Y.T@M) + rho/2*cp.sum_squares(M))
        solve_subproblem(cp.Problem(objective))
        K = Kcp.value

        # A step
        Acp = cp.Variable((n, n))
        M = cp.vstack([
            Q + Acp.T@F + Acp.T@G@K - P,
            R@K + B.T@F + B.T@G@K - s3
        ])
        objective = cp.Minimize(rAB(Acp) + cp.trace(Y.T@M) + rho/2*(cp.sum_squares(M)))
        constraints = [F - P@Acp == s1]
        solve_subproblem(cp.Problem(objective, constraints))
        A = Acp.value

        # B step
        Bcp = cp.Variable((n, m))
        M = cp.vstack([
            Q + A.T@F + A.T@G@K - P,
            R@K + B.T@F + B.T@G@K - s3
        ])
        objective = cp.Minimize(rAB(Bcp) + cp.trace(Y.T@M) + rho/2*(cp.sum_squares(M)))
        constraints = [G - P@Bcp == s2]
        solve_subproblem(cp.Problem(objective, constraints))
        B = Bcp.value

        # F step
        Fcp = cp.Variable((n, n))
        M = cp.vstack([
            Q + A.T@Fcp + A.T@G@K - P,
            R@K + B.T@Fcp + B.T@G@K - s3
        ])
        objective = cp.Minimize(cp.trace(Y.T@M) + rho/2*(cp.sum_squares(M)))
        constraints = [Fcp - P@A == s1]
        solve_subproblem(cp.Problem(objective, constraints))
        F = Fcp.value

        # F step
        Gcp = cp.Variable((n, n))
        M = cp.vstack([
            Q + A.T@F + A.T@Gcp@K - P,
            R@K + B.T@F + B.T@Gcp@K - s3
        ])
        objective = cp.Minimize(cp.trace(Y.T@M) + rho/2*(cp.sum_squares(M)))
        constraints = [Gcp - P@B == s2]
        solve_subproblem(cp.Problem(objective, constraints))
        G = Gcp.value

        # P, Q, R, s1, s2, s3 step
        Pcp = cp.Variable((n, n), PSD=True)
        Qcp = cp.Variable((n, n), PSD=True)
        Rcp = cp.Variable((m, m), PSD=True)
        s1cp = cp.Variable((n, n))
        s2cp = cp.Variable((n, m))
        s3cp = cp.Variable((m, m))
        M = cp.vstack([
            Q + A.T@F + A.T@G@K - P,
            R@K + B.T@F + B.T@G@K - s3cp
        ])
        objective = cp.Minimize(LPQR(Qcp, Rcp) + rPQR(Qcp, Rcp) + rs123(s1cp, s2cp, s3cp) + cp.trace(Y.T@M) +
                                rho/2*cp.sum_squares(M))
        constraints = [Pcp >> 0, Qcp >> 0, Rcp >> np.eye(m),
                       F - Pcp@A == s1cp,
                       G - Pcp@A == s2cp]
        solve_subproblem(cp.Problem(objective, constraints))
        P = Pcp.value
        Q = Qcp.value
        R = Rcp.value
        s1 = s1cp.value
        s2 = s2cp.value
        s3 = s3cp.value

        # Y step
        residual = np.vstack([
            Q + A.T@F + A.T@G@K - P,
            R@K + B.T@F + B.T@G@K - s3
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

    return A, B, -np.linalg.solve(R + B.T@P@B, B.T@P@A), P, Q, R


def policy_fitting(L, r, n, m):
    """
    Traditional policy fitting (no ADMM)
    :param L: L(K), Loss function
    :param r: r(K), regularization term
    :param xs: Array of observed states (N x n)
    :param us: Array of observed inputs (N x m)
    :return: Kcp (gain matrix found by policy fitting) (m x n)
    """
    Kpf = cp.Variable((m, n))
    r_obj, r_cons = r(Kpf)
    cp.Problem(cp.Minimize(L(Kpf) + r_obj), r_cons).solve()

    return Kpf.value


def policy_fitting_with_a_kalman_constraint(LK, rK, A0, B0, n_random=5, niter=50, rho=1,
                                            P0=None, Q0=None, R0=None, LPQR=None, rPQR=None, rAB=None, rs123=None):
    """
    Wrapper around _ADMM.
    """
    n, m = B0.shape

    def evaluate_L(K):
        Kcp = cp.Variable((m, n))
        Kcp.value = K
        return LK(Kcp).value

    if LPQR is None:
        LPQR = lambda Q, R: cp.Constant(0)
    if rPQR is None:
        rPQR = lambda Q, R: cp.Constant(0)
    if rAB is None:
        rsAB = lambda A, B: cp.Constant(0)
    if rs123 is None:
        rs123 = lambda s1, s2, s3: cp.Constant(0)

    # solve with zero initialization
    P = np.zeros((n, n))
    Q = np.zeros((n, n))
    R = np.zeros((m, m))
    A, B, K, P, Q, R = _ADMM(LK, LPQR, rK, rPQR, rAB, rs123, A0, B0, P, Q, R, niter=niter, rho=rho)

    bA, bB, best_K, bP, bQ, bR = A, B, K, P, Q, R
    best_L = evaluate_L(K)

    if P0 is not None and R0 is not None and Q0 is not None:
        A, B, K, P, Q, R = _ADMM(LK, LPQR, rK, rPQR, rAB, rs123, A0, B0, P, Q, R, niter=niter, rho=rho)

        L_K = evaluate_L(K)
        if L_K < best_L:
            best_L = L_K
            bA, bB, best_K, bP, bQ, bR = A, B, K, P, Q, R

    # run n_random random initializations; keep best
    for iter in range(n_random):
        A = 1./np.sqrt(n)*np.random.randn(n, n)
        B = 1./np.power(n*m, 0.25)*np.random.randn(n, m)
        P = 1./np.sqrt(n)*np.random.randn(n, n)
        Q = 1./np.sqrt(n)*np.random.randn(n, n)
        R = 1./np.sqrt(m)*np.random.randn(m, m)
        P = P.T@P
        Q = Q.T@Q
        R = R.T@R
        A, B, K, P, Q, R = _ADMM(LK, LPQR, rK, rPQR, rAB, rs123, A, B, P, Q, R, niter=niter, rho=rho)
        L_K = evaluate_L(K)
        if L_K < best_L:
            best_L = L_K
            bA, bB, best_K, bP, bQ, bR = A, B, K, P, Q, R

    return bA, bB, best_K, bP, bQ, bR
