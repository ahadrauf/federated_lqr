# Source: https://github.com/cvxgrp/lfd_lqr/blob/master/algorithms.py
import warnings

import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt

def _ADMM(L, LPQR, r, rPQR, A, B, P, Q, R, niter=50, rho=1, plot=False):
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
    s = np.zeros((m, n))
    history = {k: [] for k in 'KPQRYs'}

    try:
        import mosek
        solver = cp.MOSEK
    except:
        print("Solver MOSEK is not installed, falling back to SCS.", flush=True)
        solver = cp.SCS

    losses = []
    for k in range(niter):
        # K step
        Kcp = cp.Variable((m, n))
        M = cp.vstack([
            Q + A.T@P@(A + B@Kcp) - P,
            R@Kcp + B.T@P@(A + B@Kcp) - s
        ])
        objective = cp.Minimize(L(Kcp) + r(Kcp) + cp.trace(Y.T@M) + rho/2*cp.sum_squares(M))
        prob = cp.Problem(objective, r(Kcp))
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

        # R step
        Rcp = cp.Variable((m, m), PSD=True)
        M = cp.vstack([
            Q + A.T@P@(A + B@K) - P,
            Rcp@K + B.T@P@(B@K + A) - s
        ])
        objective = cp.Minimize(LPQR(Q, Rcp) + rPQR(Q, Rcp) + cp.trace(Y.T@M) +
                                rho/2*cp.sum_squares(M))
        prob = cp.Problem(objective, [Rcp>>np.eye(m)])
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
        R = Rcp.value


        # P, Q, R, s step
        Pcp = cp.Variable((n, n), PSD=True)
        Qcp = cp.Variable((n, n), PSD=True)
        # Rcp = cp.Variable((m, m), PSD=True)
        scp = cp.Variable((m, n))
        M = cp.vstack([
            Qcp + A.T@Pcp@(A + B@K) - Pcp,
            R@K + B.T@Pcp@(B@K + A) - scp
        ])
        # M = cp.vstack([
        #     Qcp + A.T@Pcp@(A + B@K) - Pcp,
        #     Rcp@K + B.T@Pcp@(B@K + A) - scp
        # ])
        objective = cp.Minimize(LPQR(Qcp, R) + rPQR(Qcp, R) + 20*r(scp) + cp.trace(Y.T@M) +
                                rho/2*cp.sum_squares(M))
        prob = cp.Problem(objective, [Pcp>>0, Qcp>>0])
        # objective = cp.Minimize(LPQR(Qcp, Rcp) + rPQR(Qcp, Rcp) + r(scp) + cp.trace(Y.T@M) +
        #                         rho/2*cp.sum_squares(M))
        # prob = cp.Problem(objective, [Pcp>>0, Qcp>>0, Rcp>>np.eye(m)])
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
        # R = Rcp.value
        s = scp.value
        # print('Inside ADMM:', np.shape(K), np.shape(P), np.shape(Q), np.shape(R), np.shape(A), np.shape(B), flush=True)

        # Y step
        residual = np.vstack([
            Q + A.T@P@(A + B@K) - P,
            R@K + B.T@P@(A + B@K) - s
        ])
        Y = Y + rho*residual

        M = cp.vstack([
            Q + A.T@P@(A + B@K) - P,
            R@K + B.T@P@(B@K + A) - s
        ])
        # print(Y)
        # print(M)
        # print(cp.trace(Y.T@M).value)
        losses.append(L(K).value + LPQR(Q, R).value + r(K).value + rPQR(Q, R).value + r(s).value + cp.trace(
            Y.T@M).value + rho/2*cp.sum_squares(M).value)
        history['K'].append(K)
        history['P'].append(P)
        history['Q'].append(Q)
        history['R'].append(R)
        history['s'].append(s)
        # print(L(K).value)

    R = (R + R.T)/2
    Q = (Q + Q.T)/2

    w, v = np.linalg.eigh(R)
    w[w < 1e-6] = 1e-6
    R = v@np.diag(w)@v.T

    w, v = np.linalg.eigh(Q)
    w[w < 0] = 0
    Q = v@np.diag(w)@v.T

    P = solve_discrete_are(A, B, Q, R)

    if plot:
        fig = plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.xlabel("Iteration #")
        plt.ylabel("ADMM Loss")
        plt.title("ADMM Loss vs. Iterations")
        plt.grid(True)
        for k in history:
            print(k)
            print(history[k])
            print()
        plt.show()

    return -np.linalg.solve(R + B.T@P@B, B.T@P@A), P, Q, R


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
    r_obj = r(Kpf)
    cp.Problem(cp.Minimize(L(Kpf) + r_obj)).solve()

    return Kpf.value


def policy_fitting_with_a_kalman_constraint(L, r, A, B, n_random=5, niter=50, rho=10,
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
    K, P, Q, R = _ADMM(L, LPQR, r, rPQR, A, B, P, Q, R, niter=niter, rho=rho, plot=False)

    best_K, bP, bQ, bR = K, P, Q, R
    best_L = evaluate_L(K)

    if P0 is not None and R0 is not None and Q0 is not None:
        K, P, Q, R = _ADMM(L, LPQR, r, rPQR, A, B, P0, Q0, R0, niter=niter, rho=rho)

        L_K = evaluate_L(K)
        if L_K < best_L:
            best_L = L_K
            best_K, bP, bQ, bR = K, P, Q, R

    # run n_random random initializations; keep best
    for iter in range(n_random):
        P = 1./np.sqrt(n)*np.random.randn(n, n)
        Q = 1./np.sqrt(n)*np.random.randn(n, n)
        R = 1./np.sqrt(m)*np.random.randn(m, m)
        P = P.T@P
        Q = Q.T@Q
        R = R.T@R
        K, P, Q, R = _ADMM(L, LPQR, r, rPQR, A, B, P, Q, R,
                           niter=niter, rho=rho)
        L_K = evaluate_L(K)
        if L_K < best_L:
            best_L = L_K
            best_K, bP, bQ, bR = K, P, Q, R

    return best_K, bP, bQ, bR
