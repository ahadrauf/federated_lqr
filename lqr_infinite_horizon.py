import numpy as np
from numpy.linalg import multi_dot, inv
from typing import List, Tuple
from scipy.linalg import solve_discrete_are
from scipy.stats import multivariate_normal as mvn
from utils import *


class LQR:
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 cov_dyn: np.ndarray, cov_ctrl: np.ndarray):
        assert is_positive_semidefinite(Q)
        assert is_symmetric(Q)
        assert is_positive_definite(R)
        assert is_symmetric(R)
        assert np.shape(A)[0] == np.shape(cov_dyn)[0]  # dimension of state
        assert np.shape(B)[1] == np.shape(cov_ctrl)[0]  # number of inputs

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.cov_dyn = cov_dyn
        self.cov_ctrl = cov_ctrl

    def simulate(self, x0: np.ndarray, N: int, seed=None, add_noise=False, Q=None, R=None, K=None) -> \
            Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, float]]:
        """
        Generate trajectory (given dynamics and controller noise) starting at x0 and continuing for N time steps

        :param x0: Initial position
        :param N: Number of time steps to traverse over
        :param seed: Numpy seed (to add some variance on simulations)
        :param add_noise: Whether to add noise in the simulation
        :return: (x_t, u_t, [K_t, J_t]), t = 0, ..., N-1 (or N for x_t)
        """
        if seed is not None:
            np.random.seed(seed)
        if Q is None:
            Q = self.Q
        if R is None:
            R = self.R
        P = solve_discrete_are(self.A, self.B, Q, R)
        if K is None:
            K = -inv(R + self.B.T@P@self.B)@self.B.T@P@self.A

        xs = [x0]
        us = []
        J = 0.
        metadata = []
        x = x0
        for t in range(N):
            if add_noise:
                u = mvn.rvs(np.ndarray.flatten(K@x), self.cov_ctrl)
            else:
                u = K@x
            u = np.reshape(u, (self.B.shape[1], 1))
            if add_noise:
                x_new = mvn.rvs(np.ndarray.flatten(self.A@x + self.B@u), self.cov_dyn)
            else:
                x_new = self.A@x + self.B@u
            x_new = np.reshape(x_new, (np.size(x_new), 1))

            xs.append(x_new)
            us.append(u)

            # Calculate some metadata for postprocessing
            # J = 0.5*x.T@P@x
            J += (x.T@self.Q@x + u.T@self.R@u)/N  # The true cost
            x = x_new

        metadata = (K, J[0, 0])
        return np.array(xs), np.array(us), metadata

    def loss_imitation_learning(self, xs, us, xs_true, us_true):
        return np.sum([(x.T - x_true.T)@self.Q@(x.T - x_true) for x, x_true in zip(xs, xs_true)]) \
               + np.sum([(u.T - u_true.T)@self.R@(u.T - u_true) for u, u_true in zip(us, us_true)])
