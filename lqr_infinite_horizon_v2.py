import numpy as np
from numpy.linalg import multi_dot, inv
from typing import List, Tuple
from scipy.linalg import solve_discrete_are
from scipy.stats import multivariate_normal as mvn
from utils import *


def simulate(x0, A, B, K, Q, R, Wctrl, Wdyn, N=10, seed=None, add_noise=False):
    if seed is not None:
        np.random.seed(np.random.randint(0, 1000))
    n, m = B.shape
    x = x0
    xs = []
    us = []
    cost = 0.0
    for _ in range(N):
        u = K@x
        if add_noise:
            u += Wctrl*np.random.randn(m)
        xs.append(x)
        us.append(u)
        cost += (x@Q@x + u@R@u)/N
        x = A@x + B@u + np.random.multivariate_normal(np.zeros(n), Wdyn)
    xs = np.array(xs)
    us = np.array(us)

    return cost, xs, us


def _loss_imitation_learning(xs, us, xs_true, us_true, Q_true, R_true):
    return np.sum([(x.T - x_true.T)@Q_true@(x.T - x_true) for x, x_true in zip(xs, xs_true)]) \
           + np.sum([(u.T - u_true.T)@R_true@(u.T - u_true) for u, u_true in zip(us, us_true)])


def loss_imitation_learning(xs_true, us_true, A, B, K, Q, R, Q_true, R_true, W_ctrl, W_dyn, average_over=100,
                            add_noise=True):
    losses = []
    x0 = xs_true[0]
    for _ in range(average_over):
        xs, us, _ = simulate(x0, A, B, K, Q, R, W_ctrl, W_dyn, N=10, add_noise=add_noise)
        loss = _loss_imitation_learning(xs, us, xs_true, us_true, Q_true=Q_true, R_true=R_true)
        losses.append(loss)
    return np.nanmean(losses)
