import numpy as np
from numpy.linalg import multi_dot, inv
from typing import List, Tuple
from scipy.stats import multivariate_normal as mvn
from scipy.special import erf
from utils import *


class LQR:
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, F: np.ndarray,
                 cov_dyn: np.ndarray, cov_ctrl: np.ndarray):
        # assert is_positive_semidefinite(Q)
        # assert is_symmetric(Q)
        # assert is_positive_definite(R)
        # assert is_symmetric(R)
        # assert is_positive_semidefinite(F)
        # assert is_symmetric(F)
        assert np.shape(A)[0] == np.shape(cov_dyn)[0]  # dimension of state
        assert np.shape(B)[1] == np.shape(cov_ctrl)[0]  # number of inputs

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.F = F
        self.cov_dyn = cov_dyn
        self.cov_ctrl = cov_ctrl

    def generate_trajectory(self, x0: np.ndarray, N: int) -> Tuple[List[np.ndarray], List[np.ndarray],
                                                                   List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Generate trajectory (given dynamics and controller noise) starting at x0 and continuing for N time steps

        :param x0: Initial position
        :param N: Number of time steps to traverse over
        :return: (x_t, u_t, [K_t, J_t]), t = 0, ..., N-1 (or N for x_t)
        """
        # x = x0
        P = self.F

        Ks = []
        Ps = []  # for metadata
        for t in reversed(range(N)):
            A_T = np.transpose(self.A)
            B_T = np.transpose(self.B)
            K = -multi_dot([inv(self.R + multi_dot([B_T, P, self.B])), B_T, P, self.A])
            P = self.Q + multi_dot([A_T, P, self.A]) - \
                multi_dot([multi_dot([A_T, P, self.B]),
                           inv(self.R + multi_dot([B_T, P, self.B])),
                           multi_dot([B_T, P, self.A])])

            Ks.insert(0, K)
            Ps.insert(0, P)

        # traj = []
        xs = [x0]
        us = []
        metadata = []
        x = x0
        for t in range(N):
            u = mvn.rvs(np.dot(Ks[t], x), self.cov_ctrl)
            x_new = mvn.rvs(np.ndarray.flatten(np.dot(self.A, x) + np.dot(self.B, u)), self.cov_dyn)
            x_new = np.reshape(x_new, (np.size(x_new), 1))

            x_new = np.reshape(x_new, np.shape(x))
            xs.append(x_new)
            us.append(u)
            # traj.append((x, u, x_new))

            # Calculate some metadata for postprocessing
            J = 0.5*multi_dot([np.transpose(x), Ps[t], x])
            metadata.append((Ks[t], J))
            x = x_new

        return xs, us, metadata
        # return traj, metadata

    def log_prob_state_transition(self, x: np.ndarray, u: np.ndarray, x_new: np.ndarray, resolution: float = 0.1,
                                  eps_dyn: float = 1e-30) -> np.ndarray:
        """
        Calculates the probability of a state transition

        s_{t+1} ~ Normal(Ax + Bu, cov_dynamics)
        p(s_{t+1} | s_t, a_t) = CDF(s_{t+1} + resolution/2) - CDF(s_{t+1} - resolution/2)
        :param x: x_t
        :param u: u_t
        :param x_new: x_{t+1}
        :param resolution: The resolution used to calculate the CDF difference
        :param eps_dyn: A minimum value for the probability (used to prevent np.log(0.0) errors if a transition is
        virtually impossible)
        :return: probability of state transition i p(s_{t+1} | s_t, a_t)
        """
        mu = np.dot(self.A, x) + np.dot(self.B, u)
        prob = mvn.cdf(x_new + resolution/2, mu, self.cov_dyn) - mvn.cdf(x_new - resolution/2, mu, self.cov_dyn)
        # prob = self.log_normal_pdf(x_new, mu, self.cov_dyn)
        return np.log(np.maximum(prob, eps_dyn))

    def log_prob_action(self, x: np.ndarray, u: np.ndarray, K: np.ndarray, resolution: float = 0.1,
                        eps_ctrl: float = 1e-30) -> np.ndarray:
        """
        Calculates the probability of a state transition

        a_t ~ Normal(K_t*s_t, cov_ctrl)
        p(a_t | s_t) = CDF(a_t + resolution/2) - CDF(a_t - resolution/2)
        :param x: x_t
        :param u: u_t
        :param K: control matrix K_t
        :param resolution: The resolution used to calculate the CDF difference
        :param eps_ctrl: A minimum value for the probability (used to prevent np.log(0.0) errors if an input is
        virtually impossible)
        :return: probability of state transition i p(a_t | s_t)
        """
        mu = np.dot(K, x)
        prob = mvn.cdf(u + resolution/2, mu, self.cov_ctrl) - mvn.cdf(u - resolution/2, mu, self.cov_ctrl)
        # prob = self.log_normal_pdf(u, mu, self.cov_ctrl)
        return np.log(np.maximum(prob, eps_ctrl))

    def log_prob_trajectory(self, xs: List[np.ndarray], us: List[np.ndarray],
                            metadata: List[Tuple[np.ndarray, float]],
                            resolution_dyn: float = 0.1, resolution_ctrl: float = 0.1,
                            eps_dyn: float = 1e-30, eps_ctrl: float = 1e-30) -> np.ndarray:
        """
        Calculates the log probability of a trajectory

        p(traj) = log(p(s0)) + sum_{t=0 to N} log(p(a_t | s_t)) + log(p(s_{t+1} | s_t, a_t))
        :param xs: {x_t}, t=0,...N
        :param us: {u_t}, t=0,...N-1
        :param metadata: List[Tuple[K_t, J_t]]
        :param resolution_dyn: Resolution of prob_state_transition
        :param resolution_ctrl: Resolution of prob_action
        :param eps_dyn: A minimum value for the probability (used to prevent np.log(0.0) errors if a transition is
        virtually impossible)
        :param eps_ctrl: A minimum value for the probability (used to prevent np.log(0.0) errors if an input is
        virtually impossible)
        :return: Log robability of the trajectory
        """
        prob = np.log(1.0)  # p(s0)
        for t in range(len(us)):
            prob += self.log_prob_state_transition(xs[t], us[t], xs[t+1], resolution_dyn, eps_dyn)
            prob += self.log_prob_action(xs[t], us[t], metadata[t][0], resolution_ctrl, eps_ctrl)
        return prob

    def dJ_dQ(self, xs_comp: List[np.ndarray], xs_human: List[np.ndarray]) -> np.ndarray:
        dJ = np.empty_like(self.Q)
        n, m = np.shape(self.Q)
        for i in range(n):
            for j in range(m):
                dJ[i][j] = np.sum([x[i]*x[j] for x in xs_comp]) - np.sum([x[i]*x[j] for x in xs_human])
        return dJ

    def dJ_dR(self, us_comp: List[np.ndarray], us_human: List[np.ndarray]) -> np.ndarray:
        dJ = np.empty_like(self.R)
        n, m = np.shape(self.R)
        for i in range(n):
            for j in range(m):
                dJ[i][j] = np.sum([u[i]*u[j] for u in us_comp]) - np.sum([u[i]*u[j] for u in us_human])
        return -dJ

    def dlog_prob_trajectory_dR(self, xs: List[np.ndarray], us: List[np.ndarray],
                            metadata: List[Tuple[np.ndarray, float]], r: float, S: np.ndarray,
                            resolution_dyn: float = 0.1, resolution_ctrl: float = 0.1,
                            eps_dyn: float = 1e-30, eps_ctrl: float = 1e-30):
        """
        Calculate the two point estimator of dlog_prob_trajectory / dR
        z = d*S

        :param xs:
        :param us:
        :param metadata:
        :param r:
        :param S: perturbation vector sampled from unit sphere
        :param resolution_dyn:
        :param resolution_ctrl:
        :param eps_dyn:
        :param eps_ctrl:
        :return:
        """
        pass

    def loss_imitation_learning(self, xs, us, xs_true, us_true):
        return np.sum([multi_dot([np.transpose(x - x_true), self.Q, x - x_true]) for x, x_true in zip(xs, xs_true)]) \
               + np.sum([multi_dot([np.transpose(u - u_true), self.R, u - u_true]) for u, u_true in zip(us, us_true)])


    @staticmethod
    def cdf(x, mu, var):
        # Taken from Wikipedia: https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
        # est_prob = 0.5*(1+erf((x-mu)/(var*np.sqrt(2))))
        dist = mvn(mean=mu, cov=var*var)
        prob = dist.cdf(x)
        # print(x, mu, var, est_prob, prob)
        return dist.cdf(x)
        # print("CDF:", dist.cdf(np.array([2, 4])))

    @staticmethod
    def log_normal_pdf(x: np.ndarray, mu: np.ndarray, covar: np.ndarray):
        n = np.size(x)
        prob = -0.5*(np.log(np.linalg.norm(covar)) +
                     multi_dot([np.transpose(x - mu), inv(covar), x - mu]) +
                     n*np.log(2*np.pi))
        print(prob)
        return prob

    @staticmethod
    def traj_to_transition_traj(xs: List[np.ndarray], us: List[np.ndarray]) -> \
            List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        transition_traj = []
        for t in range(len(us)):
            transition_traj.append((xs[t], us[t], xs[t+1]))
        return transition_traj
