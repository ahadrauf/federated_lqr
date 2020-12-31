import numpy as np
from numpy.linalg import multi_dot, inv
from math import erf
from typing import List, Tuple


class LQR:
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, F: np.ndarray,
                 var_dyn: float, var_ctrl: float):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.F = F
        self.var_dyn = var_dyn
        self.var_ctrl = var_ctrl

    def generate_trajectory(self, x0: np.ndarray, N: int) -> Tuple[List[Tuple[float, float, float]],
                                                                   List[Tuple[np.ndarray, float]]]:
        """
        Generate trajectory (given dynamics and controller noise) starting at x0 and continuing for N time steps

        :param x0:
        :param N:
        :return: [(s_t, a_t, s_{t+1})], t = 0, ..., N-1
        """
        # x = x0
        P = self.F

        Ks = []
        Ps = []  # for metadata
        for t in reversed(range(N)):
            # x_T = np.expand_dims(x, 1)
            A_T = np.transpose(self.A)
            B_T = np.transpose(self.B)
            # J = 0.5*multi_dot([x_T, P, x])
            K = -multi_dot([inv(self.R+multi_dot([B_T, P, self.B])), B_T, P, self.A])
            # u = np.dot(K, x)
            P = self.Q + multi_dot([A_T, P, self.A]) - \
                multi_dot([multi_dot([A_T, P, self.B]),
                           inv(self.R + multi_dot([B_T, P, self.B])),
                           multi_dot([B_T, P, self.A])])

            Ks.insert(0, K)
            Ps.insert(0, P)

        traj = []
        metadata = []
        x = x0
        for t in range(N):
            u = np.dot(Ks[t], x)+np.random.normal(0, self.var_ctrl)
            x_new = np.dot(self.A, x)+np.dot(self.B, u)+np.random.normal(0, self.var_dyn)
            traj.append((x, u, x_new))

            # Calculate some metadata for postprocessing
            x_T = np.expand_dims(x, 1)
            J = 0.5*multi_dot([x_T, Ps[t], x])
            metadata.append((Ks[t], J))
            x = x_new

        return traj, metadata

    def prob_state_transition(self, state: Tuple[float, float, float], resolution: float = 0.1,
                              eps_dyn: float = 1e-30) -> float:
        """
        Calculates the probability of a state transition

        s_{t+1} ~ Normal(Ax + Bu, var_dynamics)
        p(s_{t+1} | s_t, a_t) = CDF(s_{t+1} + resolution/2) - CDF(s_{t+1} - resolution/2)
        :param state: Tuple of 3 items: (s_t, a_t, s_{t+1})
        :param resolution: The resolution used to calculate the CDF difference
        :return: probability of state transition i p(s_{t+1} | s_t, a_t)
        """
        x, u, x_new = state
        mu = np.dot(self.A, x)+np.dot(self.B, u)
        prob = self.cdf(x_new+resolution/2, mu, self.var_dyn)-self.cdf(x_new-resolution/2, mu, self.var_dyn)
        return max(prob, eps_dyn)

    def prob_action(self, state: Tuple[float, float, float], K: np.ndarray, resolution: float = 0.1,
                    eps_ctrl: float = 1e-30) -> float:
        """
        Calculates the probability of a state transition

        a_t ~ Normal(K_t*s_t, var_ctrl)
        p(a_t | s_t) = CDF(a_t + resolution/2) - CDF(a_t - resolution/2)
        :param state: Tuple of 3 items: (s_t, a_t, s_{t+1})
        :param K: control matrix
        :param resolution: The resolution used to calculate the CDF difference
        :return: probability of state transition i p(a_t | s_t)
        """
        x, u, x_new = state
        mu = np.dot(K, x)
        prob = self.cdf(u+resolution/2, mu, self.var_ctrl)-self.cdf(u-resolution/2, mu, self.var_ctrl)
        return max(prob, eps_ctrl)

    def prob_trajectory(self, traj: List[Tuple[float, float, float]], metadata: List[Tuple[np.ndarray, float]],
                        resolution_dyn: float = 0.1, resolution_ctrl: float = 0.1,
                        eps_dyn: float = 1e-30, eps_ctrl: float=1e-30) -> float:
        """
        Calculates the probability of a trajectory

        p(traj) = p(s0) * prod_{t=0 to N} p(a_t | s_t)*p(s_{t+1} | s_t, a_t)
        :param traj: List[Tuple[s_t, a_t, s_{t+1}]]
        :param metadata: List[Tuple[K_t, J_t]]
        :param resolution_dyn: Resolution of prob_state_transition
        :param resolution_ctrl: Resolution of prob_action
        :return: Probability of the trajectory
        """
        prob = 1  # p(s0)
        for state, metadata in zip(traj, metadata):
            prob *= self.prob_state_transition(state, resolution_dyn)
            prob *= self.prob_action(state, metadata[0], resolution_ctrl)
        return prob

    def log_prob_trajectory(self, traj: List[Tuple[float, float, float]], metadata: List[Tuple[np.ndarray, float]],
                            resolution_dyn: float = 0.1, resolution_ctrl: float = 0.1,
                            eps_dyn: float = 1e-30, eps_ctrl: float=1e-30) -> float:
        """
        Calculates the log probability of a trajectory

        p(traj) = log(p(s0)) + sum_{t=0 to N} log(p(a_t | s_t)) + log(p(s_{t+1} | s_t, a_t))
        :param traj: List[Tuple[s_t, a_t, s_{t+1}]]
        :param metadata: List[Tuple[K_t, J_t]]
        :param resolution_dyn: Resolution of prob_state_transition
        :param resolution_ctrl: Resolution of prob_action
        :return: Log robability of the trajectory
        """
        prob = np.log(1.0)  # p(s0)
        for state, metadata in zip(traj, metadata):
            prob += np.log(self.prob_state_transition(state, resolution_dyn, eps_dyn))
            prob += np.log(self.prob_action(state, metadata[0], resolution_ctrl, eps_ctrl))
        return prob

    @staticmethod
    def cdf(x, mu, var):
        # Taken from Wikipedia: https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
        return 0.5*(1+erf((x-mu)/(var*np.sqrt(2))))

    @staticmethod
    def extract_trajectory_from_transition_trajectory(traj: List[Tuple[float, float, float]]) -> List[float]:
        return [s[0] for s in traj] + [traj[-1][-1]]  # Add the final state to the end of the list
