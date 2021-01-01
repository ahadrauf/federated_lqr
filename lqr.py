import numpy as np
from numpy.linalg import multi_dot, inv
from typing import List, Tuple
from scipy.stats import multivariate_normal as mvn
from scipy.special import erf


class LQR:
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, F: np.ndarray,
                 cov_dyn: np.ndarray, cov_ctrl: np.ndarray):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.F = F
        self.cov_dyn = cov_dyn
        self.cov_ctrl = cov_ctrl

    def generate_trajectory(self, x0: np.ndarray, N: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
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
            A_T = np.transpose(self.A)
            B_T = np.transpose(self.B)
            K = -multi_dot([inv(self.R + multi_dot([B_T, P, self.B])), B_T, P, self.A])
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
            # u = np.dot(Ks[t], x) + np.random.normal(0, self.cov_ctrl)
            # x_new = np.dot(self.A, x) + np.dot(self.B, u) + np.random.normal(0, self.cov_dyn)
            u_shape = np.shape(np.dot(Ks[t], x))
            u = mvn.rvs(np.dot(Ks[t], x), self.cov_ctrl)
            x_new = mvn.rvs(np.dot(self.A, x) + np.dot(self.B, u), self.cov_dyn)

            u = np.reshape(u, u_shape)
            x_new = np.reshape(x_new, np.shape(x))
            traj.append((x, u, x_new))

            # Calculate some metadata for postprocessing
            x_T = np.transpose(x)
            J = 0.5*multi_dot([x_T, Ps[t], x])
            metadata.append((Ks[t], J))
            x = x_new

        return traj, metadata

    def log_prob_state_transition(self, state: Tuple[np.ndarray, np.ndarray, np.ndarray], resolution: float = 0.1,
                                  eps_dyn: float = 1e-30) -> np.ndarray:
        """
        Calculates the probability of a state transition

        s_{t+1} ~ Normal(Ax + Bu, cov_dynamics)
        p(s_{t+1} | s_t, a_t) = CDF(s_{t+1} + resolution/2) - CDF(s_{t+1} - resolution/2)
        :param state: Tuple of 3 items: (s_t, a_t, s_{t+1})
        :param resolution: The resolution used to calculate the CDF difference
        :return: probability of state transition i p(s_{t+1} | s_t, a_t)
        """
        x, u, x_new = state
        mu = np.dot(self.A, x) + np.dot(self.B, u)
        prob = mvn.cdf(x_new + resolution/2, mu, self.cov_dyn) - mvn.cdf(x_new - resolution/2, mu, self.cov_dyn)
        # prob = self.log_normal_pdf(x_new, mu, self.cov_dyn)
        return np.log(np.maximum(prob, eps_dyn))

    def log_prob_action(self, state: Tuple[np.ndarray, np.ndarray, np.ndarray], K: np.ndarray, resolution: float = 0.1,
                        eps_ctrl: float = 1e-30) -> np.ndarray:
        """
        Calculates the probability of a state transition

        a_t ~ Normal(K_t*s_t, cov_ctrl)
        p(a_t | s_t) = CDF(a_t + resolution/2) - CDF(a_t - resolution/2)
        :param state: Tuple of 3 items: (s_t, a_t, s_{t+1})
        :param K: control matrix
        :param resolution: The resolution used to calculate the CDF difference
        :return: probability of state transition i p(a_t | s_t)
        """
        x, u, x_new = state
        mu = np.dot(K, x)
        prob = mvn.cdf(u + resolution/2, mu, self.cov_ctrl) - mvn.cdf(u - resolution/2, mu, self.cov_ctrl)
        # prob = self.log_normal_pdf(u, mu, self.cov_ctrl)
        return np.log(np.maximum(prob, eps_ctrl))

    # def prob_trajectory(self, traj: List[Tuple[float, float, float]], metadata: List[Tuple[np.ndarray, float]],
    #                     resolution_dyn: float = 0.1, resolution_ctrl: float = 0.1,
    #                     eps_dyn: float = 1e-30, eps_ctrl: float = 1e-30) -> float:
    #     """
    #     Calculates the probability of a trajectory
    #
    #     p(traj) = p(s0) * prod_{t=0 to N} p(a_t | s_t)*p(s_{t+1} | s_t, a_t)
    #     :param traj: List[Tuple[s_t, a_t, s_{t+1}]]
    #     :param metadata: List[Tuple[K_t, J_t]]
    #     :param resolution_dyn: Resolution of prob_state_transition
    #     :param resolution_ctrl: Resolution of prob_action
    #     :return: Probability of the trajectory
    #     """
    #     prob = 1  # p(s0)
    #     for state, metadata in zip(traj, metadata):
    #         prob *= self.prob_state_transition(state, resolution_dyn, eps_dyn)
    #         prob *= self.prob_action(state, metadata[0], resolution_ctrl, eps_ctrl)
    #     return prob

    def log_prob_trajectory(self, traj: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                            metadata: List[Tuple[np.ndarray, float]],
                            resolution_dyn: float = 0.1, resolution_ctrl: float = 0.1,
                            eps_dyn: np.ndarray = 1e-30, eps_ctrl: np.ndarray = 1e-30) -> np.ndarray:
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
            prob += self.log_prob_state_transition(state, resolution_dyn, eps_dyn)
            prob += self.log_prob_action(state, metadata[0], resolution_ctrl, eps_ctrl)
        return prob

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
    def extract_trajectory_from_transition_trajectory(traj: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> \
            List[np.ndarray]:
        return [s[0] for s in traj] + [traj[-1][-1]]  # Add the final state to the end of the list
