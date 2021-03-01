import numpy as np
import cvxpy as cp
from lqr_infinite_horizon import LQR
from scipy.stats import wishart
import matplotlib.pyplot as plt
from utils import *
from ADMM import *

A = None
B = None


def initialize_LQR(n, m, VQ, VR, cacheAB=True):
    if cacheAB:
        global A, B
    else:
        A, B = None, None
    if A is None:
        A = np.random.randn(n, n)
        # A = A/np.abs(np.linalg.eig(A)[0]).max()
        # print(A)
    if B is None:
        B = np.random.randn(n, m)

    Q = np.reshape(wishart.rvs(n*n, VQ), (n, n))
    R = np.reshape(wishart.rvs(m*m, VR), (m, m))
    # Q = (Q + Q.T)/2  # Need to figure out a better distribution (Gaussian Reimann distribution?)
    # R = (R + R.T)/2
    cov_dyn = .5 * n*n*VQ
    cov_ctrl = .5 * m*m*VR
    return LQR(A, B, Q, R, cov_dyn, cov_ctrl)


if __name__ == "__main__":
    n, m = 4, 2  # n = dimension of state space, m = # of inputs
    N = 5  # trajectory length
    M = 10  # number of robots
    Ntraj = 1  # number of trajectories we sample from each robot
    VQ = 3*np.eye(n)  # covariance of Wishart distribution of Q
    VR = 4*np.eye(m)  # covariance of Wishart distribution of R
    x0 = np.random.randint(100, size=(n, 1))

    # Generate controllers
    controllers = []
    for _ in range(M):
        cont = initialize_LQR(n, m, VQ, VR)
        controllers.append(cont)

    # Print some stats
    avgQ = sum([cont.Q for cont in controllers])/M
    avgR = sum([cont.R for cont in controllers])/M
    print("Average Q and R vs. Expected")
    print("Q", avgQ, n*n*VQ)
    print("R", avgR, m*m*VR)

    N_test = 1000
    cost_true = np.mean([cont.simulate(x0, N, add_noise=False)[2][1] for cont in controllers], axis=0)
    cost_noisy = np.mean([cont.simulate(x0, N, add_noise=True)[2][1] for cont in controllers], axis=0)
    print("Cost true: {}, cost noisy: {}".format(cost_true, cost_noisy))

    # Where all the custom operations go
    costs_lr_vsN = []
    costs_admm_vsN = []
    costs_fedadmmK_vsN = []
    costs_fedadmmQR_vsN = []
    average = lambda lst: sum(lst)/len(lst)
    N_range = range(1, 50)
    for N in N_range:
        print("N =", N, end=" - ")
        costs_lr = []
        costs_admm = []
        costs_admmQ = []
        costs_admmR = []
        out_lr = []
        out_admm = []
        costs_fedadmmK = []
        costs_fedadmmQR = []
        for i in range(M):
            print(i, end=", ")
            cont = controllers[i]
            xs, us, metadata = cont.simulate(x0, N)
            # plt.plot(range(N + 1), [x[0, 0] for x in xs], label="Q={}, R={}".format(cont.Q[0, 0], cont.R[0, 0]))

            L = lambda K: sum(cp.sum_squares(K@x - u) for x, u in zip(xs, us))
            r = lambda K: 0.01 * cp.sum_squares(K)
            LQ = lambda Q: np.linalg.norm(Q - cont.Q)
            LR = lambda R: np.linalg.norm(R - cont.R)

            Klr = policy_fitting(L, r, xs, us)
            out_lr.append(Klr)
            Kadmm, Padmm, Qadmm, Radmm = policy_fitting_with_kalman_constraint(L, r, xs, us, cont.A, cont.B)
            out_admm.append((Kadmm, Padmm, Qadmm, Radmm))

            cost_lr = cont.simulate(x0, N, K=Klr, add_noise=False)[2][1]
            xs, us, metadata = cont.simulate(x0, N, Q=Qadmm, R=Radmm, add_noise=False)
            cost_admm = metadata[1]
            if np.isnan(cost_lr) or cost_lr > 1e5 or cost_lr == np.inf:
                cost_lr = np.nan

            costs_lr.append(cost_lr)
            costs_admm.append(cost_admm)
            costs_admmQ.append(LQ(Qadmm))
            costs_admmR.append(LR(Radmm))

            # plt.plot(range(N + 1), [x[0, 0] for x in xs], label="Qadmm={}, Radmm={}".format(Qadmm[0, 0], Radmm[0, 0]))

        Kavg = sum([K for K, _, _, _ in out_admm])/len(out_admm)
        Qavg = sum([Q for K, P, Q, R in out_admm])/len(out_admm)
        Ravg = sum([R for K, P, Q, R in out_admm])/len(out_admm)

        for i in range(M):
            cont = controllers[i]
            cost_fedadmmK = cont.simulate(x0, N, K=Kavg, add_noise=False)[2][1]
            cost_fedadmmQR = cont.simulate(x0, N, Q=Qavg, R=Ravg, add_noise=False)[2][1]
            costs_fedadmmK.append(cost_fedadmmK)
            costs_fedadmmQR.append(cost_fedadmmQR)

        costs_lr_vsN.append(average(costs_lr))
        costs_admm_vsN.append(average(costs_admm))
        costs_fedadmmK_vsN.append(average(costs_fedadmmK))
        costs_fedadmmQR_vsN.append(average(costs_fedadmmQR))
        print()

    # print(costs_lr)
    # print(costs_admm)
    # print(costs_admmQ)
    # print(costs_admmR)
    # plt.plot(range(M), costs_lr, label="Policy Learning")
    # plt.plot(range(M), costs_admm, label="ADMM")
    # plt.plot(range(M), costs_fedadmmK, label="FedADMM on K")
    # plt.plot(range(M), costs_fedadmmQR, label="FedADMM on QR")
    plt.plot(N_range, costs_lr_vsN, label="Policy Learning")
    plt.plot(N_range, costs_admm_vsN, label="ADMM")
    plt.plot(N_range, costs_fedadmmK_vsN, label="FedADMM on K")
    plt.plot(N_range, costs_fedadmmQR_vsN, label="FedADMM on QR")

    plt.grid(True)
    plt.xlabel("Trajectory Length N")
    plt.ylabel(r'$L(\tau; \theta)$')
    plt.title('Cost vs. Method, M=' + str(M))
    plt.legend()
    plt.show()
