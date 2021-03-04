import numpy as np
import cvxpy as cp
from lqr_infinite_horizon import LQR
from scipy.stats import wishart
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from utils import *
from ADMM import *
import pickle

A = None
B = None


def initialize_LQR(n, m, VQ, VR, cacheAB=True):
    if cacheAB:
        global A, B
    else:
        A, B = None, None
    if A is None:
        A = np.random.randn(n, n)
        A = A/np.abs(np.linalg.eig(A)[0]).max()
        # print(A)
    if B is None:
        B = np.random.randn(n, m)

    Q = np.reshape(wishart.rvs(n*n, VQ), (n, n))
    R = np.reshape(wishart.rvs(m*m, VR), (m, m))
    cov_dyn = .5*n*n*VQ
    cov_ctrl = .5*m*m*VR
    return LQR(A, B, Q, R, cov_dyn, cov_ctrl)


if __name__ == "__main__":
    n, m = 4, 2  # n = dimension of state space, m = # of inputs
    N = 5  # trajectory length
    M = 10  # number of robots
    Ntraj = 1  # number of trajectories we sample from each robot
    VQ = np.eye(n)/n/n  # covariance of Wishart distribution of Q
    VR = np.eye(m)/m/m  # covariance of Wishart distribution of R
    # x0 = np.random.randint(100, size=(n, 1))
    x0 = np.reshape(mvn.rvs(np.zeros(n), .5*n*n*VQ), (n, 1))

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
    deviation_Q = [(cont.Q - avgQ) for cont in controllers]
    deviation_R = [(cont.R - avgR) for cont in controllers]
    print("Mean and Std of Q's:", np.mean(deviation_Q), np.std(deviation_Q))
    print("Mean and Std of R's:", np.mean(deviation_R), np.std(deviation_R))

    N_test = 1000
    cost_true = np.mean([cont.simulate(x0, N, add_noise=False)[2][1] for cont in controllers], axis=0)
    cost_noisy = np.mean([cont.simulate(x0, N, add_noise=True)[2][1] for cont in controllers], axis=0)
    print("Cost true: {}, cost noisy: {}".format(cost_true, cost_noisy))

    # Where all the custom operations go
    costs_lr_vsN, std_costs_lr_vsN = [], []
    costs_admm_vsN, std_costs_admm_vsN = [], []
    costs_fedadmmK_vsN, std_costs_fedadmmK_vsN = [], []
    costs_fedadmmQR_vsN, std_costs_fedadmmQR_vsN = [], []
    N_range = np.arange(1, 51)
    seed_range = np.arange(1, 6)
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

        for seed in seed_range:
            for i in range(M):
                print(i, end=", ")
                cont = controllers[i]
                xs, us, metadata = cont.simulate(x0, N, seed=seed, add_noise=True)
                # plt.plot(range(N + 1), [x[0, 0] for x in xs], label="Q={}, R={}".format(cont.Q[0, 0], cont.R[0, 0]))

                L = lambda K, Q, R: sum(cp.sum_squares(K@x - u) for x, u in zip(xs, us))
                r = lambda K, Q, R: 0.01*cp.sum_squares(K)
                LQ = lambda Q: np.linalg.norm(Q - cont.Q)
                LR = lambda R: np.linalg.norm(R - cont.R)

                Klr = policy_fitting(L, r, xs, us, cont.Q, cont.R)
                out_lr.append(Klr)
                Kadmm, Padmm, Qadmm, Radmm = policy_fitting_with_kalman_constraint(L, r, xs, us, cont.A, cont.B)
                out_admm.append((Kadmm, Padmm, Qadmm, Radmm))

                for _ in range(3):  # For a little added robustness in the cost measurement
                    cost_lr = cont.simulate(x0, N, K=Klr, seed=0, add_noise=True)[2][1]
                    xs, us, metadata = cont.simulate(x0, N, Q=Qadmm, R=Radmm, seed=0, add_noise=True)
                    cost_admm = metadata[1]
                    if np.isnan(cost_lr) or cost_lr > 1e6 or cost_lr == np.inf:
                        cost_lr = np.nan

                    costs_lr.append(cost_lr)
                    costs_admm.append(cost_admm)
                    costs_admmQ.append(LQ(Qadmm))
                    costs_admmR.append(LR(Radmm))

                # plt.plot(range(N + 1), [x[0, 0] for x in xs], label="Qadmm={}, Radmm={}".format(Qadmm[0, 0], Radmm[0, 0]))

            Kavg = sum([K for K, P, Q, R in out_admm])/len(out_admm)
            Qavg = sum([Q for K, P, Q, R in out_admm])/len(out_admm)
            Ravg = sum([R for K, P, Q, R in out_admm])/len(out_admm)

            for i in range(M):
                cont = controllers[i]
                for _ in range(3):
                    cost_fedadmmK = cont.simulate(x0, N, K=Kavg, seed=0, add_noise=True)[2][1]
                    cost_fedadmmQR = cont.simulate(x0, N, Q=Qavg, R=Ravg, seed=0, add_noise=True)[2][1]
                    costs_fedadmmK.append(cost_fedadmmK)
                    costs_fedadmmQR.append(cost_fedadmmQR)

        costs_lr_vsN.append(np.nanmean(costs_lr))
        std_costs_lr_vsN.append(np.nanstd(costs_lr))
        costs_admm_vsN.append(np.nanmean(costs_admm))
        std_costs_admm_vsN.append(np.nanstd(costs_admm))
        costs_fedadmmK_vsN.append(np.nanmean(costs_fedadmmK))
        std_costs_fedadmmK_vsN.append(np.nanstd(costs_fedadmmK))
        costs_fedadmmQR_vsN.append(np.nanmean(costs_fedadmmQR))
        std_costs_fedadmmQR_vsN.append(np.nanstd(costs_fedadmmQR))
        print("| %3.3f | %3.3f | %3.3f | %3.3f"%(costs_lr_vsN[-1], costs_admm_vsN[-1], costs_fedadmmK_vsN[-1],
                                                 costs_fedadmmQR_vsN[-1]))

        np.save("costs_lr_vsN.npy", costs_lr_vsN)
        np.save("costs_admm_vsN.npy", costs_admm_vsN)
        np.save("costs_fedadmmK_vsN.npy", costs_fedadmmK_vsN)
        np.save("costs_fedadmmQR_vsN.npy", costs_fedadmmQR_vsN)
        np.save("std_costs_lr_vsN.npy", std_costs_lr_vsN)
        np.save("std_costs_admm_vsN.npy", std_costs_admm_vsN)
        np.save("std_costs_fedadmmK_vsN.npy", std_costs_fedadmmK_vsN)
        np.save("std_costs_fedadmmQR_vsN.npy", std_costs_fedadmmQR_vsN)

    # print(costs_lr)
    # print(costs_admm)
    # print(costs_admmQ)
    # print(costs_admmR)
    # plt.plot(N_range, costs_lr_vsN, label="Policy Learning")
    # plt.plot(N_range, costs_admm_vsN, label="ADMM")
    # plt.plot(N_range, costs_fedadmmK_vsN, label="FedADMM on K")
    # plt.plot(N_range, costs_fedadmmQR_vsN, label="FedADMM on QR")

    costs_lr_vsN = np.array(costs_lr_vsN)
    std_costs_lr_vsN = np.array(std_costs_lr_vsN)
    costs_admm_vsN = np.array(costs_admm_vsN)
    std_costs_admm_vsN = np.array(std_costs_admm_vsN)
    costs_fedadmmK_vsN = np.array(costs_fedadmmK_vsN)
    std_costs_fedadmmK_vsN = np.array(std_costs_fedadmmK_vsN)
    costs_fedadmmQR_vsN = np.array(costs_fedadmmQR_vsN)
    std_costs_fedadmmQR_vsN = np.array(std_costs_fedadmmQR_vsN)

    plt.axhline(cost_noisy, ls='--', c='k', label='Optimal (no noise)')
    plt.axhline(cost_true, ls='-', c='k', label='Optimal (with noise)')
    plt.scatter(N_range, costs_lr_vsN, s=4, marker='o', c='cyan', label='policy fitting')
    plt.fill_between(N_range, costs_lr_vsN - std_costs_lr_vsN/3, costs_lr_vsN + std_costs_lr_vsN/3, alpha=.5,
                     color='cyan')
    plt.scatter(N_range, costs_admm_vsN, s=4, marker='o', c='green', label='ADMM')
    plt.fill_between(N_range, costs_admm_vsN - std_costs_admm_vsN/3, costs_admm_vsN + std_costs_admm_vsN/3,
                     alpha=.5, color='green')
    plt.scatter(N_range, costs_fedadmmK_vsN, s=4, marker='o', c='red', label='FedADMM on K')
    plt.fill_between(N_range, costs_fedadmmK_vsN - std_costs_fedadmmK_vsN/3,
                     costs_fedadmmK_vsN + std_costs_fedadmmK_vsN/3,
                     alpha=.5, color='red')
    plt.scatter(N_range, costs_fedadmmQR_vsN, s=4, marker='o', c='purple', label='FedADMM on Q, R')
    plt.fill_between(N_range, costs_fedadmmQR_vsN - std_costs_fedadmmQR_vsN/3,
                     costs_fedadmmQR_vsN + std_costs_fedadmmQR_vsN/3,
                     alpha=.5, color='purple')

    np.save("costs_lr_vsN.npy", costs_lr_vsN)
    np.save("costs_admm_vsN.npy", costs_admm_vsN)
    np.save("costs_fedadmmK_vsN.npy", costs_fedadmmK_vsN)
    np.save("costs_fedadmmQR_vsN.npy", costs_fedadmmQR_vsN)
    np.save("std_costs_lr_vsN.npy", std_costs_lr_vsN)
    np.save("std_costs_admm_vsN.npy", std_costs_admm_vsN)
    np.save("std_costs_fedadmmK_vsN.npy", std_costs_fedadmmK_vsN)
    np.save("std_costs_fedadmmQR_vsN.npy", std_costs_fedadmmQR_vsN)

    plt.grid(True)
    plt.xlabel("Trajectory Length N")
    plt.ylabel(r'$L(\tau; \theta)$')
    plt.title('Cost vs. Method, M=' + str(M))
    plt.legend()
    plt.show()
