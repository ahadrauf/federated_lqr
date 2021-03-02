import numpy as np
import cvxpy as cp
from lqr_infinite_horizon import LQR
from scipy.stats import wishart
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from utils import *
from ADMM import *
import pickle
from datetime import datetime
import time

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
    N = 10  # trajectory length
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

    # For saving files
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H_%M_%S")

    N_test = 1000
    cost_true = np.mean([cont.simulate(x0, N, add_noise=False)[2][1] for cont in controllers], axis=0)
    cost_noisy = np.mean([cont.simulate(x0, N, add_noise=True)[2][1] for cont in controllers], axis=0)
    print("Cost true: {}, cost noisy: {}".format(cost_true, cost_noisy))

    # Where all the custom operations go
    costs_lr_vsN, std_costs_lr_vsN = [], []
    costs_admm_vsN, std_costs_admm_vsN = [], []
    costs_fedadmmK_vsN, std_costs_fedadmmK_vsN = [], []
    costs_fedadmmQR_vsN, std_costs_fedadmmQR_vsN = [], []
    niter_range = np.arange(1, 102, 5)
    seed_range = np.arange(1, 4)

    times = []
    for niter in niter_range:
        start_time = time.time()
        print("niter =", niter, end=" - ", flush=True)
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
                print(i, end=", ", flush=True)
                cont = controllers[i]
                xs, us, metadata = cont.simulate(x0, N, seed=seed, add_noise=True)
                plt.plot(range(N + 1), [x[0, 0] for x in xs], label="Q={}, R={}".format(cont.Q[0, 0], cont.R[0, 0]))

                L = lambda K, Q, R: sum(cp.sum_squares(K@x - u) for x, u in zip(xs, us))
                r = lambda K, Q, R: 0.01*cp.sum_squares(K)
                LQ = lambda Q: np.linalg.norm(Q - cont.Q)
                LR = lambda R: np.linalg.norm(R - cont.R)

                # temp_start_time = time.time()
                Klr = policy_fitting(L, r, xs, us, cont.Q, cont.R)
                # print("Time elapsed after LR: ", time.time() - temp_start_time)
                out_lr.append(Klr)
                # temp_start_time = time.time()
                Kadmm, Padmm, Qadmm, Radmm = policy_fitting_with_kalman_constraint(L, r, xs, us, cont.A, cont.B,
                                                                                   niter=100)
                # print("Time elapsed after ADMM: ", time.time() - temp_start_time)
                out_admm.append((niter, i, Kadmm, Padmm, Qadmm, Radmm))

                for _ in range(100):  # For a little added robustness in the cost measurement
                    cost_lr = cont.simulate(x0, N, K=Klr, seed=0, add_noise=True)[2][1]
                    xs, us, metadata = cont.simulate(x0, N, Q=Qadmm, R=Radmm, seed=0, add_noise=True)
                    cost_admm = metadata[1]
                    if np.isnan(cost_lr) or cost_lr > 1e5 or cost_lr == np.inf:
                        cost_lr = np.nan

                    costs_lr.append(cost_lr)
                    costs_admm.append(cost_admm)
                    costs_admmQ.append(LQ(Qadmm))
                    costs_admmR.append(LR(Radmm))

                # plt.plot(range(N + 1), [x[0, 0] for x in xs], label="Qadmm={}, Radmm={}".format(Qadmm[0, 0], Radmm[0, 0]))

            Kavg = sum([K for _, _, K, P, Q, R in out_admm[-M:]])/M
            Pavg = sum([P for _, _, K, P, Q, R in out_admm[-M:]])/M
            Qavg = sum([Q for _, _, K, P, Q, R in out_admm[-M:]])/M
            Ravg = sum([R for _, _, K, P, Q, R in out_admm[-M:]])/M

            for i in range(M):
                cont = controllers[i]
                for _ in range(100):
                    cost_fedadmmK = cont.simulate(x0, N, K=Kavg, seed=0, add_noise=True)[2][1]
                    cost_fedadmmQR = cont.simulate(x0, N, Q=Qavg, R=Ravg, seed=0, add_noise=True)[2][1]
                    costs_fedadmmK.append(cost_fedadmmK)
                    costs_fedadmmQR.append(cost_fedadmmQR)

        end_time = time.time()
        times.append((end_time - start_time)/len(seed_range)/M)

        costs_lr_vsN.append(np.nanmean(costs_lr))
        std_costs_lr_vsN.append(np.nanstd(costs_lr))
        costs_admm_vsN.append(np.nanmean(costs_admm))
        std_costs_admm_vsN.append(np.nanstd(costs_admm))
        costs_fedadmmK_vsN.append(np.nanmean(costs_fedadmmK))
        std_costs_fedadmmK_vsN.append(np.nanstd(costs_fedadmmK))
        costs_fedadmmQR_vsN.append(np.nanmean(costs_fedadmmQR))
        std_costs_fedadmmQR_vsN.append(np.nanstd(costs_fedadmmQR))
        print("| %3.3f | %3.3f | %3.3f | %3.3f (runtime/robot ~ %3.3f sec)"%(costs_lr_vsN[-1],
                                                                             costs_admm_vsN[-1],
                                                                             costs_fedadmmK_vsN[-1],
                                                                             costs_fedadmmQR_vsN[-1],
                                                                             times[-1]), flush=True)

        # np.save("costs_lr_vsW.npy", costs_lr_vsN)
        # np.save("costs_admm_vsW.npy", costs_admm_vsN)
        # np.save("costs_fedadmmK_vsW.npy", costs_fedadmmK_vsN)
        # np.save("costs_fedadmmQR_vsW.npy", costs_fedadmmQR_vsN)
        # np.save("std_costs_lr_vsW.npy", std_costs_lr_vsN)
        # np.save("std_costs_admm_vsW.npy", std_costs_admm_vsN)
        # np.save("std_costs_fedadmmK_vsW.npy", std_costs_fedadmmK_vsN)
        # np.save("std_costs_fedadmmQR_vsW.npy", std_costs_fedadmmQR_vsN)
        np.save(timestamp + "_test_fedadmm_vs_noise.npy", [costs_lr_vsN, std_costs_lr_vsN,
                                                           costs_admm_vsN, std_costs_admm_vsN,
                                                           costs_fedadmmK_vsN, std_costs_fedadmmK_vsN,
                                                           costs_fedadmmQR_vsN, std_costs_fedadmmQR_vsN])

    # print(costs_lr)
    # print(costs_admm)
    # print(costs_admmQ)
    # print(costs_admmR)
    # plt.plot(niter_range, costs_lr_vsN, label="Policy Learning")
    # plt.plot(niter_range, costs_admm_vsN, label="ADMM")
    # plt.plot(niter_range, costs_fedadmmK_vsN, label="FedADMM on K")
    # plt.plot(niter_range, costs_fedadmmQR_vsN, label="FedADMM on QR")

    plt.xlabel("t")
    plt.ylabel("x_0")
    plt.title("Trajectories")
    plt.show()

    costs_lr_vsN = np.array(costs_lr_vsN)
    std_costs_lr_vsN = np.array(std_costs_lr_vsN)
    costs_admm_vsN = np.array(costs_admm_vsN)
    std_costs_admm_vsN = np.array(std_costs_admm_vsN)
    costs_fedadmmK_vsN = np.array(costs_fedadmmK_vsN)
    std_costs_fedadmmK_vsN = np.array(std_costs_fedadmmK_vsN)
    costs_fedadmmQR_vsN = np.array(costs_fedadmmQR_vsN)
    std_costs_fedadmmQR_vsN = np.array(std_costs_fedadmmQR_vsN)

    # Plot
    fig, axs = plt.subplots(1, 2)
    axs[0].axhline(cost_noisy, ls='--', c='k', label='expert')
    axs[0].axhline(cost_true, ls='-', c='k', label='optimal')
    axs[0].scatter(niter_range, costs_lr_vsN, s=4, marker='o', c='cyan', label='policy fitting')
    axs[0].fill_between(niter_range, costs_lr_vsN - std_costs_lr_vsN/3, costs_lr_vsN + std_costs_lr_vsN/3, alpha=.5,
                        color='cyan')
    axs[0].scatter(niter_range, costs_admm_vsN, s=4, marker='o', c='green', label='ADMM')
    axs[0].fill_between(niter_range, costs_admm_vsN - std_costs_admm_vsN/3, costs_admm_vsN + std_costs_admm_vsN/3,
                        alpha=.5, color='green')
    axs[0].scatter(niter_range, costs_fedadmmK_vsN, s=4, marker='o', c='red', label='Average ADMM on K')
    axs[0].fill_between(niter_range, costs_fedadmmK_vsN - std_costs_fedadmmK_vsN/3,
                        costs_fedadmmK_vsN + std_costs_fedadmmK_vsN/3,
                        alpha=.5, color='red')
    axs[0].scatter(niter_range, costs_fedadmmQR_vsN, s=4, marker='o', c='purple', label='Average ADMM on Q, R')
    axs[0].fill_between(niter_range, costs_fedadmmQR_vsN - std_costs_fedadmmQR_vsN/3,
                        costs_fedadmmQR_vsN + std_costs_fedadmmQR_vsN/3,
                        alpha=.5, color='purple')
    axs[0].grid(True)
    axs[0].set_xlabel("niter")
    axs[0].set_ylabel(r'$L(\tau; \theta)$')
    axs[0].set_title('Cost vs. Method, N=' + str(N) + ', M=' + str(M))
    axs[0].legend()

    # Plot time elapsed
    axs[1].plot(niter_range, times)
    axs[1].grid(True)
    axs[1].set_xlabel("Number of Local ADMM Iterations, niter")
    axs[1].set_ylabel("Rough Time Elapsed Per Robot")
    axs[1].set_title("Time vs. niter")

    # np.save("costs_lr_vsW.npy", costs_lr_vsN)
    # np.save("costs_admm_vsW.npy", costs_admm_vsN)
    # np.save("costs_fedadmmK_vsW.npy", costs_fedadmmK_vsN)
    # np.save("costs_fedadmmQR_vsW.npy", costs_fedadmmQR_vsN)
    # np.save("std_costs_lr_vsW.npy", std_costs_lr_vsN)
    # np.save("std_costs_admm_vsW.npy", std_costs_admm_vsN)
    # np.save("std_costs_fedadmmK_vsW.npy", std_costs_fedadmmK_vsN)
    # np.save("std_costs_fedadmmQR_vsW.npy", std_costs_fedadmmQR_vsN)
    np.save(timestamp + "_test_fedadmm_vs_noise.npy", [costs_lr_vsN, std_costs_lr_vsN,
                                                       costs_admm_vsN, std_costs_admm_vsN,
                                                       costs_fedadmmK_vsN, std_costs_fedadmmK_vsN,
                                                       costs_fedadmmQR_vsN, std_costs_fedadmmQR_vsN])

    plt.savefig("figures/" + timestamp + "_fedadmm_vs_noise.png")
    plt.savefig("figures/" + timestamp + "_fedadmm_vs_noise.pdf")
    plt.show()
