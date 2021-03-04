import numpy as np
import cvxpy as cp
from lqr_infinite_horizon import LQR
from scipy.stats import wishart
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from utils import *
from ADMM_palan_v2 import *
import pickle
from datetime import datetime

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
    cov_dyn = 0.5*n*n*VQ
    cov_ctrl = 0.5*m*m*VR
    return LQR(A, B, Q, R, cov_dyn, cov_ctrl)


if __name__ == "__main__":
    n, m = 4, 2  # n = dimension of state space, m = # of inputs
    N = 10  # trajectory length
    M = 2  # number of robots
    Ntraj = 2  # number of trajectories we sample from each robot
    VQ = np.eye(n)/n/n  # covariance of Wishart distribution of Q
    VR = np.eye(m)/m/m  # covariance of Wishart distribution of R
    # x0 = np.reshape(mvn.rvs(np.zeros(n), .5*n*n*VQ), (n, 1))
    x0 = np.reshape(mvn.rvs(np.zeros(n), n*n*VQ), (n, 1))

    # Generate controllers
    controllers = []
    for _ in range(M):
        cont = initialize_LQR(n, m, VQ, VR)
        controllers.append(cont)

    # Print some stats
    avgQ = np.nanmean([cont.Q for cont in controllers], axis=0)
    avgR = np.nanmean([cont.R for cont in controllers], axis=0)
    print("Average Q and R vs. Expected")
    print("Q", avgQ, n*n*VQ)
    print("R", avgR, m*m*VR)
    deviation_Q = np.nanstd([cont.Q for cont in controllers], axis=0)
    deviation_R = np.nanstd([cont.R for cont in controllers], axis=0)
    print("Mean and Std of Q's:", np.mean(deviation_Q), np.std(deviation_Q))
    print("Mean and Std of R's:", np.mean(deviation_R), np.std(deviation_R))

    N_test = 10000
    cost_true = np.mean([cont.simulate(x0, N, add_noise=False)[2][1] for cont in controllers], axis=0)
    cost_noisy = np.mean([cont.simulate(x0, N, add_noise=True)[2][1] for cont in controllers], axis=0)
    print("Cost true: {}, cost noisy: {}".format(cost_true, cost_noisy))

    # For saving files
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H_%M_%S")

    # Where all the custom operations go
    costs_lr_vsN, std_costs_lr_vsN = [], []
    costs_admm_vsN, std_costs_admm_vsN = [], []
    # costs_fedadmmK_vsN, std_costs_fedadmmK_vsN = [], []
    costs_fedadmmQR_vsN, std_costs_fedadmmQR_vsN = [], []
    # lossQ_lr_vsN, lossR_lr_vsN, std_lossQ_lr_vsN, std_lossR_lr_vsN = [], [], [], []
    lossK_lr_vsN, std_lossK_lr_vsN = [], []
    lossK_admm_vsN, lossQ_admm_vsN, lossR_admm_vsN = [], [], []
    std_lossK_admm_vsN, std_lossQ_admm_vsN, std_lossR_admm_vsN = [], [], []
    # lossK_fedadmmK_vsN, std_lossK_fedadmmK_vsN = [], []
    lossK_fedadmmQR_vsN, lossQ_fedadmmQR_vsN, lossR_fedadmmQR_vsN = [], [], []
    std_lossK_fedadmmQR_vsN, std_lossQ_fedadmmQR_vsN, std_lossR_fedadmmQR_vsN = [], [], []

    out_admm_aggregate = []

    traj_range = range(1, Ntraj + 1)
    # Assume one seed (different for sampling and testing?)
    seed_range = np.arange(1, 3)
    seed = 1

    xs_aggregate = {i: [] for i in range(M)}
    us_aggregate = {i: [] for i in range(M)}
    for traj in traj_range:
        print("Traj # =", traj, end=" - ", flush=True)
        costs_lr = []
        costs_admm = []
        # costs_fedadmmK = []
        costs_fedadmmQR = []

        out_lr = []
        out_admm = []
        # out_fedaddmmK = []
        out_fedadmmQR = []

        # lossQ_lr, lossR_lr = [], []
        lossK_lr = []
        lossK_admm, lossQ_admm, lossR_admm = [], [], []
        # lossQ_fedadmmK, lossR_fedadmmK = [], []
        # lossK_fedadmmK = []
        lossK_fedadmmQR, lossQ_fedadmmQR, lossR_fedadmmQR = [], [], []

        # Get previous PQR as a starter for ADMM
        if traj > 1:
            prevP, prevQ, prevR = out_admm_aggregate[-1][1:]
        else:
            prevP, prevQ, prevR = None, None, None

        for m in range(M):
            print(m, end=", ", flush=True)
            cont = controllers[m]
            # x0 = np.random.randint(100, size=(n, 1))
            x0 = np.reshape(mvn.rvs(np.zeros(n), n*n*VQ), (n, 1))
            xs, us, metadata = cont.simulate(x0, N, seed=np.random.randint(0, 1e6), add_noise=True)
            xs_aggregate[m].append(xs)
            us_aggregate[m].append(us)
            if traj == 1:
                plt.plot(range(N + 1), [x[0, 0] for x in xs], label="Q={}, R={}".format(cont.Q[0, 0], cont.R[0, 0]))

            XS = np.hstack(xs[:-1])  # N x (n, 1) --> (n, N)
            US = np.hstack(us)  # N x (m, 1) --> (m, N)


            def L(K):
                return cp.sum_squares(K@XS - US)


            def L_lr(K, Q, R):
                return cp.sum_squares(K@XS - US)


            L_QR = lambda Q, R: cp.Constant(0.)

            r = lambda K: (0.01*cp.sum_squares(K), [])
            rQR = lambda Q, R: (cp.Constant(0.), [])
            r_lr = lambda K, Q, R: 0.01*cp.sum_squares(K)
            LK = lambda K: np.linalg.norm(K - cont.getK())
            LQ = lambda Q: np.linalg.norm(Q - cont.Q)
            LR = lambda R: np.linalg.norm(R - cont.R)

            xs_agg = np.vstack(xs_aggregate[m])
            us_agg = np.vstack(us_aggregate[m])
            Klr = policy_fitting(L_lr, r_lr, xs, us, cont.Q, cont.R)
            out_lr.append(Klr)
            lossK_lr.append(LK(Klr))

            Kadmm, Padmm, Qadmm, Radmm = policy_fitting_with_a_kalman_constraint_extra(L, L_QR, r, rQR, xs, us,
                                                                                       cont.A, cont.B,
                                                                                       niter=50)
            out_admm.append((traj, m, Kadmm, Padmm, Qadmm, Radmm))
            lossK_admm.append(LK(Kadmm))
            lossQ_admm.append(LQ(Qadmm))
            lossR_admm.append(LR(Radmm))


            def Lfed(K, Q, R):
                return cp.sum_squares(K@XS - US)
                # return cp.sum_squares(Q - cont.Q) + cp.sum_squares(R - cont.R)


            def Lfed_QR(K, Q, R):
                return cp.sum_squares(K@XS - US) + cp.sum_squares(Q - prevQ) + \
                       cp.sum_squares(R - prevR)
                # return cp.sum_squares(Q - cont.Q) + cp.sum_squares(R - cont.R)


            LfedK = lambda K: cp.sum_squares(K@XS - US)
            LfedQR2 = lambda Q, R: cp.sum_squares(Q - prevQ) + \
                                   cp.sum_squares(R - prevR)
            LfedQR1 = lambda Q, R: cp.Constant(0.)

            if traj == 1:
                KfedadmmQR, PfedadmmQR, QfedadmmQR, RfedadmmQR = \
                    policy_fitting_with_a_kalman_constraint_extra(LfedK, LfedQR1, r, rQR, xs, us, cont.A, cont.B,
                                                                  P0=prevP, Q0=prevQ, R0=prevR)
            else:
                KfedadmmQR, PfedadmmQR, QfedadmmQR, RfedadmmQR = \
                    policy_fitting_with_a_kalman_constraint_extra(LfedK, LfedQR2, r, rQR, xs, us, cont.A, cont.B,
                                                                  P0=prevP, Q0=prevQ, R0=prevR)

            # print(sum([np.linalg.norm(KfedadmmQR@x - u)**2 for x, u in zip(xs, us)]))
            # if traj > 1:
            #     print(np.linalg.norm(QfedadmmQR - cont.Q)**2)
            #     print(np.linalg.norm(RfedadmmQR - cont.R)**2)
            out_fedadmmQR.append((traj, m, KfedadmmQR, PfedadmmQR, QfedadmmQR, RfedadmmQR))
            lossK_fedadmmQR.append(LK(KfedadmmQR))
            lossQ_fedadmmQR.append(LQ(QfedadmmQR))
            lossR_fedadmmQR.append(LR(RfedadmmQR))

            x0 = np.reshape(mvn.rvs(np.zeros(n), cont.cov_dyn), (n, 1))
            for _ in range(100):  # For a little added robustness in the cost measurement
                seed = np.random.randint(0, 1e6)
                cost_lr = cont.simulate(x0, N, K=Klr, seed=seed, add_noise=True)[2][1]
                if np.linalg.norm(Qadmm) == np.inf:
                    cost_admm = np.nan
                else:
                    cost_admm = cont.simulate(x0, N, Q=Qadmm, R=Radmm, seed=seed, add_noise=True)[2][1]
                if np.linalg.norm(QfedadmmQR) == np.inf:
                    cost_fedadmmQR = np.nan
                else:
                    cost_fedadmmQR = cont.simulate(x0, N, Q=QfedadmmQR, R=RfedadmmQR, seed=seed, add_noise=True)[2][1]

                if np.isnan(cost_lr) or cost_lr > 1e4 or cost_lr == np.inf:
                    cost_lr = np.nan
                if np.isnan(cost_admm) or cost_admm > 1e5 or cost_admm == np.inf:
                    cost_admm = np.nan
                if np.isnan(cost_fedadmmQR) or cost_fedadmmQR > 1e5 or cost_fedadmmQR == np.inf:
                    cost_fedadmmQR = np.nan
                # Add the above costs to a list of costs
                costs_lr.append(cost_lr)
                costs_admm.append(cost_admm)
                costs_fedadmmQR.append(cost_fedadmmQR)

        Kavg = np.nanmean([K for _, _, K, P, Q, R in out_fedadmmQR[-M:]], axis=0)
        Pavg = np.nanmean([P for _, _, K, P, Q, R in out_fedadmmQR[-M:]], axis=0)
        Qavg = np.nanmean([Q for _, _, K, P, Q, R in out_fedadmmQR[-M:]], axis=0)
        Ravg = np.nanmean([R for _, _, K, P, Q, R in out_fedadmmQR[-M:]], axis=0)
        # out_fedaddmmK.append((traj, Kavg))
        out_admm_aggregate.append((traj, Pavg, Qavg, Ravg))

        # for i in range(M):
        #     cont = controllers[i]
        #     for _ in range(100):
        #         # cost_fedadmmK = cont.simulate(x0, N, K=Kavg, seed=0, add_noise=True)[2][1]
        #         cost_fedadmmQR = cont.simulate(x0, N, Q=Qavg, R=Ravg, seed=0, add_noise=True)[2][1]
        #         # costs_fedadmmK.append(cost_fedadmmK)
        #         costs_fedadmmQR.append(cost_fedadmmQR)
        #
        #     LK = lambda K: np.linalg.norm(K - cont.getK())
        #     LQ = lambda Q: np.linalg.norm(Q - cont.Q)
        #     LR = lambda R: np.linalg.norm(R - cont.R)
        #     # lossK_fedadmmK.append(LK(Kavg))
        #     lossK_fedadmmQR.append(LK(cont.getK(Qavg, Ravg)))
        #     lossQ_fedadmmQR.append(LQ(Qavg))
        #     lossR_fedadmmQR.append(LR(Ravg))

        costs_lr_vsN.append(np.nanmean(costs_lr))
        std_costs_lr_vsN.append(np.nanstd(costs_lr))
        costs_admm_vsN.append(np.nanmean(costs_admm))
        std_costs_admm_vsN.append(np.nanstd(costs_admm))
        # costs_fedadmmK_vsN.append(np.nanmean(costs_fedadmmK))
        # std_costs_fedadmmK_vsN.append(np.nanstd(costs_fedadmmK))
        costs_fedadmmQR_vsN.append(np.nanmean(costs_fedadmmQR))
        std_costs_fedadmmQR_vsN.append(np.nanstd(costs_fedadmmQR))

        lossK_lr_vsN.append(np.nanmean(lossK_lr))
        std_lossK_lr_vsN.append(np.nanstd(lossK_lr))

        lossK_admm_vsN.append(np.nanmean(lossK_admm))
        std_lossK_admm_vsN.append(np.nanstd(lossK_admm))
        lossQ_admm_vsN.append(np.nanmean(lossQ_admm))
        std_lossQ_admm_vsN.append(np.nanstd(lossQ_admm))
        lossR_admm_vsN.append(np.nanmean(lossR_admm))
        std_lossR_admm_vsN.append(np.nanstd(lossR_admm))

        # lossK_fedadmmK_vsN.append(np.nanmean(lossK_fedadmmK))
        # std_lossK_fedadmmK_vsN.append(np.nanstd(lossK_fedadmmK))

        lossK_fedadmmQR_vsN.append(np.nanmean(lossK_fedadmmQR))
        std_lossK_fedadmmQR_vsN.append(np.nanstd(lossK_fedadmmQR))
        # print(lossQ_fedadmmQR, np.nanmean(lossQ_fedadmmQR), np.nanstd(lossQ_fedadmmQR))
        lossQ_fedadmmQR_vsN.append(np.nanmean(lossQ_fedadmmQR))
        std_lossQ_fedadmmQR_vsN.append(np.nanstd(lossQ_fedadmmQR))
        lossR_fedadmmQR_vsN.append(np.nanmean(lossR_fedadmmQR))
        std_lossR_fedadmmQR_vsN.append(np.nanstd(lossR_fedadmmQR))

        out_format = "| %3.3f (%3.3f) | %3.3f (%3.3f) | %3.3f (%3.3f), " \
                     "Losses: K_lr: %3.3f (%3.3f), (KQR)_admm: %3.3f (%3.3f) | " \
                     "%3.3f (%3.3f) | %3.3f (%3.3f), (KQR)_fedadmm: %3.3f (%3.3f) | %3.3f (%3.3f) | %3.3f (%3.3f)"
        print(
            out_format%(
                costs_lr_vsN[-1], std_costs_lr_vsN[-1], costs_admm_vsN[-1], std_costs_admm_vsN[-1],
                costs_fedadmmQR_vsN[-1], std_costs_fedadmmQR_vsN[-1],
                lossK_lr_vsN[-1], std_lossK_lr_vsN[-1],
                lossK_admm_vsN[-1], std_lossK_admm_vsN[-1], lossQ_admm_vsN[-1], std_lossQ_admm_vsN[-1],
                lossR_admm_vsN[-1], std_lossR_admm_vsN[-1],
                lossK_fedadmmQR_vsN[-1], std_lossK_fedadmmQR_vsN[-1], lossQ_fedadmmQR_vsN[-1],
                std_lossQ_fedadmmQR_vsN[-1],
                lossR_fedadmmQR_vsN[-1], std_lossR_fedadmmQR_vsN[-1]), flush=True)

        np.save('data/' + timestamp + "_fedadmm_palan_v2.npy", [costs_lr_vsN, std_costs_lr_vsN,
                                                                costs_admm_vsN, std_costs_admm_vsN,
                                                                costs_fedadmmQR_vsN, std_costs_fedadmmQR_vsN,
                                                                lossK_lr_vsN,
                                                                lossK_admm_vsN, lossQ_admm_vsN, lossR_admm_vsN,
                                                                lossK_fedadmmQR_vsN, lossQ_fedadmmQR_vsN,
                                                                lossR_fedadmmQR_vsN])
        # np.save(timestamp + "costs_lr_vsW.npy", costs_lr_vsN)
        # np.save(timestamp + "costs_admm_vsW.npy", costs_admm_vsN)
        # # np.save(timestamp + "costs_fedadmmK_vsW.npy", costs_fedadmmK_vsN)
        # np.save(timestamp + "costs_fedadmmQR_vsW.npy", costs_fedadmmQR_vsN)
        # np.save(timestamp + "std_costs_lr_vsW.npy", std_costs_lr_vsN)
        # np.save(timestamp + "std_costs_admm_vsW.npy", std_costs_admm_vsN)
        # # np.save(timestamp + "std_costs_fedadmmK_vsW.npy", std_costs_fedadmmK_vsN)
        # np.save(timestamp + "std_costs_fedadmmQR_vsW.npy", std_costs_fedadmmQR_vsN)

        if traj == 1:
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title("Trajectories")
            plt.savefig("figures/" + timestamp + "_fedadmm_palan_v2_trajectories.png")
            plt.savefig("figures/" + timestamp + "_fedadmm_palan_v2_trajectories.pdf")
            # plt.show()

    # print(costs_lr)
    # print(costs_admm)
    # print(costs_admmQ)
    # print(costs_admmR)
    # plt.plot(W_range, costs_lr_vsN, label="Policy Learning")
    # plt.plot(W_range, costs_admm_vsN, label="ADMM")
    # plt.plot(W_range, costs_fedadmmK_vsN, label="FedADMM on K")
    # plt.plot(W_range, costs_fedadmmQR_vsN, label="FedADMM on QR")

    # Convert everything tp numpy arrays
    costs_lr_vsN = np.array(costs_lr_vsN)
    std_costs_lr_vsN = np.array(std_costs_lr_vsN)
    costs_admm_vsN = np.array(costs_admm_vsN)
    std_costs_admm_vsN = np.array(std_costs_admm_vsN)
    # costs_fedadmmK_vsN = np.array(costs_fedadmmK_vsN)
    # std_costs_fedadmmK_vsN = np.array(std_costs_fedadmmK_vsN)
    costs_fedadmmQR_vsN = np.array(costs_fedadmmQR_vsN)
    std_costs_fedadmmQR_vsN = np.array(std_costs_fedadmmQR_vsN)

    lossK_lr_vsN = np.array(lossK_lr_vsN)
    std_lossK_lr_vsN = np.array(std_lossK_lr_vsN)

    lossK_admm_vsN = np.array(lossK_admm_vsN)
    lossQ_admm_vsN = np.array(lossQ_admm_vsN)
    lossR_admm_vsN = np.array(lossR_admm_vsN)
    std_lossK_admm_vsN = np.array(std_lossK_admm_vsN)
    std_lossQ_admm_vsN = np.array(std_lossQ_admm_vsN)
    std_lossR_admm_vsN = np.array(std_lossR_admm_vsN)

    lossK_fedadmmQR_vsN = np.array(lossK_fedadmmQR_vsN)
    lossQ_fedadmmQR_vsN = np.array(lossQ_fedadmmQR_vsN)
    lossR_fedadmmQR_vsN = np.array(lossR_fedadmmQR_vsN)
    std_lossK_fedadmmQR_vsN = np.array(std_lossK_fedadmmQR_vsN)
    std_lossQ_fedadmmQR_vsN = np.array(std_lossQ_fedadmmQR_vsN)
    std_lossR_fedadmmQR_vsN = np.array(std_lossR_fedadmmQR_vsN)

    # Plot
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].axhline(cost_noisy, ls='--', c='k', label='expert (with noise)')
    axs[0, 0].axhline(cost_true, ls='-', c='k', label='optimal (without noise)')
    axs[0, 0].scatter(traj_range, costs_lr_vsN, s=4, marker='o', c='cyan', label='policy fitting')
    axs[0, 0].fill_between(traj_range, costs_lr_vsN - std_costs_lr_vsN/3, costs_lr_vsN + std_costs_lr_vsN/3, alpha=.5,
                           color='cyan')
    axs[0, 0].scatter(traj_range, costs_admm_vsN, s=4, marker='o', c='green', label='ADMM')
    axs[0, 0].fill_between(traj_range, costs_admm_vsN - std_costs_admm_vsN/3, costs_admm_vsN + std_costs_admm_vsN/3,
                           alpha=.5, color='green')
    # plt.scatter(traj_range, costs_fedadmmK_vsN, s=4, marker='o', c='red', label='FedADMM on K')
    # plt.fill_between(traj_range, costs_fedadmmK_vsN - std_costs_fedadmmK_vsN/3, costs_fedadmmK_vsN + std_costs_fedadmmK_vsN/3,
    #                  alpha=.5, color='red')
    axs[0, 0].scatter(traj_range, costs_fedadmmQR_vsN, s=4, marker='o', c='purple', label='FedADMM on Q, R')
    axs[0, 0].fill_between(traj_range, costs_fedadmmQR_vsN - std_costs_fedadmmQR_vsN/3,
                           costs_fedadmmQR_vsN + std_costs_fedadmmQR_vsN/3,
                           alpha=.5, color='purple')
    axs[0, 0].grid(True)
    axs[0, 0].set_xlabel(r"Trajectory Number \tau")
    axs[0, 0].set_ylabel(r'$L(\tau; \theta)$')
    axs[0, 0].set_title('Cost vs. Method, N=' + str(N) + ', M=' + str(M) + ', W~' + str(np.linalg.norm(controllers[
                                                                                                           0].cov_dyn)))
    axs[0, 0].legend()

    # Plot K
    axs[1, 0].scatter(traj_range, lossK_lr_vsN, s=4, marker='o', c='cyan', label='policy fitting')
    axs[1, 0].fill_between(traj_range, lossK_lr_vsN - std_lossK_lr_vsN/3, lossK_lr_vsN + std_lossK_lr_vsN/3, alpha=.5,
                           color='cyan')
    axs[1, 0].scatter(traj_range, lossK_admm_vsN, s=4, marker='o', c='green', label='ADMM')
    axs[1, 0].fill_between(traj_range, lossK_admm_vsN - std_lossK_admm_vsN/3, lossK_admm_vsN + std_lossK_admm_vsN/3,
                           alpha=.5, color='green')
    axs[1, 0].scatter(traj_range, lossK_fedadmmQR_vsN, s=4, marker='o', c='purple', label='FedADMM on Q, R')
    axs[1, 0].fill_between(traj_range, lossK_fedadmmQR_vsN - std_lossK_fedadmmQR_vsN/3,
                           lossK_fedadmmQR_vsN + std_lossK_fedadmmQR_vsN/3,
                           alpha=.5, color='purple')
    axs[1, 0].grid(True)
    axs[1, 0].set_xlabel(r"Trajectory Number $\tau$")
    axs[1, 0].set_ylabel(r'$||K - K_{true}||$')
    axs[1, 0].set_title('K Loss, N=' + str(N) + ', M=' + str(M))
    axs[1, 0].legend()

    # Plot Q Loss
    axs[0, 1].scatter(traj_range, lossQ_admm_vsN, s=4, marker='o', c='green', label='ADMM')
    axs[0, 1].fill_between(traj_range, lossQ_admm_vsN - std_lossQ_admm_vsN/3, lossQ_admm_vsN + std_lossQ_admm_vsN/3,
                           alpha=.5, color='green')
    axs[0, 1].scatter(traj_range, lossQ_fedadmmQR_vsN, s=4, marker='o', c='purple', label='FedADMM on Q, R')
    axs[0, 1].fill_between(traj_range, lossQ_fedadmmQR_vsN - std_lossQ_fedadmmQR_vsN/3,
                           lossQ_fedadmmQR_vsN + std_lossQ_fedadmmQR_vsN/3,
                           alpha=.5, color='purple')
    axs[0, 1].grid(True)
    axs[0, 1].set_xlabel(r'Trajectory Number $\tau$')
    axs[0, 1].set_ylabel(r'$||Q - Q_{true}||$')
    axs[0, 1].set_title('Q Loss, N=' + str(N) + ', M=' + str(M))
    axs[0, 1].legend()

    # Plot R Loss
    axs[1, 1].scatter(traj_range, lossR_admm_vsN, s=4, marker='o', c='green', label='ADMM')
    axs[1, 1].fill_between(traj_range, lossR_admm_vsN - std_lossR_admm_vsN/3, lossR_admm_vsN + std_lossR_admm_vsN/3,
                           alpha=.5, color='green')
    axs[1, 1].scatter(traj_range, lossR_fedadmmQR_vsN, s=4, marker='o', c='purple', label='FedADMM on Q, R')
    axs[1, 1].fill_between(traj_range, lossR_fedadmmQR_vsN - std_lossR_fedadmmQR_vsN/3,
                           lossR_fedadmmQR_vsN + std_lossR_fedadmmQR_vsN/3,
                           alpha=.5, color='purple')
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel(r'Trajectory Number $\tau$')
    axs[1, 1].set_ylabel(r'$||R - R_{true}||$')
    axs[1, 1].set_title('R Loss, N=' + str(N) + ', M=' + str(M))
    axs[1, 1].legend()

    np.save('data/' + timestamp + "_fedadmm_palan_v2.npy", [costs_lr_vsN, std_costs_lr_vsN,
                                                            costs_admm_vsN, std_costs_admm_vsN,
                                                            costs_fedadmmQR_vsN, std_costs_fedadmmQR_vsN,
                                                            lossK_lr_vsN, std_lossK_lr_vsN,
                                                            lossK_admm_vsN, std_lossK_admm_vsN,
                                                            lossQ_admm_vsN, std_lossQ_admm_vsN,
                                                            lossR_admm_vsN, std_lossR_admm_vsN,
                                                            lossK_fedadmmQR_vsN, std_lossK_fedadmmQR_vsN,
                                                            lossQ_fedadmmQR_vsN, std_lossQ_fedadmmQR_vsN,
                                                            lossR_fedadmmQR_vsN, std_lossR_fedadmmQR_vsN])
    # np.save(timestamp + "costs_lr_vsW.npy", costs_lr_vsN)
    # np.save(timestamp + "costs_admm_vsW.npy", costs_admm_vsN)
    # # np.save(timestamp + "costs_fedadmmK_vsW.npy", costs_fedadmmK_vsN)
    # np.save(timestamp + "costs_fedadmmQR_vsW.npy", costs_fedadmmQR_vsN)
    # np.save(timestamp + "std_costs_lr_vsW.npy", std_costs_lr_vsN)
    # np.save(timestamp + "std_costs_admm_vsW.npy", std_costs_admm_vsN)
    # # np.save(timestamp + "std_costs_fedadmmK_vsW.npy", std_costs_fedadmmK_vsN)
    # np.save(timestamp + "std_costs_fedadmmQR_vsW.npy", std_costs_fedadmmQR_vsN)

    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    # fig.tight_layout()
    fig.set_size_inches(32, 18)  # set figure's size manually to your full screen (32x18)
    plt.savefig("figures/" + timestamp + "_fedadmm_palan_v2.png", bbox_inches='tight')
    plt.savefig("figures/" + timestamp + "_fedadmm_palan_v2.pdf", bbox_inches='tight')
    print(timestamp)
    # plt.show()
